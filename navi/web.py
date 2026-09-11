"""インターネットからの自己学習。

過剰アクセスをしないための多重の歯止め:
  1. 全体 / ホスト別の最小リクエスト間隔 (既定 4 秒 / 8 秒)
  2. 時間あたり・1 日あたりのリクエスト上限 (既定 12/時, 120/日) を
     プロセス再起動をまたいで永続化
  3. 429 / 503 は Retry-After を尊重し、指数バックオフ
  4. 連続失敗でサーキットブレーカ (既定 30 分の完全停止)
  5. robots.txt の尊重 (公開 API エンドポイントを除く。理由は README 参照)
  6. 応答サイズ上限・タイムアウト・1 回の取得で学習する文数の上限
  7. 声の保護: Web 由来トークンがオペレーター由来を大きく超えたら学習を休む

「知識」はネットから得るが、「喋り方」はオペレーターからしか学ばない。
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .config import FLAG_STUDIED, ORIGIN_OPERATOR, ORIGIN_WEB, Config
from .jp import split_sentences

# 公開 API エンドポイント(robots.txt はクローラ向けであり、
# これらは「クライアントからの利用」を前提に提供されているため別扱いにする。
# その代わりレート制限は本物のクローラより厳しく設定している)
API_PREFIXES = (
    "https://ja.wikipedia.org/w/api.php",
    "https://ja.wikipedia.org/api/rest_v1/",
    "https://ja.wiktionary.org/w/api.php",
)

_JP_RE = re.compile(r"[ぁ-んァ-ヶ一-龥]")


class WebBlocked(Exception):
    """レート制限・予算・ブレーカにより、いま取得してはいけない。"""


@dataclass
class Fetched:
    """取得しただけの状態。まだ脳には入っていない。"""
    ok: bool
    topic: str = ""
    title: str = ""
    text: str = ""
    reason: str = ""


@dataclass
class StudyResult:
    ok: bool
    topic: str = ""
    title: str = ""
    sentences: int = 0
    new_words: int = 0
    excerpt: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# 予算(永続カウンタ)
# ---------------------------------------------------------------------------
class Budget:
    def __init__(self, cfg: Config, path: str):
        self.cfg = cfg
        self.path = path
        self.hour_start = 0.0
        self.hour_count = 0
        self.day_start = 0.0
        self.day_count = 0
        self.fail_streak = 0
        self.blocked_until = 0.0
        self.total = 0
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for k in ("hour_start", "hour_count", "day_start", "day_count",
                      "fail_streak", "blocked_until", "total"):
                if k in d:
                    setattr(self, k, d[k])
        except Exception:
            pass

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({k: getattr(self, k) for k in
                           ("hour_start", "hour_count", "day_start", "day_count",
                            "fail_streak", "blocked_until", "total")}, f)
        except Exception:
            pass

    def _roll(self, now: float) -> None:
        if now - self.hour_start >= 3600:
            self.hour_start, self.hour_count = now, 0
        if now - self.day_start >= 86400:
            self.day_start, self.day_count = now, 0

    def check(self) -> Tuple[bool, str]:
        now = time.time()
        self._roll(now)
        if now < self.blocked_until:
            left = int(self.blocked_until - now)
            return False, f"サーキットブレーカ作動中(あと{left}秒)"
        if self.hour_count >= self.cfg.web_max_per_hour:
            return False, f"時間あたり上限({self.cfg.web_max_per_hour}件)に到達"
        if self.day_count >= self.cfg.web_max_per_day:
            return False, f"1日あたり上限({self.cfg.web_max_per_day}件)に到達"
        return True, ""

    def consume(self) -> None:
        now = time.time()
        self._roll(now)
        self.hour_count += 1
        self.day_count += 1
        self.total += 1
        self.save()

    def on_success(self) -> None:
        self.fail_streak = 0
        self.save()

    def on_failure(self, retry_after: float = 0.0) -> None:
        self.fail_streak += 1
        now = time.time()
        if retry_after > 0:
            self.blocked_until = max(self.blocked_until, now + min(retry_after, 3600))
        if self.fail_streak >= self.cfg.web_max_failures:
            self.blocked_until = max(self.blocked_until,
                                     now + self.cfg.web_breaker_cooldown_sec)
        else:
            self.blocked_until = max(self.blocked_until, now + 2.0 ** self.fail_streak)
        self.save()

    def status(self) -> str:
        now = time.time()
        self._roll(now)
        s = f"{self.hour_count}/{self.cfg.web_max_per_hour}件(時) " \
            f"{self.day_count}/{self.cfg.web_max_per_day}件(日) 累計{self.total}件"
        if now < self.blocked_until:
            s += f" ※停止中 あと{int(self.blocked_until - now)}秒"
        return s


# ---------------------------------------------------------------------------
# 礼儀正しい取得
# ---------------------------------------------------------------------------
class PoliteFetcher:
    def __init__(self, cfg: Config, budget: Budget,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.cfg = cfg
        self.budget = budget
        self.should_stop = should_stop or (lambda: False)
        self._last_global = 0.0
        self._last_host: Dict[str, float] = {}
        self._robots: Dict[str, Tuple[float, Optional[urllib.robotparser.RobotFileParser]]] = {}

    # -- 待機 ----------------------------------------------------------
    def _sleep(self, sec: float) -> bool:
        end = time.time() + sec
        while time.time() < end:
            if self.should_stop():
                return False
            time.sleep(min(0.25, end - time.time()))
        return True

    def _throttle(self, host: str) -> bool:
        now = time.time()
        wait = max(self.cfg.web_min_interval_sec - (now - self._last_global),
                   self.cfg.web_host_interval_sec - (now - self._last_host.get(host, 0.0)))
        wait += random.uniform(0.0, 0.6)  # ジッタ(同時刻への集中を避ける)
        if wait > 0 and not self._sleep(wait):
            return False
        self._last_global = time.time()
        self._last_host[host] = self._last_global
        return True

    # -- robots.txt ----------------------------------------------------
    def _robots_ok(self, url: str) -> bool:
        if not self.cfg.web_respect_robots:
            return True
        if url.startswith(API_PREFIXES):
            return True
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        cached = self._robots.get(base)
        if cached and time.time() - cached[0] < 86400:
            rp = cached[1]
        else:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None  # 取得できない場合は保守的に許可しない
            self._robots[base] = (time.time(), rp)
        if rp is None:
            return False
        return rp.can_fetch(self.cfg.web_user_agent, url)

    # -- 取得 ----------------------------------------------------------
    def get(self, url: str, accept: str = "application/json") -> bytes:
        ok, reason = self.budget.check()
        if not ok:
            raise WebBlocked(reason)
        if not self._robots_ok(url):
            raise WebBlocked("robots.txt により許可されていません")
        host = urllib.parse.urlsplit(url).netloc
        if not self._throttle(host):
            raise WebBlocked("停止要求")

        req = urllib.request.Request(url, headers={
            "User-Agent": self.cfg.web_user_agent,
            "Accept": accept,
            "Accept-Encoding": "gzip",
            "Accept-Language": "ja",
        })
        self.budget.consume()
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.web_timeout_sec) as resp:
                raw = resp.read(self.cfg.web_max_bytes + 1)
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                if len(raw) > self.cfg.web_max_bytes:
                    raw = raw[: self.cfg.web_max_bytes]
            self.budget.on_success()
            return raw
        except urllib.error.HTTPError as e:
            retry = 0.0
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra:
                try:
                    retry = float(ra)
                except ValueError:
                    retry = 60.0
            elif e.code in (429, 503):
                retry = 120.0
            self.budget.on_failure(retry)
            raise WebBlocked(f"HTTP {e.code}") from e
        except Exception as e:
            self.budget.on_failure()
            raise WebBlocked(str(e)) from e


# ---------------------------------------------------------------------------
# 情報源
# ---------------------------------------------------------------------------
class WikipediaSource:
    """1 トピック = 1 リクエストで済むよう generator+extracts を使う。"""

    API = "https://ja.wikipedia.org/w/api.php"
    RANDOM = "https://ja.wikipedia.org/api/rest_v1/page/random/summary"

    def __init__(self, fetcher: PoliteFetcher):
        self.f = fetcher

    def lookup(self, topic: str) -> Tuple[str, str]:
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "formatversion": "2",
            "prop": "extracts", "exintro": "1", "explaintext": "1",
            "redirects": "1", "generator": "search",
            "gsrsearch": topic, "gsrlimit": "1", "gsrnamespace": "0",
        })
        raw = self.f.get(f"{self.API}?{q}")
        data = json.loads(raw.decode("utf-8", "replace"))
        pages = (data.get("query") or {}).get("pages") or []
        if not pages:
            return "", ""
        p = pages[0]
        return p.get("title", ""), p.get("extract", "") or ""

    def random(self) -> Tuple[str, str]:
        raw = self.f.get(self.RANDOM)
        d = json.loads(raw.decode("utf-8", "replace"))
        return d.get("title", ""), d.get("extract", "") or ""


class RSSSource:
    """任意の RSS/Atom。config.json の web_rss_feeds に URL を足すと有効になる。"""

    def __init__(self, fetcher: PoliteFetcher, feeds: List[str]):
        self.f = fetcher
        self.feeds = list(feeds)

    def fetch_one(self) -> Tuple[str, str]:
        if not self.feeds:
            return "", ""
        url = random.choice(self.feeds)
        raw = self.f.get(url, accept="application/rss+xml, application/xml, text/xml")
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(raw.decode("utf-8", "replace"))
        except ET.ParseError:
            return "", ""
        texts: List[str] = []
        title = ""
        for tag in (".//item", ".//{http://www.w3.org/2005/Atom}entry"):
            for item in root.findall(tag):
                t = item.findtext("title") or item.findtext(
                    "{http://www.w3.org/2005/Atom}title") or ""
                d = item.findtext("description") or item.findtext(
                    "{http://www.w3.org/2005/Atom}summary") or ""
                if t and not title:
                    title = t
                if t:
                    texts.append(t)
                if d:
                    texts.append(re.sub(r"<[^>]+>", " ", d))
        return title, " ".join(texts[:20])


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------
class WebLearner:
    def __init__(self, cfg: Config, brain, mood,
                 should_stop: Optional[Callable[[], bool]] = None):
        self.cfg = cfg
        self.brain = brain
        self.mood = mood
        self.budget = Budget(cfg, cfg.file("web_budget.json"))
        self.fetcher = PoliteFetcher(cfg, self.budget, should_stop)
        self.wiki = WikipediaSource(self.fetcher)
        self.rss = RSSSource(self.fetcher, cfg.web_rss_feeds)
        self.studied: Dict[str, float] = {}   # topic -> 最後に調べた時刻
        self.last_study = 0.0
        self.log: List[str] = []

    # -- 状態 ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {"studied": self.studied, "last_study": self.last_study,
                "log": self.log[-50:]}

    def load(self, d: dict) -> None:
        self.studied = {str(k): float(v) for k, v in (d.get("studied") or {}).items()}
        self.last_study = float(d.get("last_study", 0.0))
        self.log = list(d.get("log") or [])

    # -- 声の保護 ------------------------------------------------------
    def voice_is_safe(self) -> bool:
        """Web の言葉ばかりになっていないか。"""
        own = self.brain.total_tokens - self.brain.web_tokens
        return self.brain.web_tokens <= 3 * own + 200

    def due(self) -> bool:
        return time.time() - self.last_study >= self.cfg.web_study_interval_sec

    # -- 好奇心 --------------------------------------------------------
    def pick_topic(self) -> Optional[str]:
        """オペレーターがよく口にするのに、自分がよく知らない言葉を選ぶ。"""
        b = self.brain
        now = time.time()
        best, best_score = None, -1e9
        imp = b.importance()
        for wid in range(3, b.vocab_size):
            if not b.is_content_id(wid):
                continue
            if b.origin_of(wid) != ORIGIN_OPERATOR:
                continue
            word = b.id2word[wid]
            if len(word) < 2:
                continue
            last = self.studied.get(word)
            if last and now - last < 7 * 86400:
                continue
            score = 2.2 * float(imp[wid].item()) - 0.75 * b.knowledge_score(wid)
            score += random.uniform(0.0, 0.35)
            if score > best_score:
                best, best_score = word, score
        return best

    # -- 本体 ----------------------------------------------------------
    # 取得(ネットワーク待ち)と、脳への書き込みを明確に分ける。
    # 常駐スレッドが調べ物をしている最中もオペレーターは会話を続けられるよう、
    # 呼び出し側は fetch() をロックの外で、absorb() をロックの中で呼ぶ。
    def can_study(self) -> Tuple[bool, str]:
        if not self.cfg.web_enabled:
            return False, "ネット学習が無効です(/web on で有効化)"
        if not self.voice_is_safe():
            return False, "ネット由来の語が増えすぎたので休憩中"
        return self.budget.check()

    def fetch(self, topic: Optional[str]) -> Fetched:
        """ネットから文章を取ってくるだけ。脳には一切触れない。"""
        explicit = topic is not None
        try:
            if topic:
                title, text = self.wiki.lookup(topic)
            else:
                title, text = self.wiki.random()
                topic = title
            if not text and self.rss.feeds:
                title, text = self.rss.fetch_one()
                topic = topic or title
        except WebBlocked as e:
            self.last_study = time.time()
            return Fetched(False, topic=topic or "", reason=str(e))
        finally:
            self.last_study = time.time()

        if not text:
            return Fetched(False, topic=topic or "", title=title,
                           reason="有用な文章が見つからず")
        return Fetched(True, topic=topic or "", title=title, text=text,
                       reason="" if explicit else "好奇心")

    def absorb(self, f: Fetched) -> StudyResult:
        """取ってきた文章を脳に流し込む。呼び出し側でロックすること。"""
        if not f.ok:
            if f.topic:
                self.studied[f.topic] = time.time()
            return StudyResult(False, topic=f.topic, title=f.title, reason=f.reason)

        sentences = [s for s in split_sentences(f.text) if self._useful(s)]
        sentences = sentences[: self.cfg.web_sentences_per_fetch]
        if not sentences:
            self.studied[f.topic] = time.time()
            return StudyResult(False, topic=f.topic, title=f.title,
                               reason="日本語の文が取れず")

        before = self.brain.vocab_size
        all_keys: List[int] = []
        for line in sentences:
            _, keys, _ = self.brain.learn_text(line, self.cfg.w_web, ORIGIN_WEB)
            all_keys.extend(keys)

        # トピック語と、拾ってきた内容語を連想で結ぶ
        tid = self.brain.word2id.get(f.topic)
        if tid is not None:
            self.brain.set_flag(tid, FLAG_STUDIED, True)
            self.brain.associate([tid], all_keys, self.cfg.w_web)
            self.brain.associate(all_keys, [tid], self.cfg.w_web * 0.5)

        self.studied[f.topic] = time.time()
        new_words = self.brain.vocab_size - before
        self.mood.on_study(len(all_keys))
        self.log.append(f"{time.strftime('%m/%d %H:%M')} {f.topic} <- {f.title} "
                        f"({len(sentences)}文/新語{new_words})")
        del self.log[:-200]
        return StudyResult(True, topic=f.topic, title=f.title,
                           sentences=len(sentences), new_words=new_words,
                           excerpt=sentences[0][:60], reason=f.reason)

    def study(self, topic: Optional[str] = None) -> StudyResult:
        """can_study -> fetch -> absorb をまとめて行う(単一スレッド用)。"""
        ok, reason = self.can_study()
        if not ok:
            return StudyResult(False, topic=topic or "", reason=reason)
        return self.absorb(self.fetch(topic or self.pick_topic()))

    @staticmethod
    def _useful(s: str) -> bool:
        if not (12 <= len(s) <= 140):
            return False
        jp = len(_JP_RE.findall(s))
        return jp / len(s) >= 0.4
