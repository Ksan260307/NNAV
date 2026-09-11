"""機能語の自己発見。

このプロダクトで唯一「教えていないのに知っている」語が、疑問詞・人称・指示詞の
52 語でした。ここではそれを **分布から見つけ直せるか** を実装します。
種(組み込みリスト)は `bootstrap_function_words: false` で丸ごと外せるので、
「消して、何ターンで自力で見つけ直すか」を測れます (tools/rediscover.py)。

手がかりは 3 つとも観測できます。

  疑問詞  … ？で終わる文に偏り、しかも **そのスロットが後から別の値で埋まる**
            「名前は何？」→「名前はロックマン」 = 何 は値ではなく穴だった
  人称    … 同じ述語の別の項として同時に現れる代名詞どうしは別の実体。
            その排他グラフを 2 彩色し、問いかけに多く出るほうを聞き手とする
  指示詞  … 出現回数の割に、あらゆる格スロットに散らばる。
            意味ベクトルも文脈の平均になるので凝集しない
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

from .parse import FunctionWords, Pred

MIN_COUNT = 6           # これだけ見ていない語は判定しない
MIN_PRONOUN = 8
Q_THRESHOLD = 0.50
D_THRESHOLD = 0.62
OPEN_SLOT_MEMORY = 40
OWNER_MIN = 0.12


class Lexicon:
    def __init__(self) -> None:
        # --- 疑問詞 -------------------------------------------------------
        self.in_ask: Dict[str, float] = defaultdict(float)
        self.in_plain: Dict[str, float] = defaultdict(float)
        self.slot_seen: Dict[str, float] = defaultdict(float)
        self.slot_filled: Dict[str, float] = defaultdict(float)
        # (主語, 関係) -> その穴に置かれていた候補語
        self.open_slots: Deque[Tuple[Tuple[str, str], str]] = deque(
            maxlen=OPEN_SLOT_MEMORY)
        # --- 人称 ---------------------------------------------------------
        self.pron_count: Dict[str, float] = defaultdict(float)
        self.pron_ask: Dict[str, float] = defaultdict(float)
        self.pron_excl: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.vocative: Dict[str, float] = defaultdict(float)
        # --- 指示詞 -------------------------------------------------------
        # 指示詞は解決されて別の語に置き換わるので、解決前にここで数えておく
        self.cand_slots: Dict[str, Dict[tuple, float]] = defaultdict(dict)
        # 「AのB」の A に立つ回数。人称は所有者になり、指示詞はならない。
        self.as_owner: Dict[str, float] = defaultdict(float)
        self.as_head: Dict[str, float] = defaultdict(float)
        # 品詞は初出時の解析に引きずられる(「何」が 名詞,数 になる等)ので、
        # ここで多数決を取り直す。
        self.pos_seen: Dict[str, Dict[tuple, float]] = defaultdict(dict)
        self.observed = 0

    # ------------------------------------------------------------------
    # 観測
    # ------------------------------------------------------------------
    def observe(self, toks: Sequence, preds: Sequence[Pred], asking: bool) -> None:
        if not toks:
            return
        self.observed += 1

        # 呼びかけ「ロックマン、おはよう」= 文頭の格標識なし名詞句 + 読点
        if (len(toks) > 1 and toks[0].pos == "名詞"
                and toks[1].surface in "、，,"):
            self.vocative[toks[0].surface] += 1.0

        table = self.in_ask if asking else self.in_plain
        for t in toks:
            if t.pos not in ("名詞", "副詞") or len(t.surface) > 4:
                continue
            table[t.surface] += 1.0
            row = self.pos_seen[t.surface]
            row[(t.pos, t.pos1)] = row.get((t.pos, t.pos1), 0.0) + 1.0
            if t.pos1 == "代名詞":
                self.pron_count[t.surface] += 1.0
                if asking:
                    self.pron_ask[t.surface] += 1.0

        prons = {t.surface for t in toks if t.pos1 == "代名詞"}
        for p in preds:
            for a in p.args:
                if a.head in prons:
                    self.as_head[a.head] += 1.0
                if a.owner in prons:
                    self.as_owner[a.owner] += 1.0
        for p in preds:
            if not p.lemma or p.kind == "query":
                continue
            for a in p.args:
                if a.case and a.head in prons:
                    row = self.cand_slots[a.head]
                    key = (p.lemma, a.case)
                    row[key] = row.get(key, 0.0) + 1.0

        # 同じ発話に別々の代名詞が出たら、それらは別の実体を指している
        # (「ぼくはきみのオペレーターだよ」)。所有者の位置も数える。
        seen_prons: List[str] = []
        for p in preds:
            for a in p.args:
                for w in (a.head, a.owner):
                    if w and w in self.pron_count and w not in seen_prons:
                        seen_prons.append(w)
        for i, a in enumerate(seen_prons):
            for b in seen_prons[i + 1:]:
                self.pron_excl[a][b] = self.pron_excl[a].get(b, 0.0) + 1.0
                self.pron_excl[b][a] = self.pron_excl[b].get(a, 0.0) + 1.0

    def observe_slots(self, preds: Sequence[Pred], asking: bool) -> None:
        """問いかけの「穴」を覚える。

        人称を役割に畳んだ **後** に呼ぶこと。事実ストアの鍵と揃わないと
        「後から埋まったか」を照合できない。
        """
        if not asking:
            return
        for p in preds:
            self._open_slot(p)

    def _open_slot(self, p: Pred) -> None:
        """問いかけの「値が入るはずの場所」と、そこに置かれていた語を覚える。"""
        topic = p.arg("は") or p.arg("も")
        if not topic or not topic.head:
            return
        key = (topic.owner or topic.head, topic.head if topic.owner else "=")
        cands: List[str] = []
        if p.kind == "copula" and p.lemma:
            cands.append(p.lemma)
        cands.extend(a.head for a in p.args if not a.case and a.head)
        for w in cands:
            if len(w) <= 4:
                self.slot_seen[w] += 1.0
                self.open_slots.append((key, w))

    def note_fact(self, subj: str, rel: str, obj: str) -> None:
        """事実が言い切られたとき、同じ穴に置かれていた語を疑問詞候補に加点。"""
        key = (subj, rel)
        for k, w in list(self.open_slots):
            if k == key and w != obj:
                self.slot_filled[w] += 1.0

    # ------------------------------------------------------------------
    # 判定
    # ------------------------------------------------------------------
    def q_score(self, w: str) -> float:
        ask, plain = self.in_ask.get(w, 0.0), self.in_plain.get(w, 0.0)
        if ask + plain < MIN_COUNT:
            return 0.0
        bias = math.log((ask + 0.5) / (plain + 0.5))
        bias = max(0.0, min(1.0, bias / 2.5))          # 0..1
        seen = self.slot_seen.get(w, 0.0)
        filled = (self.slot_filled.get(w, 0.0) / seen) if seen >= 2 else 0.0
        return 0.5 * bias + 0.5 * min(1.0, filled)

    def questions(self, brain=None) -> Set[str]:
        """疑問詞候補は代名詞か副詞に限る(品詞は構造であって語彙知識ではない)。"""
        out = set()
        for w in set(self.in_ask) | set(self.in_plain):
            if not self.pro_like(w):
                continue
            if self.q_score(w) >= Q_THRESHOLD:
                out.add(w)
        return out

    def pro_like(self, w: str) -> bool:
        """代名詞か副詞か(多数決)。機能語候補はこの 2 つに限る。"""
        row = self.pos_seen.get(w)
        if not row:
            return False
        (pos, pos1), _ = max(row.items(), key=lambda kv: kv[1])
        return pos1 == "代名詞" or pos == "副詞"

    def owner_ratio(self, w: str) -> float:
        """「AのB」の A に立つ割合。人称は高く、指示詞はほぼゼロ。"""
        o, h = self.as_owner.get(w, 0.0), self.as_head.get(w, 0.0)
        return o / (o + h) if (o + h) else 0.0

    def ask_only(self, w: str) -> float:
        """問いかけでしか使われない度合い。人称グラフの汚染を除くのに使う。"""
        a, pl = self.in_ask.get(w, 0.0), self.in_plain.get(w, 0.0)
        return a / (a + pl) if (a + pl) else 0.0

    # ------------------------------------------------------------------
    def person_groups(self) -> Tuple[Set[str], Set[str]]:
        """相互排他グラフを 2 彩色し、(一人称, 二人称) を返す。"""
        # 疑問詞・指示詞も IPADIC では代名詞なので、人称のグラフから外す。
        # これを入れると彩色が滅茶苦茶になり、どちらが聞き手かも反転する。
        nodes = [w for w, c in self.pron_count.items()
                 if c >= MIN_PRONOUN and self.pron_excl.get(w)
                 and self.ask_only(w) < 0.8
                 and self.owner_ratio(w) >= OWNER_MIN]
        if len(nodes) < 2:
            return set(), set()
        nodes.sort(key=lambda w: -self.pron_count[w])
        allowed = set(nodes)
        color: Dict[str, int] = {}
        for root in nodes:
            if root in color:
                continue
            color[root] = 0
            queue = deque([root])
            while queue:
                cur = queue.popleft()
                for nxt, v in self.pron_excl.get(cur, {}).items():
                    # 広げる先も同じ条件を満たす語に限る。
                    # ここを緩めると疑問詞・指示詞が人称グラフに流れ込む。
                    if v < 1.0 or nxt not in allowed or nxt in color:
                        continue
                    color[nxt] = 1 - color[cur]
                    queue.append(nxt)
        if not color:
            return set(), set()
        g0 = {w for w, c in color.items() if c == 0}
        g1 = {w for w, c in color.items() if c == 1}
        if not g0 or not g1:
            return set(), set()

        # 聞き手はどちらか: 問いかけに出る率が高いほうが二人称
        def ask_ratio(group: Set[str]) -> float:
            tot = sum(self.pron_count[w] for w in group) or 1.0
            return sum(self.pron_ask.get(w, 0.0) for w in group) / tot

        if ask_ratio(g0) > ask_ratio(g1):
            return g1, g0
        return g0, g1

    # ------------------------------------------------------------------
    def raw_deictic(self, w: str) -> float:
        """いろんな述語・格に散らばって現れるか 0..1。

        「回数あたりのスロット数」だと繰り返すほど下がってしまうので、
        スロットの種類数そのもので測る(尺度不変)。
        """
        slots = self.cand_slots.get(w)
        if not slots:
            return 0.0
        if sum(slots.values()) < 3.0:
            return 0.0
        return max(0.0, min(1.0, len(slots) / 8.0))

    def deictic_score(self, w: str, frames=None, embed=None, brain=None) -> float:
        """指示詞らしさ。

        代名詞は 3 つしかない: 人称・疑問詞・指示詞。だから指示詞とは
        「所有者に立たない(人称でない)」かつ「穴として後から埋まらない
        (疑問詞でない)」代名詞のこと。この 2 つの比だけで代名詞が三分割できる。
        述語・格の散らばりは補助的に足すだけ(コーパスが小さいと効きにくい)。
        """
        not_person = 1.0 - min(1.0, self.owner_ratio(w) / OWNER_MIN)
        not_question = 1.0 - min(1.0, self.q_score(w) / Q_THRESHOLD)
        return (0.5 * not_person + 0.3 * not_question
                + 0.2 * self.raw_deictic(w))

    def deictics(self, frames=None, embed=None, brain=None,
                 exclude: Optional[Set[str]] = None) -> Set[str]:
        out = set()
        for w in self.cand_slots:
            # 疑問詞も代名詞なので、先に見つけた分と、問いかけ偏重の語は外す
            if len(w) > 4 or self.ask_only(w) >= 0.6:
                continue
            if exclude and w in exclude:
                continue
            if self.owner_ratio(w) >= OWNER_MIN:
                continue          # 所有者に立つ語は人称。指示詞ではない。
            if not self.pro_like(w):
                continue
            if self.deictic_score(w, frames, embed, brain) >= D_THRESHOLD:
                out.add(w)
        return out

    # ------------------------------------------------------------------
    def discovered(self, frames=None, embed=None, brain=None) -> FunctionWords:
        first, second = self.person_groups()
        qs = self.questions(brain)
        deic = self.deictics(frames, embed, brain, exclude=qs | first | second)
        return FunctionWords(qs, first, second, deic)

    def stats(self) -> Dict[str, float]:
        first, second = self.person_groups()
        return {"observed": self.observed, "questions": len(self.questions()),
                "deictics": len(self.deictics()),
                "first": len(first), "second": len(second),
                "pronouns": len(self.pron_count)}

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"in_ask": dict(self.in_ask), "in_plain": dict(self.in_plain),
                "slot_seen": dict(self.slot_seen),
                "slot_filled": dict(self.slot_filled),
                "pron_count": dict(self.pron_count),
                "pron_ask": dict(self.pron_ask),
                "pron_excl": {a: dict(r) for a, r in self.pron_excl.items()},
                "vocative": dict(self.vocative),
                "as_owner": dict(self.as_owner), "as_head": dict(self.as_head),
                "pos_seen": [[w, list(k), v] for w, r in self.pos_seen.items()
                             for k, v in r.items()],
                "cand_slots": [[w, list(k), v] for w, r in self.cand_slots.items()
                               for k, v in r.items()],
                "observed": self.observed}

    def load(self, d: dict) -> None:
        if not d:
            return
        self.in_ask = defaultdict(float, d.get("in_ask") or {})
        self.in_plain = defaultdict(float, d.get("in_plain") or {})
        self.slot_seen = defaultdict(float, d.get("slot_seen") or {})
        self.slot_filled = defaultdict(float, d.get("slot_filled") or {})
        self.pron_count = defaultdict(float, d.get("pron_count") or {})
        self.pron_ask = defaultdict(float, d.get("pron_ask") or {})
        self.pron_excl = defaultdict(
            dict, {a: dict(r) for a, r in (d.get("pron_excl") or {}).items()})
        self.vocative = defaultdict(float, d.get("vocative") or {})
        self.as_owner = defaultdict(float, d.get("as_owner") or {})
        self.as_head = defaultdict(float, d.get("as_head") or {})
        self.pos_seen = defaultdict(dict)
        for w, k, v in (d.get("pos_seen") or []):
            self.pos_seen[w][tuple(k)] = float(v)
        self.cand_slots = defaultdict(dict)
        for w, k, v in (d.get("cand_slots") or []):
            self.cand_slots[w][tuple(k)] = float(v)
        self.observed = int(d.get("observed", 0))
