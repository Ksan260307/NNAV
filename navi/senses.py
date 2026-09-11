"""接地。言葉と、ナビが living している環境を結び付ける。

内容語ごとに「その語が使われた瞬間の環境」を貯める。時刻・曜日・放置時間・
セッションの長さ（psutil があれば CPU 負荷とバッテリも）。

    おはよう -> 朝に尖る
    おやすみ -> 深夜に尖る
    3時     -> 15時前後に尖る

これで 3 つのことができる。

  1. 朝には朝の語が出やすくなる（生成時の事前確率）
  2. 「いつも話しかけてくる時間」にナビから声をかけられる
  3. オペレーターが時々口にする時刻表現から、**時計 API を叩かずに**
     「いま何時か」に近い語を選べる。一度も時刻を口にしなければ答えられない、
     という挙動もそれで正しい

環境値は完全にローカルで、ネット学習の送信経路には一切載せない。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:  # 任意依存。無ければ該当次元が 0 のままになるだけ
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

N_HOUR, N_WDAY, N_IDLE, N_SESS, N_SYS = 24, 7, 5, 3, 2
ENV_DIM = N_HOUR + N_WDAY + N_IDLE + N_SESS + N_SYS

OFF_HOUR = 0
OFF_WDAY = OFF_HOUR + N_HOUR
OFF_IDLE = OFF_WDAY + N_WDAY
OFF_SESS = OFF_IDLE + N_IDLE
OFF_SYS = OFF_SESS + N_SESS

IDLE_EDGES = (60.0, 300.0, 1800.0, 10800.0)     # 1分 5分 30分 3時間
SESS_EDGES = (300.0, 3600.0)
MIN_OBS = 3.0          # これだけ観測していない語は接地しているとみなさない


class Senses:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stat = torch.zeros((64, ENV_DIM), dtype=torch.float32)
        self.count = torch.zeros(64, dtype=torch.float32)
        self.session_start = time.time()
        self.samples = 0

    # ------------------------------------------------------------------
    def _grow(self, need: int) -> None:
        cap = self.stat.shape[0]
        if need <= cap:
            return
        new = cap
        while new < need:
            new *= 2
        pad = new - cap
        self.stat = torch.cat([self.stat, torch.zeros((pad, ENV_DIM))])
        self.count = torch.cat([self.count, torch.zeros(pad)])

    # ------------------------------------------------------------------
    def env_now(self, last_input_at: float) -> torch.Tensor:
        """いまの環境をベクトルにする。"""
        v = torch.zeros(ENV_DIM, dtype=torch.float32)
        now = time.localtime()
        h = now.tm_hour
        # 時刻は隣の時間帯にも少し漏らす(23時と0時は近い)
        v[OFF_HOUR + h] = 1.0
        v[OFF_HOUR + (h - 1) % N_HOUR] = 0.4
        v[OFF_HOUR + (h + 1) % N_HOUR] = 0.4
        v[OFF_WDAY + now.tm_wday] = 1.0

        idle = max(0.0, time.time() - last_input_at)
        b = sum(1 for e in IDLE_EDGES if idle >= e)
        v[OFF_IDLE + b] = 1.0

        sess = max(0.0, time.time() - self.session_start)
        b = sum(1 for e in SESS_EDGES if sess >= e)
        v[OFF_SESS + b] = 1.0

        if psutil is not None:  # pragma: no cover - 環境依存
            try:
                v[OFF_SYS + 0] = float(psutil.cpu_percent(interval=None)) / 100.0
                bat = psutil.sensors_battery()
                if bat is not None:
                    v[OFF_SYS + 1] = float(bat.percent) / 100.0
            except Exception:
                pass
        return v

    # ------------------------------------------------------------------
    def observe(self, ids: Sequence[int], env: torch.Tensor, w: float = 1.0) -> None:
        if not self.enabled or not ids:
            return
        uniq = sorted(set(int(i) for i in ids))
        self._grow(uniq[-1] + 1)
        idx = torch.tensor(uniq, dtype=torch.long)
        self.stat[idx] += env.unsqueeze(0) * w
        self.count[idx] += w
        self.samples += 1

    def decay(self, rate: float) -> None:
        if self.enabled:
            self.stat *= rate
            self.count *= rate

    # ------------------------------------------------------------------
    def score_all(self, env: torch.Tensor, vocab_size: int) -> Optional[torch.Tensor]:
        """全語について「いまらしさ」を出す。生成時に 1 ターン 1 回だけ呼ぶ。"""
        if not self.enabled or self.samples < 20:
            return None
        n = min(vocab_size, self.stat.shape[0])
        if n <= 0:
            return None
        S = self.stat[:n]
        norms = S.norm(dim=1).clamp_min(1e-6)
        enorm = env.norm().clamp_min(1e-6)
        sims = (S @ env) / (norms * enorm)
        sims = torch.where(self.count[:n] >= MIN_OBS, sims, torch.zeros_like(sims))
        out = torch.zeros(vocab_size, dtype=torch.float32)
        out[:n] = sims
        return out

    def affinity(self, wid: int, env: torch.Tensor) -> float:
        if not self.enabled or wid >= self.stat.shape[0]:
            return 0.0
        if float(self.count[wid].item()) < MIN_OBS:
            return 0.0
        v = self.stat[wid]
        d = v.norm().clamp_min(1e-6) * env.norm().clamp_min(1e-6)
        return float((v @ env / d).item())

    # ------------------------------------------------------------------
    def hour_hist(self, wid: int) -> Optional[List[float]]:
        if wid >= self.stat.shape[0] or float(self.count[wid].item()) < MIN_OBS:
            return None
        h = self.stat[wid, OFF_HOUR:OFF_HOUR + N_HOUR]
        tot = float(h.sum().item())
        if tot <= 0:
            return None
        return (h / tot).tolist()

    def peakedness(self, wid: int) -> float:
        """時刻分布の尖り具合 0..1。1 に近いほど特定の時間にしか使われない。"""
        hist = self.hour_hist(wid)
        if not hist:
            return 0.0
        ent = -sum(p * math.log(p) for p in hist if p > 0)
        return max(0.0, 1.0 - ent / math.log(N_HOUR))

    def profile(self, wid: int) -> str:
        hist = self.hour_hist(wid)
        if not hist:
            return "(まだ分からない)"
        peak = max(range(N_HOUR), key=lambda i: hist[i])
        band = ("深夜" if peak < 5 else "朝" if peak < 10 else "昼" if peak < 15
                else "夕方" if peak < 19 else "夜")
        return f"{band}({peak}時ごろ) 尖り{self.peakedness(wid):.2f}"

    def best_match(self, env: torch.Tensor, brain,
                   min_peak: float = 0.35, min_sim: float = 0.72
                   ) -> Optional[Tuple[int, float]]:
        """いまの環境に最も結び付いている語。「いま何時?」の答えの素。"""
        scores = self.score_all(env, brain.vocab_size)
        if scores is None:
            return None
        best, best_s = None, min_sim
        for wid in range(3, min(brain.vocab_size, self.stat.shape[0])):
            if not brain.is_content_id(wid):
                continue
            s = float(scores[wid].item())
            if s <= best_s:
                continue
            if self.peakedness(wid) < min_peak:
                continue
            best, best_s = wid, s
        return (best, best_s) if best is not None else None

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        grounded = int((self.count >= MIN_OBS).sum().item())
        return {"samples": self.samples, "grounded": grounded,
                "psutil": 1.0 if psutil is not None else 0.0}

    def to_dict(self) -> dict:
        n = int((self.count > 0).nonzero().max().item()) + 1 if self.samples else 0
        return {"stat": self.stat[:n].to(torch.float16),
                "count": self.count[:n].to(torch.float16),
                "samples": self.samples}

    def load(self, d: dict) -> None:
        if not d or "stat" not in d:
            return
        stat = d["stat"].to(torch.float32)
        cnt = d["count"].to(torch.float32)
        n = stat.shape[0]
        self._grow(max(n, 64))
        self.stat[:n] = stat
        self.count[:n] = cnt
        self.samples = int(d.get("samples", 0))
        self.session_start = time.time()
