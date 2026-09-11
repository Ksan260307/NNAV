"""指示詞の解決。

「それを食べた」の「それ」が何を指すのかを決める。使うのは 2 つだけ。

  1. **顕著性** — 最近出てきた実体ほど、主題や主格で出た実体ほど指されやすい
  2. **選択選好** — その述語のその格に置ける型かどうか(frames.py の PMI)

2 が効くのがこの設計の旨味で、「それを食べた」なら
pmi(食べる, を, カレー) > pmi(食べる, を, 公園) なので型の合う候補が選ばれる。
格フレームがそのまま型チェッカーとして再利用できている。

指示詞は疑問詞・人称と同じく閉じた機能語の集合として扱う
(Config.use_demonstratives で無効化できる)。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .parse import SEED_DEICTIC, FunctionWords

DEMONSTRATIVES = SEED_DEICTIC       # 後方互換

# 格ごとの目立ちやすさ(主題・主格で出た実体ほど後から指されやすい)
CASE_WEIGHT = {"は": 1.0, "も": 0.9, "が": 0.9, "を": 0.7,
               "に": 0.5, "で": 0.4, "と": 0.4}
DEFAULT_CASE_WEIGHT = 0.35

HALF_LIFE_TURNS = 6.0
MAX_MENTIONS = 60
MIN_SCORE = -2.0


class Discourse:
    def __init__(self, enabled: bool = True,
                 fw: Optional[FunctionWords] = None) -> None:
        self.enabled = enabled
        self.fw = fw if fw is not None else FunctionWords.seeded()
        self.mentions: Deque[Tuple[str, int, str]] = deque(maxlen=MAX_MENTIONS)
        self.attempted = 0
        self.resolved = 0
        self.last: List[Tuple[str, str]] = []     # (指示詞, 解決先) 直近の履歴

    # ------------------------------------------------------------------
    def is_demonstrative(self, s: str) -> bool:
        return s in self.fw.deictics

    def salience(self, turn_now: int, turn_then: int, case: str) -> float:
        age = max(0, turn_now - turn_then)
        decay = math.exp(-age / HALF_LIFE_TURNS)
        return decay * CASE_WEIGHT.get(case, DEFAULT_CASE_WEIGHT)

    # ------------------------------------------------------------------
    def resolve(self, preds: Sequence, frames, embed, brain, turn: int) -> int:
        """指示詞を直近の実体に結び付ける。preds を破壊的に書き換える。

        この発話自身はまだ mentions に入っていないので、自分自身を指すことはない。
        """
        if not self.enabled or not self.mentions:
            return 0
        n = 0
        for p in preds:
            others = [a.head for a in p.args if not self.is_demonstrative(a.head)]
            for a in p.args:
                if not self.is_demonstrative(a.head):
                    continue
                self.attempted += 1
                pick = self._best(a.case, p.lemma, others, frames, embed, brain, turn)
                if pick:
                    self.last.append((a.head, pick))
                    del self.last[:-10]
                    a.head = pick
                    self.resolved += 1
                    n += 1
        return n

    def _best(self, case: str, lemma: str, others: Sequence[str],
              frames, embed, brain, turn: int) -> Optional[str]:
        seen: Dict[str, float] = {}
        for entity, then, mcase in self.mentions:
            if self.is_demonstrative(entity) or entity in others:
                continue
            s = self.salience(turn, then, mcase)
            if s > seen.get(entity, 0.0):
                seen[entity] = s
        if not seen:
            return None

        best, best_score = None, MIN_SCORE
        for entity, sal in seen.items():
            score = math.log(max(sal, 1e-6))
            if frames is not None and lemma and case:
                score += 1.2 * frames.pmi(lemma, case, entity)
            if embed is not None and others and getattr(embed, "ready", None) \
                    and embed.ready():
                wid = brain.word2id.get(entity)
                if wid is not None:
                    sims = [embed.sim(wid, brain.word2id[o]) for o in others
                            if o in brain.word2id]
                    if sims:
                        score += 0.5 * max(sims)
            if score > best_score:
                best, best_score = entity, score
        return best

    # ------------------------------------------------------------------
    def observe(self, preds: Sequence, turn: int) -> None:
        """この発話に出てきた実体を談話に積む。"""
        if not self.enabled:
            return
        for p in preds:
            for a in p.args:
                if not a.head or self.is_demonstrative(a.head):
                    continue
                if len(a.head) > 24:
                    continue
                self.mentions.append((a.head, turn, a.case))

    # ------------------------------------------------------------------
    def rate(self) -> float:
        return self.resolved / self.attempted if self.attempted else 0.0

    def stats(self) -> Dict[str, float]:
        return {"mentions": len(self.mentions), "attempted": self.attempted,
                "resolved": self.resolved, "rate": self.rate()}

    def to_dict(self) -> dict:
        return {"mentions": list(self.mentions), "attempted": self.attempted,
                "resolved": self.resolved, "last": self.last[-10:]}

    def load(self, d: dict) -> None:
        self.mentions = deque((tuple(m) for m in (d.get("mentions") or [])),
                              maxlen=MAX_MENTIONS)
        self.attempted = int(d.get("attempted", 0))
        self.resolved = int(d.get("resolved", 0))
        self.last = [tuple(x) for x in (d.get("last") or [])]
