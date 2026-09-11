"""格フレームと選択選好。

「その述語のその格には、どんな名詞が来るのか」を数え続ける。

    食べる[を] <- カレー, パン, 野菜 …
    歩く[まで] <- 公園 …

これが貯まると 2 つのことができる。

  1. **選択選好** — 生成した文が「その述語にその名詞を置いてよいか」を
     PMI で採点できる。「料理を直す」のような破綻に減点が入る。
  2. **名詞クラス** — 同じスロットを埋める名詞は同類。埋め込みを作る前から
     「カレーとパンは似ている」が言える。

語彙知識は持ち込まない。全部オペレーターとネットの文から数えたものだけ。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .parse import Pred

Slot = Tuple[str, str]          # (述語, 格)

PMI_CLAMP = 3.0
MIN_SLOT_COUNT = 0.8            # これ未満しか見ていないスロットは判断を保留
FULL_TRUST_COUNT = 3.0          # これだけ見ていれば PMI を丸ごと信用する
SMOOTH_K = 0.35


class CaseFrames:
    def __init__(self) -> None:
        # 述語 -> 格 -> 名詞 -> 頻度
        self.frames: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict))
        self.slot_tot: Dict[Slot, float] = defaultdict(float)
        self.noun_tot: Dict[str, float] = defaultdict(float)
        self.pred_tot: Dict[str, float] = defaultdict(float)
        # 名詞 -> そのスロット集合(逆引き。類似度と埋め込みの文脈に使う)
        self.noun_slots: Dict[str, Dict[Slot, float]] = defaultdict(dict)
        self.total = 0.0
        self.observed = 0

    # ------------------------------------------------------------------
    def observe(self, preds: Sequence[Pred], w: float = 1.0) -> int:
        n = 0
        for p in preds:
            if not p.lemma or p.kind == "query":
                continue
            self.pred_tot[p.lemma] += w
            for pred, case, noun in p.triples():
                if not noun or len(noun) > 24:
                    continue
                slot = (pred, case)
                row = self.frames[pred][case]
                row[noun] = row.get(noun, 0.0) + w
                self.slot_tot[slot] += w
                self.noun_tot[noun] += w
                self.total += w
                ns = self.noun_slots[noun]
                ns[slot] = ns.get(slot, 0.0) + w
                n += 1
        self.observed += n
        return n

    # ------------------------------------------------------------------
    def pmi(self, pred: str, case: str, noun: str) -> float:
        """その名詞がそのスロットに来る自然さ。0 は「判断材料なし」。"""
        slot = (pred, case)
        tot = self.slot_tot.get(slot, 0.0)
        if tot < MIN_SLOT_COUNT or self.total <= 0:
            return 0.0
        c = self.frames.get(pred, {}).get(case, {}).get(noun, 0.0)
        n_types = max(1, len(self.frames.get(pred, {}).get(case, {})))
        p_cond = (c + SMOOTH_K) / (tot + SMOOTH_K * n_types)
        p_marg = (self.noun_tot.get(noun, 0.0) + SMOOTH_K) / (self.total + SMOOTH_K)
        if p_marg <= 0:
            return 0.0
        pmi = max(-PMI_CLAMP, min(PMI_CLAMP, math.log(p_cond / p_marg)))
        # 観測が少ないスロットの判断は弱める(切り捨てずに confidence で薄める)
        return pmi * min(1.0, tot / FULL_TRUST_COUNT)

    def score(self, preds: Sequence[Pred]) -> Tuple[float, int]:
        """文全体の選択選好スコアと、判定できた項の数。"""
        vals: List[float] = []
        for p in preds:
            if not p.lemma or p.kind == "query":
                continue
            for pred, case, noun in p.triples():
                v = self.pmi(pred, case, noun)
                if v != 0.0:
                    vals.append(v)
        if not vals:
            return 0.0, 0
        return sum(vals) / len(vals), len(vals)

    # ------------------------------------------------------------------
    def fillers(self, pred: str, case: str, topn: int = 8) -> List[Tuple[str, float]]:
        row = self.frames.get(pred, {}).get(case, {})
        return sorted(row.items(), key=lambda kv: -kv[1])[:topn]

    def cases_of(self, pred: str) -> List[Tuple[str, float]]:
        row = self.frames.get(pred, {})
        return sorted(((c, self.slot_tot.get((pred, c), 0.0)) for c in row),
                      key=lambda kv: -kv[1])

    def similar_nouns(self, noun: str, topn: int = 8) -> List[Tuple[str, float]]:
        """同じスロットを埋める名詞ほど似ている(埋め込みを作る前の近似)。"""
        mine = self.noun_slots.get(noun)
        if not mine:
            return []
        weights = {s: self._slot_weight(s) for s in mine}
        norm_a = math.sqrt(sum((mine[s] * weights[s]) ** 2 for s in mine)) or 1.0
        cand: Dict[str, float] = {}
        for slot in mine:
            pred, case = slot
            for other in self.frames.get(pred, {}).get(case, {}):
                if other != noun:
                    cand[other] = 0.0
        out: List[Tuple[str, float]] = []
        for other in cand:
            theirs = self.noun_slots.get(other, {})
            num = sum(mine[s] * theirs[s] * weights[s] ** 2
                      for s in mine if s in theirs)
            if num <= 0:
                continue
            norm_b = math.sqrt(sum((v * self._slot_weight(s)) ** 2
                                   for s, v in theirs.items())) or 1.0
            out.append((other, num / (norm_a * norm_b)))
        out.sort(key=lambda kv: -kv[1])
        return out[:topn]

    def _slot_weight(self, slot: Slot) -> float:
        """よくあるスロットほど情報量が低い(IDF 的な重み)。"""
        return 1.0 / (1.0 + math.log1p(self.slot_tot.get(slot, 0.0)))

    # ------------------------------------------------------------------
    def contexts(self) -> Iterable[Tuple[str, Slot, float]]:
        """埋め込み構築用: (名詞, スロット, 頻度) を列挙する。"""
        for noun, slots in self.noun_slots.items():
            for slot, v in slots.items():
                yield noun, slot, v

    def decay(self, rate: float, prune: float) -> None:
        """脳の忘却と歩調を合わせる。"""
        self.total = 0.0
        for pred, cases in list(self.frames.items()):
            for case, row in list(cases.items()):
                for noun in list(row.keys()):
                    v = row[noun] * rate
                    if v < prune:
                        del row[noun]
                        slots = self.noun_slots.get(noun)
                        if slots:
                            slots.pop((pred, case), None)
                            if not slots:
                                del self.noun_slots[noun]
                    else:
                        row[noun] = v
                if not row:
                    del cases[case]
                    self.slot_tot.pop((pred, case), None)
            if not cases:
                del self.frames[pred]
                self.pred_tot.pop(pred, None)
        # 集計を張り直す
        self.slot_tot = defaultdict(float)
        self.noun_tot = defaultdict(float)
        for pred, cases in self.frames.items():
            for case, row in cases.items():
                s = sum(row.values())
                self.slot_tot[(pred, case)] = s
                for noun, v in row.items():
                    self.noun_tot[noun] += v
                    self.total += v

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        return {
            "preds": len(self.frames),
            "slots": len(self.slot_tot),
            "nouns": len(self.noun_tot),
            "triples": self.total,
        }

    def to_dict(self) -> dict:
        return {
            "frames": {p: {c: dict(r) for c, r in cs.items()}
                       for p, cs in self.frames.items()},
            "pred_tot": dict(self.pred_tot),
            "observed": self.observed,
        }

    def load(self, d: dict) -> None:
        self.__init__()  # type: ignore[misc]
        for pred, cases in (d.get("frames") or {}).items():
            for case, row in cases.items():
                self.frames[pred][case] = dict(row)
                s = sum(row.values())
                self.slot_tot[(pred, case)] = s
                for noun, v in row.items():
                    self.noun_tot[noun] += v
                    self.total += v
                    self.noun_slots[noun][(pred, case)] = v
        self.pred_tot = defaultdict(float, d.get("pred_tot") or {})
        self.observed = int(d.get("observed", 0))
