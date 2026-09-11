"""理解度の測定。

「意味が分かってきた」を主観ではなく数値で見るための層。

ホールドアウトを別に取ると、ただでさえ少ないオペレーターの発話が
さらに減ってしまう。そこで **prequential 評価**(予測してから学習する)を使う。
オペレーターの発話が届いたら、まだ学習していない状態でそれを予測させ、
当たり具合を記録してから学習する。データを一切捨てずに汎化性能が測れる。

  * ppl   … 次の語をどれだけ言い当てられるか(対数確率の平均)
  * cloze … 文中の内容語を伏せて当てられるか(MRR)
  * slot  … その述語のその格に来る名詞を予測できているか(precision@k)

この 3 つを合成した 0〜1 の「理解度」が、成長段階のゲートになる。
語彙数ではなく、分かった度合いで育つ。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from .brain import BOS, EOS, N_SPECIAL

SMOOTH_K = 0.05
EMA = 0.05
CLOZE_TOPK_CAND = 1500
SLOT_K = 5


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


class Comprehension:
    def __init__(self) -> None:
        self.logprob = -9.0     # 1トークンあたりの対数確率(EMA)
        self.cloze = 0.0        # 穴埋めの MRR(EMA)
        self.slot = 0.0         # 格スロット予測の precision@k(EMA)
        self.samples = 0
        self.cloze_samples = 0
        self.slot_samples = 0
        # 外から差し込まれる軸(意図の当たり具合・指示詞の解決率)
        self.intent_skill = 0.0
        self.intent_ready = False
        self.anaphora_rate = 0.0
        self.anaphora_samples = 0

    # ------------------------------------------------------------------
    # 1トークンあたりの対数確率
    # ------------------------------------------------------------------
    @staticmethod
    def token_logprob(brain, ids: Sequence[int]) -> float:
        if not ids:
            return -9.0
        V = max(1, brain.vocab_size)
        seq = [BOS] + list(ids) + [EOS]
        total = 0.0
        for a, b in zip(seq, seq[1:]):
            row = brain.bi.get(a)
            tot = brain.bi_tot.get(a, 0.0)
            c = row.get(b, 0.0) if row else 0.0
            p = (c + SMOOTH_K) / (tot + SMOOTH_K * V)
            total += math.log(max(p, 1e-12))
        return total / (len(seq) - 1)

    # ------------------------------------------------------------------
    # 穴埋め
    # ------------------------------------------------------------------
    @staticmethod
    def cloze_mrr(brain, ids: Sequence[int]) -> Optional[float]:
        """内容語を 1 つ伏せ、文脈から復元できるかを順位で測る。"""
        positions = [i for i, w in enumerate(ids)
                     if w >= N_SPECIAL and brain.is_content_id(w)]
        if not positions:
            return None
        # 情報量の大きい語(IDF が高い語)を隠す
        i = max(positions, key=lambda j: brain.idf(ids[j]))
        target = ids[i]
        prev = ids[i - 1] if i > 0 else BOS
        nxt = ids[i + 1] if i + 1 < len(ids) else EOS

        left = brain.bi.get(prev) or {}
        ltot = brain.bi_tot.get(prev, 0.0)
        V = max(1, brain.vocab_size)

        cands = [w for w in range(N_SPECIAL, brain.vocab_size)
                 if brain.is_content_id(w)]
        if len(cands) > CLOZE_TOPK_CAND:
            cands.sort(key=lambda w: -brain.bi_tot.get(w, 0.0))
            cands = cands[:CLOZE_TOPK_CAND]
            if target not in cands:
                cands.append(target)
        if len(cands) < 5:
            return None

        def score(w: int) -> float:
            pl = (left.get(w, 0.0) + SMOOTH_K) / (ltot + SMOOTH_K * V)
            row = brain.bi.get(w) or {}
            tot = brain.bi_tot.get(w, 0.0)
            pr = (row.get(nxt, 0.0) + SMOOTH_K) / (tot + SMOOTH_K * V)
            return math.log(pl) + math.log(pr)

        target_score = score(target)
        rank = 1
        for w in cands:
            if w != target and score(w) > target_score:
                rank += 1
                if rank > 200:
                    break
        return 1.0 / rank

    # ------------------------------------------------------------------
    # 格スロットの予測
    # ------------------------------------------------------------------
    @staticmethod
    def slot_precision(frames, preds) -> Optional[Tuple[int, int]]:
        hit = tried = 0
        for p in preds:
            if not p.lemma or p.kind == "query":
                continue
            for pred, case, noun in p.triples():
                tot = frames.slot_tot.get((pred, case), 0.0)
                if tot < 2.0:
                    continue          # まだ判断材料がない
                tried += 1
                top = [n for n, _ in frames.fillers(pred, case, SLOT_K)]
                if noun in top:
                    hit += 1
        if tried == 0:
            return None
        return hit, tried

    # ------------------------------------------------------------------
    def observe(self, brain, frames, ids: Sequence[int], preds) -> Dict[str, float]:
        """学習する **前** に呼ぶこと。これが汎化性能の測定になる。"""
        out: Dict[str, float] = {}
        if len(ids) >= 2:
            lp = self.token_logprob(brain, ids)
            self.logprob = (1 - EMA) * self.logprob + EMA * lp
            self.samples += 1
            out["logprob"] = lp

            mrr = self.cloze_mrr(brain, ids)
            if mrr is not None:
                self.cloze = (1 - EMA) * self.cloze + EMA * mrr
                self.cloze_samples += 1
                out["cloze"] = mrr

        sp = self.slot_precision(frames, preds)
        if sp is not None:
            hit, tried = sp
            v = hit / tried
            self.slot = (1 - EMA) * self.slot + EMA * v
            self.slot_samples += 1
            out["slot"] = v
        return out

    # ------------------------------------------------------------------
    def core(self) -> float:
        """常に測れる 3 軸だけの理解度。軸が増減しても比較できる基準線。"""
        if self.samples < 5:
            return 0.0
        return (0.30 * _norm(self.logprob, -9.0, -1.5)
                + 0.20 * min(1.0, self.cloze * 3.0)
                + 0.20 * self.slot) / 0.70

    def understanding(self) -> float:
        """0〜1 の総合スコア。成長段階のゲートに使う。

        軸は 5 つあるが、まだ測れていない軸(意図・照応)は分母から外す。
        「オペレーターが指示詞を使わないから育たない」が起きないようにするため。
        """
        if self.samples < 5:
            return 0.0
        axes = [
            (0.30, _norm(self.logprob, -9.0, -1.5)),
            (0.20, min(1.0, self.cloze * 3.0)),     # MRR 0.33 で満点扱い
            (0.20, self.slot),
        ]
        if self.intent_ready:
            axes.append((0.15, self.intent_skill))
        if self.anaphora_samples >= 5:
            axes.append((0.15, self.anaphora_rate))
        total = sum(w for w, _ in axes)
        return sum(w * v for w, v in axes) / total

    def detail(self) -> Dict[str, float]:
        return {
            "understanding": self.understanding(),
            "core": self.core(),
            "logprob": self.logprob,
            "perplexity": math.exp(-self.logprob),
            "cloze_mrr": self.cloze,
            "slot_precision": self.slot,
            "intent_skill": self.intent_skill,
            "intent_ready": 1.0 if self.intent_ready else 0.0,
            "anaphora_rate": self.anaphora_rate,
            "anaphora_samples": self.anaphora_samples,
            "samples": self.samples,
        }

    def to_dict(self) -> dict:
        return {"logprob": self.logprob, "cloze": self.cloze, "slot": self.slot,
                "samples": self.samples, "cloze_samples": self.cloze_samples,
                "slot_samples": self.slot_samples,
                "intent_skill": self.intent_skill,
                "intent_ready": self.intent_ready,
                "anaphora_rate": self.anaphora_rate,
                "anaphora_samples": self.anaphora_samples}

    def load(self, d: dict) -> None:
        self.logprob = float(d.get("logprob", -9.0))
        self.cloze = float(d.get("cloze", 0.0))
        self.slot = float(d.get("slot", 0.0))
        self.samples = int(d.get("samples", 0))
        self.cloze_samples = int(d.get("cloze_samples", 0))
        self.slot_samples = int(d.get("slot_samples", 0))
        self.intent_skill = float(d.get("intent_skill", 0.0))
        self.intent_ready = bool(d.get("intent_ready", False))
        self.anaphora_rate = float(d.get("anaphora_rate", 0.0))
        self.anaphora_samples = int(d.get("anaphora_samples", 0))
