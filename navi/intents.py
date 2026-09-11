"""意図の分類。

「これは質問なのか、挨拶なのか、報告なのか」を **教師なし** で覚える。
ラベルも辞書も与えない。文末の品詞、述語の種類、モダリティ、疑問符の有無
といった形の特徴だけを並べて、球面 k-means でまとめるだけ。

そのうえで「どの意図には、どの意図で返ってきたか」を、オペレーター自身の
連続発話(Phrase.next_idx)から学習する。生成時はまず返答の意図を決めてから
候補を選ぶので、「質問には答える」「挨拶には挨拶」がルール無しで出てくる。

クラスタは眠っている間に貼り直す(意味ベクトルと同じ枠)。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

from .parse import parse

POS1_BUCKETS = ["終助詞", "句点", "助動詞", "接続助詞", "格助詞", "係助詞",
                "自立", "一般", "代名詞", "形容動詞語幹"]
KINDS = ["verb", "adj", "na", "copula", "query"]
MODS = ["past", "desire", "guess", "hearsay", "passive", "causative",
        "should", "polite"]
FLAGS = 6
SCALARS = 3
DIM = len(POS1_BUCKETS) + 1 + len(KINDS) + 1 + len(MODS) + FLAGS + SCALARS

MIN_SAMPLES = 300
RESERVOIR = 2000
ITERS = 12


class Intents:
    def __init__(self, k: int = 6) -> None:
        self.k = k
        self.centroids: Optional[torch.Tensor] = None   # [k, DIM] 正規化済み
        self.trans: Dict[int, Dict[int, float]] = {}
        self.reservoir: List[List[float]] = []
        self.examples: Dict[int, List[str]] = {}
        self.fits = 0
        # 予測の当たり具合(prequential)。独話の遷移なので表示用。
        self.hits = 0.0
        self.tries = 0.0
        self.marginal: Dict[int, float] = {}
        # 文脈付きバンディット: 入力意図 -> 返答意図 -> [報酬の和, 回数]
        # 「その返し方でよかったか」をオペレーターの反応から直に学ぶ。
        self.reward: Dict[int, Dict[int, List[float]]] = {}
        self.reward_sum = 0.0
        self.reward_n = 0.0

    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return self.centroids is not None

    def measurable(self) -> bool:
        """返し方の良し悪しを語れるだけの反応を見たか。"""
        return self.ready() and self.reward_n >= 20

    # ------------------------------------------------------------------
    def featurize(self, brain, ids: Sequence[int],
                  preds: Optional[Sequence] = None) -> List[float]:
        """語彙 ID の並びだけから特徴を作る(入力にも候補にも同じ式を使う)。"""
        v = [0.0] * DIM
        if not ids:
            return v
        toks = [brain.tok_of(i) for i in ids]
        if preds is None:
            preds = parse(toks)

        # 1) 文末トークンの品詞細分類
        last = toks[-1]
        off = 0
        idx = POS1_BUCKETS.index(last.pos1) if last.pos1 in POS1_BUCKETS else len(POS1_BUCKETS)
        v[off + idx] = 1.0
        off += len(POS1_BUCKETS) + 1

        # 2) 述語の種類
        kind = preds[-1].kind if preds else None
        idx = KINDS.index(kind) if kind in KINDS else len(KINDS)
        v[off + idx] = 1.0
        off += len(KINDS) + 1

        # 3) モダリティ
        mods = set()
        for p in preds:
            mods.update(p.modality)
        for i, m in enumerate(MODS):
            if m in mods:
                v[off + i] = 1.0
        off += len(MODS)

        # 4) フラグ
        surfaces = [t.surface for t in toks]
        v[off + 0] = 1.0 if any(s in "？?" for s in surfaces) else 0.0
        v[off + 1] = 1.0 if any(s in "！!" for s in surfaces) else 0.0
        v[off + 2] = 1.0 if any(p.is_question for p in preds) else 0.0
        v[off + 3] = 1.0 if toks[0].pos == "感動詞" else 0.0
        v[off + 4] = 1.0 if any(t.conj.startswith("命令") for t in toks) else 0.0
        v[off + 5] = 1.0 if any(p.polarity < 0 for p in preds) else 0.0
        off += FLAGS

        # 5) 連続値
        v[off + 0] = min(1.0, math.log1p(len(ids)) / 3.0)
        content = sum(1 for i in ids if brain.is_content_id(i))
        v[off + 1] = content / len(ids)
        v[off + 2] = min(1.0, sum(len(p.args) for p in preds) / 3.0)
        return v

    # ------------------------------------------------------------------
    def classify(self, vec: Sequence[float]) -> int:
        if self.centroids is None:
            return 0
        x = torch.tensor(list(vec), dtype=torch.float32)
        n = x.norm().clamp_min(1e-8)
        sims = self.centroids @ (x / n)
        return int(torch.argmax(sims).item())

    def add_sample(self, vec: Sequence[float], text: str = "") -> None:
        self.reservoir.append(list(vec))
        if len(self.reservoir) > RESERVOIR:
            del self.reservoir[: len(self.reservoir) - RESERVOIR]
        if text and self.ready():
            i = self.classify(vec)
            ex = self.examples.setdefault(i, [])
            if text not in ex:
                ex.append(text)
                del ex[:-3]

    # ------------------------------------------------------------------
    def fit(self, rng) -> bool:
        """球面 k-means。眠っている間に呼ぶ。"""
        if len(self.reservoir) < MIN_SAMPLES:
            return False
        X = torch.tensor(self.reservoir, dtype=torch.float32)
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)
        n = X.shape[0]
        k = min(self.k, n)

        # k-means++ 風の初期化
        idx = [rng.randrange(n)]
        for _ in range(k - 1):
            C = X[torch.tensor(idx, dtype=torch.long)]
            d = 1.0 - (X @ C.T).max(dim=1).values
            d = torch.clamp(d, min=1e-6)
            probs = (d / d.sum()).tolist()
            idx.append(rng.choices(range(n), weights=probs, k=1)[0])
        C = X[torch.tensor(idx, dtype=torch.long)].clone()

        for _ in range(ITERS):
            assign = torch.argmax(X @ C.T, dim=1)
            for j in range(k):
                mask = assign == j
                if int(mask.sum().item()) == 0:
                    C[j] = X[rng.randrange(n)]
                else:
                    C[j] = X[mask].mean(0)
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-8)

        self.centroids = C
        self.k = k
        self.fits += 1
        self.examples = {}
        return True

    # ------------------------------------------------------------------
    def observe_pair(self, prev: int, cur: int) -> None:
        """「この意図の次には、この意図が来た」を数える。"""
        row = self.trans.setdefault(prev, {})
        row[cur] = row.get(cur, 0.0) + 1.0
        self.marginal[cur] = self.marginal.get(cur, 0.0) + 1.0

    def predict(self, prev: int) -> Dict[int, float]:
        row = self.trans.get(prev)
        if not row:
            tot = sum(self.marginal.values()) or 1.0
            return {i: v / tot for i, v in self.marginal.items()}
        tot = sum(row.values()) or 1.0
        return {i: v / tot for i, v in row.items()}

    # ------------------------------------------------------------------
    # 報酬バンディット
    # ------------------------------------------------------------------
    def observe_reward(self, prev: int, reply: int, r: float) -> None:
        """返した後のオペレーターの反応を、その返し方の評価として記録する。"""
        row = self.reward.setdefault(prev, {})
        cell = row.setdefault(reply, [0.0, 0.0])
        cell[0] += r
        cell[1] += 1.0
        self.reward_sum += r
        self.reward_n += 1.0

    def _cell(self, prev: int, reply: int) -> List[float]:
        return self.reward.get(prev, {}).get(reply, [0.0, 0.0])

    def score_reply(self, prev: int, cand: int) -> float:
        """その意図で返すことの期待値 0..1。

        まだ試していない返し方には楽観的な下駄(UCB)を履かせるので、
        探索は自然に起きる。報酬が貯まるほど下駄は小さくなる。
        """
        row = self.reward.get(prev)
        if not row:
            return 0.55        # 手がかり無し。わずかに前向き。
        total = sum(c[1] for c in row.values())
        rsum, n = self._cell(prev, cand)
        mean = rsum / n if n else 0.0
        ucb = mean + 0.8 * math.sqrt(math.log(total + 1.0) / (n + 1.0))
        return max(0.0, min(1.0, (ucb + 2.0) / 4.0))   # [-2,2] -> [0,1]

    def appropriateness(self) -> float:
        """返し方の平均報酬を 0..1 に直したもの。理解度の意図軸。"""
        if self.reward_n < 20:
            return 0.0
        mean = self.reward_sum / self.reward_n
        return max(0.0, min(1.0, (mean + 1.0) / 2.0))   # [-1,1] -> [0,1]

    def best_reply(self, prev: int) -> int:
        row = self.reward.get(prev)
        if not row:
            return -1
        return max(row.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1.0))[0]

    def evaluate(self, prev: int, actual: int) -> None:
        """prequential: 予測してから正解を見る。"""
        dist = self.predict(prev)
        if dist:
            pred = max(dist.items(), key=lambda kv: kv[1])[0]
            self.hits += 1.0 if pred == actual else 0.0
            self.tries += 1.0

    def skill(self) -> float:
        """当たり具合を、常に最頻値を答えるベースラインとの差で測る。"""
        if self.tries < 20 or not self.marginal:
            return 0.0
        acc = self.hits / self.tries
        tot = sum(self.marginal.values()) or 1.0
        base = max(self.marginal.values()) / tot
        if base >= 0.999:
            return 0.0
        return max(0.0, min(1.0, (acc - base) / (1.0 - base)))

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        return {"k": self.k, "ready": 1.0 if self.ready() else 0.0,
                "samples": len(self.reservoir), "fits": self.fits,
                "accuracy": self.hits / self.tries if self.tries else 0.0,
                "skill": self.skill(),
                "appropriateness": self.appropriateness(),
                "rewarded": self.reward_n,
                "mean_reward": self.reward_sum / self.reward_n if self.reward_n else 0.0}

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "centroids": self.centroids.to(torch.float16) if self.ready() else None,
            "trans": {int(a): dict(r) for a, r in self.trans.items()},
            "marginal": dict(self.marginal),
            "reservoir": self.reservoir[-RESERVOIR:],
            "examples": {int(k): v for k, v in self.examples.items()},
            "fits": self.fits, "hits": self.hits, "tries": self.tries,
            "reward": {int(a): {int(b): list(c) for b, c in r.items()}
                       for a, r in self.reward.items()},
            "reward_sum": self.reward_sum, "reward_n": self.reward_n,
        }

    def load(self, d: dict) -> None:
        if not d:
            return
        self.k = int(d.get("k", self.k))
        c = d.get("centroids")
        self.centroids = c.to(torch.float32) if c is not None else None
        self.trans = {int(a): {int(b): float(v) for b, v in r.items()}
                      for a, r in (d.get("trans") or {}).items()}
        self.marginal = {int(a): float(v) for a, v in (d.get("marginal") or {}).items()}
        self.reservoir = [list(x) for x in (d.get("reservoir") or [])]
        self.examples = {int(k): list(v) for k, v in (d.get("examples") or {}).items()}
        self.fits = int(d.get("fits", 0))
        self.hits = float(d.get("hits", 0.0))
        self.tries = float(d.get("tries", 0.0))
        self.reward = {int(a): {int(b): list(c) for b, c in r.items()}
                       for a, r in (d.get("reward") or {}).items()}
        self.reward_sum = float(d.get("reward_sum", 0.0))
        self.reward_n = float(d.get("reward_n", 0.0))
