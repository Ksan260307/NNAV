"""分布意味ベクトル。

共起を PPMI 化して SVD で圧縮し、語を密ベクトルにする。文脈には
  * 線形文脈  … 直前/直後に来た語 (2-gram から)
  * 構文文脈  … (述語, 格) スロット (frames.py から)
の 2 種類を使う。データが少ないうちは構文文脈のほうが圧倒的に効くので
重みを厚くしてある。

ここで得たベクトルの使いどころは 2 つ。

  1. **意味的な近さ** — 「犬」と「猫」が近い、という判断ができる
  2. **クラスベース補間** — 未観測の遷移を「似た語の経験」で埋める。
     語彙が薄い時期の発話品質がこれで大きく変わる。

構築は擬似睡眠(夢)の最中に回す。ナビが眠っている間に意味空間が
編成し直される、という設計。
"""

from __future__ import annotations

import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

MIN_WORDS = 60          # これ未満なら構築しない
MIN_NNZ = 400
MAX_NBR_WORDS = 3000    # 近傍表を作る対象語数の上限
SYNTAX_WEIGHT = 2.5     # 構文文脈の重み
NEIGHBORS = 8


class Embedding:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self.vecs: Optional[torch.Tensor] = None      # [V, dim] 行は正規化済み
        self.nbr: Dict[int, List[Tuple[int, float]]] = {}
        self.built_vocab = 0
        self.built_at = 0.0
        self.builds = 0

    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return self.vecs is not None and self.vecs.numel() > 0

    def stale(self, vocab: int, min_growth: float = 0.15) -> bool:
        if not self.ready():
            return True
        if self.built_vocab <= 0:
            return True
        return (vocab - self.built_vocab) / self.built_vocab >= min_growth

    # ------------------------------------------------------------------
    def build(self, brain, frames) -> bool:
        """共起行列を組んで SVD する。成功したら True。"""
        from .brain import N_SPECIAL

        ctx_id: Dict[object, int] = {}

        def cid(key) -> int:
            i = ctx_id.get(key)
            if i is None:
                i = len(ctx_id)
                ctx_id[key] = i
            return i

        counts: Dict[Tuple[int, int], float] = defaultdict(float)

        # --- 線形文脈 (2-gram) -------------------------------------------
        for a, row in brain.bi.items():
            if a < N_SPECIAL:
                continue
            for b, v in row.items():
                if b < N_SPECIAL:
                    continue
                counts[(a, cid(("R", b)))] += v
                counts[(b, cid(("L", a)))] += v

        # --- 構文文脈 (述語, 格) ------------------------------------------
        for noun, slot, v in frames.contexts():
            wid = brain.word2id.get(noun)
            if wid is None or wid < N_SPECIAL:
                continue
            counts[(wid, cid(("S", slot)))] += v * SYNTAX_WEIGHT

        if len(counts) < MIN_NNZ:
            return False
        rows_present = {r for r, _ in counts}
        if len(rows_present) < MIN_WORDS:
            return False

        # --- PPMI ---------------------------------------------------------
        row_sum: Dict[int, float] = defaultdict(float)
        col_sum: Dict[int, float] = defaultdict(float)
        total = 0.0
        for (r, c), v in counts.items():
            row_sum[r] += v
            col_sum[c] += v
            total += v
        if total <= 0:
            return False

        ri: List[int] = []
        ci: List[int] = []
        vs: List[float] = []
        for (r, c), v in counts.items():
            pmi = math.log((v * total) / (row_sum[r] * col_sum[c]))
            if pmi > 0.0:
                ri.append(r)
                ci.append(c)
                vs.append(pmi)
        if len(vs) < MIN_NNZ:
            return False

        V = brain.vocab_size
        C = len(ctx_id)
        q = min(self.dim, max(2, min(V, C) - 1))
        idx = torch.tensor([ri, ci], dtype=torch.long)
        val = torch.tensor(vs, dtype=torch.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            A = torch.sparse_coo_tensor(idx, val, (V, C)).coalesce()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                U, S, _ = torch.svd_lowrank(A, q=q, niter=4)
        except Exception:
            if V * C > 20_000_000:
                return False
            U, S, _ = torch.svd_lowrank(A.to_dense(), q=q, niter=4)

        emb = U * S.sqrt().unsqueeze(0)
        norm = emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.vecs = (emb / norm).contiguous()
        self.built_vocab = V
        self.built_at = time.time()
        self.builds += 1
        self._build_neighbors(brain)
        return True

    # ------------------------------------------------------------------
    def _build_neighbors(self, brain) -> None:
        from .brain import N_SPECIAL

        assert self.vecs is not None
        V = self.vecs.shape[0]
        # 出現の多い語だけを近傍表の対象にする(計算量とノイズの抑制)
        cand = [w for w in range(N_SPECIAL, min(V, brain.vocab_size))
                if brain.bi_tot.get(w, 0.0) >= 1.0]
        if len(cand) > MAX_NBR_WORDS:
            cand.sort(key=lambda w: -brain.bi_tot.get(w, 0.0))
            cand = cand[:MAX_NBR_WORDS]
        if len(cand) < 8:
            self.nbr = {}
            return
        ids = torch.tensor(cand, dtype=torch.long)
        M = self.vecs[ids]                       # [n, d]
        nbr: Dict[int, List[Tuple[int, float]]] = {}
        step = 512
        k = min(NEIGHBORS + 1, len(cand))
        for s in range(0, len(cand), step):
            block = M[s:s + step]
            sims = block @ M.T                   # [b, n]
            top = torch.topk(sims, k, dim=1)
            for bi in range(block.shape[0]):
                src = cand[s + bi]
                out: List[Tuple[int, float]] = []
                for v, j in zip(top.values[bi].tolist(), top.indices[bi].tolist()):
                    dst = cand[j]
                    if dst == src or v <= 0.12:
                        continue
                    out.append((dst, float(v)))
                if out:
                    nbr[src] = out[:NEIGHBORS]
        self.nbr = nbr

    # ------------------------------------------------------------------
    def neighbors(self, wid: int, k: int = 4) -> List[Tuple[int, float]]:
        return self.nbr.get(wid, [])[:k]

    def sim(self, a: int, b: int) -> float:
        if not self.ready():
            return 0.0
        v = self.vecs
        if a >= v.shape[0] or b >= v.shape[0]:
            return 0.0
        return float(torch.dot(v[a], v[b]).item())

    def similarity_of_sets(self, xs, ys) -> float:
        """語集合どうしの意味的な近さ(話題の一致度)。"""
        if not self.ready() or not xs or not ys:
            return 0.0
        v = self.vecs
        n = v.shape[0]
        xi = [x for x in xs if x < n]
        yi = [y for y in ys if y < n]
        if not xi or not yi:
            return 0.0
        a = v[torch.tensor(xi, dtype=torch.long)].mean(0)
        b = v[torch.tensor(yi, dtype=torch.long)].mean(0)
        na, nb = a.norm().clamp_min(1e-8), b.norm().clamp_min(1e-8)
        return float((a @ b / (na * nb)).item())

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        if not self.ready():
            return {}
        return {
            "dim": self.dim,
            "vecs": self.vecs.to(torch.float16),
            "nbr": {int(k): v for k, v in self.nbr.items()},
            "built_vocab": self.built_vocab,
            "built_at": self.built_at,
            "builds": self.builds,
        }

    def load(self, d: dict) -> None:
        if not d or "vecs" not in d:
            return
        self.dim = int(d.get("dim", self.dim))
        self.vecs = d["vecs"].to(torch.float32)
        self.nbr = {int(k): [(int(a), float(b)) for a, b in v]
                    for k, v in (d.get("nbr") or {}).items()}
        self.built_vocab = int(d.get("built_vocab", 0))
        self.built_at = float(d.get("built_at", 0.0))
        self.builds = int(d.get("builds", 0))
