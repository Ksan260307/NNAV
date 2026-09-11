"""小型ニューラル言語モデルと、n-gram からの引き継ぎ。

事前学習済みのモデルは使わない。**ナビ自身が溜めた会話だけ**で、深い睡眠の
あいだに一から学習する小さな GRU。SVD で作った意味ベクトルを初期値に使う
(せっかく作った意味空間を捨てずにニューラルネットの種にする)。

コスト設計が肝。生成の 1 ステップごとに GRU を回すと候補 16 本 x 20 ステップで
320 回の forward になって現実的でないので、

    候補を出すのは n-gram、採点だけニューラルネット

にしてある。候補 16 本を 1 バッチで評価するので 1 ターンあたり forward 1 回。

引き継ぎは自動。学習前の発話をどれだけ言い当てられるか(prequential)を
n-gram と競わせ、その差だけニューラル側の採点比重 alpha が上がる。
賢くなった分だけ主役になる。
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .brain import BOS, EOS, UNK

MIN_VOCAB = 150
MIN_SENTS = 300
CORPUS_CAP = 20000
MAX_LEN = 40
EMA = 0.05


class Corpus:
    """学習用の文(語彙 ID 列)のリングバッファ。"""

    def __init__(self, cap: int = CORPUS_CAP) -> None:
        self.cap = cap
        self.sents: List[List[int]] = []

    def add(self, ids: Sequence[int]) -> None:
        if len(ids) < 2:
            return
        self.sents.append([int(i) for i in ids[:MAX_LEN]])
        if len(self.sents) > self.cap:
            del self.sents[: len(self.sents) - self.cap]

    def sample(self, n: int, rng: random.Random) -> List[List[int]]:
        if not self.sents:
            return []
        return [self.sents[rng.randrange(len(self.sents))] for _ in range(n)]

    def __len__(self) -> int:
        return len(self.sents)

    def to_dict(self) -> dict:
        return {"sents": self.sents[-self.cap:]}

    def load(self, d: dict) -> None:
        self.sents = [list(s) for s in (d.get("sents") or [])]


class NeuralLM(nn.Module):
    def __init__(self, vocab: int, dim: int = 96, hidden: int = 192):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.emb = nn.Embedding(vocab, dim)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.drop(self.emb(x)))
        # 出力層は埋め込みと重み共有(パラメータを減らし、少データで効く)
        return self.proj(h) @ self.emb.weight.T


class Neural:
    def __init__(self, dim: int = 96, hidden: int = 192) -> None:
        self.dim = dim
        self.hidden = hidden
        self.model: Optional[NeuralLM] = None
        self.opt: Optional[torch.optim.Optimizer] = None
        self.corpus = Corpus()
        self.logprob = -9.0        # prequential な 1 トークンあたり対数確率(EMA)
        self.samples = 0
        self.steps = 0
        self.trainings = 0
        self.built_vocab = 0
        self.last_loss = 0.0
        self.last_trained = 0.0

    # ------------------------------------------------------------------
    def ready(self) -> bool:
        return self.model is not None

    def _clamp(self, ids: Sequence[int]) -> List[int]:
        v = self.built_vocab
        return [i if 0 <= i < v else UNK for i in ids]

    # ------------------------------------------------------------------
    def ensure(self, brain, embed=None) -> bool:
        """語彙が育ったらモデルを作り直す(学習済みの行は引き継ぐ)。"""
        vocab = brain.vocab_size
        if vocab < MIN_VOCAB or len(self.corpus) < MIN_SENTS:
            return False
        if self.model is not None and vocab <= self.built_vocab * 1.25:
            return False

        new = NeuralLM(vocab, self.dim, self.hidden)
        with torch.no_grad():
            # 1) 意味ベクトル(SVD)を初期値にする
            if embed is not None and embed.ready():
                src = embed.vecs
                n = min(src.shape[0], vocab)
                d = min(src.shape[1], self.dim)
                new.emb.weight[:n, :d] = src[:n, :d] * 0.5
            # 2) すでに学習済みの行はそのまま引き継ぐ
            if self.model is not None:
                old = self.model
                n = min(old.built_vocab if hasattr(old, "built_vocab")
                        else old.vocab, vocab)
                new.emb.weight[:n] = old.emb.weight[:n]
                new.gru.load_state_dict(old.gru.state_dict())
                new.proj.load_state_dict(old.proj.state_dict())

        self.model = new
        self.built_vocab = vocab
        self.opt = torch.optim.AdamW(new.parameters(), lr=2e-3, weight_decay=0.01)
        return True

    # ------------------------------------------------------------------
    def _batch(self, sents: Sequence[Sequence[int]]):
        n = max(len(s) for s in sents) + 1
        x = torch.zeros((len(sents), n), dtype=torch.long)
        y = torch.full((len(sents), n), -100, dtype=torch.long)
        for i, s in enumerate(sents):
            ids = self._clamp(s)
            seq = [BOS] + ids
            tgt = ids + [EOS]
            x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            y[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        return x, y

    def train(self, budget_sec: float = 20.0, batch: int = 32,
              rng: Optional[random.Random] = None) -> Dict[str, float]:
        """深い睡眠中に少しだけ学習する。時間予算を必ず守る。"""
        if not self.ready() or len(self.corpus) < MIN_SENTS:
            return {"steps": 0}
        rng = rng or random
        self.model.train()
        t0 = time.time()
        steps = 0
        total = 0.0
        while time.time() - t0 < budget_sec:
            sents = self.corpus.sample(batch, rng)
            if not sents:
                break
            x, y = self._batch(sents)
            logits = self.model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1), ignore_index=-100)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += float(loss.item())
            steps += 1
        self.model.eval()
        self.steps += steps
        self.trainings += 1 if steps else 0
        self.last_loss = total / steps if steps else self.last_loss
        self.last_trained = time.time()
        return {"steps": steps, "loss": self.last_loss,
                "sec": time.time() - t0}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_batch(self, seqs: Sequence[Sequence[int]]) -> List[float]:
        """各列の 1 トークンあたり対数確率。候補の採点用(1 バッチ 1 回)。"""
        model = self.model         # 学習スレッドが差し替えても読み中は固定
        if model is None or not seqs:
            return [0.0] * len(seqs)
        usable = [s for s in seqs if s]
        if not usable:
            return [0.0] * len(seqs)
        model.eval()
        x, y = self._batch(usable)
        logits = model(x)
        logp = F.log_softmax(logits, dim=-1)
        mask = y != -100
        safe = torch.where(mask, y, torch.zeros_like(y))
        picked = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        picked = torch.where(mask, picked, torch.zeros_like(picked))
        means = (picked.sum(1) / mask.sum(1).clamp_min(1)).tolist()
        out, k = [], 0
        for s in seqs:
            if s:
                out.append(float(means[k]))
                k += 1
            else:
                out.append(-9.0)
        return out

    def observe(self, ids: Sequence[int]) -> Optional[float]:
        """学習する **前** の発話で測る(prequential)。"""
        if not self.ready() or len(ids) < 2:
            return None
        lp = self.score_batch([list(ids)])[0]
        self.logprob = (1 - EMA) * self.logprob + EMA * lp
        self.samples += 1
        return lp

    # ------------------------------------------------------------------
    def alpha(self, ngram_logprob: float) -> float:
        """ニューラル側に採点をどれだけ任せるか 0..0.9。

        学習前の発話をどちらがよく言い当てたかの差だけで決まるので、
        賢くなった分だけ自動で主役が移る。
        """
        if not self.ready() or self.samples < 30:
            return 0.0
        diff = self.logprob - ngram_logprob
        a = 1.0 / (1.0 + math.exp(-1.2 * diff))
        return max(0.0, min(0.9, a))

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        return {"ready": 1.0 if self.ready() else 0.0,
                "vocab": self.built_vocab, "corpus": len(self.corpus),
                "steps": self.steps, "trainings": self.trainings,
                "loss": self.last_loss, "logprob": self.logprob,
                "perplexity": math.exp(-self.logprob) if self.samples else 0.0,
                "samples": self.samples,
                "params": sum(p.numel() for p in self.model.parameters())
                if self.ready() else 0}

    def to_dict(self) -> dict:
        d = {"corpus": self.corpus.to_dict(), "logprob": self.logprob,
             "samples": self.samples, "steps": self.steps,
             "trainings": self.trainings, "built_vocab": self.built_vocab,
             "last_loss": self.last_loss, "dim": self.dim, "hidden": self.hidden}
        if self.ready():
            d["state"] = {k: v.to(torch.float16)
                          for k, v in self.model.state_dict().items()}
        return d

    def load(self, d: dict) -> None:
        if not d:
            return
        self.dim = int(d.get("dim", self.dim))
        self.hidden = int(d.get("hidden", self.hidden))
        self.corpus.load(d.get("corpus") or {})
        self.logprob = float(d.get("logprob", -9.0))
        self.samples = int(d.get("samples", 0))
        self.steps = int(d.get("steps", 0))
        self.trainings = int(d.get("trainings", 0))
        self.built_vocab = int(d.get("built_vocab", 0))
        self.last_loss = float(d.get("last_loss", 0.0))
        state = d.get("state")
        if state and self.built_vocab >= MIN_VOCAB:
            try:
                model = NeuralLM(self.built_vocab, self.dim, self.hidden)
                model.load_state_dict({k: v.to(torch.float32)
                                       for k, v in state.items()})
                model.eval()
                self.model = model
                self.opt = torch.optim.AdamW(model.parameters(), lr=2e-3,
                                             weight_decay=0.01)
            except Exception:
                self.model = None
