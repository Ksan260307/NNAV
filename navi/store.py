"""永続化層: WAL(Write-Ahead Log) + スナップショット。

元記事の課題② への回答。毎ターン脳全体をディスクへ書き戻すのをやめ、
「そのターンに起きた出来事(差分)」だけを追記していく。

  navi_data/wal.jsonl    … 追記のみ。ナビが経験したことの時系列日記でもある
  navi_data/snapshot.pt  … 圧縮済みの脳。WAL が伸びたら作り直して WAL を切る

WAL には「脳への低レベルな書き込み」ではなく「起きた出来事」を記録する。
再生は出来事をもう一度体験させるだけなので、脳の内部表現を変更しても
過去ログがそのまま使える(マイグレーション不要)という利点がある。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import torch

SNAPSHOT_VERSION = 3


class Journal:
    """出来事の記録と再生。"""

    def __init__(self, path: str, flush_every: int = 8):
        self.path = path
        self.flush_every = max(1, flush_every)
        self._buf: List[str] = []
        self._lock = threading.Lock()
        self.lines = self._count_lines()
        self.replaying = False

    def _count_lines(self) -> int:
        if not os.path.exists(self.path):
            return 0
        n = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for _ in f:
                n += 1
        return n

    def record(self, op: str, **fields: Any) -> None:
        if self.replaying:
            return
        rec = {"op": op, "ts": round(time.time(), 3)}
        rec.update(fields)
        with self._lock:
            self._buf.append(json.dumps(rec, ensure_ascii=False))
            self.lines += 1
            if len(self._buf) >= self.flush_every:
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()

    def replay(self, apply: Callable[[Dict[str, Any]], None]) -> int:
        if not os.path.exists(self.path):
            return 0
        n = 0
        self.replaying = True
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 途中で電源が落ちた行は捨てる
                    try:
                        apply(rec)
                        n += 1
                    except Exception:
                        continue
        finally:
            self.replaying = False
        return n

    def truncate(self) -> None:
        with self._lock:
            self._buf.clear()
            open(self.path, "w", encoding="utf-8").close()
            self.lines = 0

    def archive(self, dest_dir: str) -> Optional[str]:
        """WAL を日記として退避してから切り詰める。"""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return None
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, time.strftime("diary-%Y%m%d-%H%M%S.jsonl"))
        self.flush()
        shutil.copyfile(self.path, dest)
        return dest


def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_snapshot(brain, growth, mood, extra: Dict[str, Any], path: str) -> None:
    n = brain.vocab_size
    payload = {
        "version": SNAPSHOT_VERSION,
        "saved_at": time.time(),
        # 語彙
        "id2word": brain.id2word,
        "pos": brain.pos,
        "pos1": brain.pos1,
        "conj": brain.conj,
        "content": bytes(brain._content[:n]),
        "state": brain.state[:n].cpu(),
        "valence": brain.valence[:n].cpu(),
        # シナプス
        "bi": {a: dict(r) for a, r in brain.bi.items()},
        "bi_tot": dict(brain.bi_tot),
        "tri": {k: dict(r) for k, r in brain.tri.items()},
        "assoc": {a: dict(r) for a, r in brain.assoc.items()},
        "assoc_tot": dict(brain.assoc_tot),
        "edges": brain.edges,
        # 記憶
        "phrases": [
            {"text": p.text, "ids": p.ids, "keys": p.keys, "count": p.count,
             "turn": p.turn, "next_idx": p.next_idx, "origin": p.origin,
             "is_question": p.is_question}
            for p in brain.phrases
        ],
        "df": dict(brain.df),
        "doc_count": brain.doc_count,
        "question_tails": [[list(k), v] for k, v in brain.question_tails.items()],
        "interjections": dict(brain.interjections),
        # カウンタ
        "turns": brain.turns,
        "total_tokens": brain.total_tokens,
        "web_tokens": brain.web_tokens,
        # 付随状態
        "growth": growth.to_dict(),
        "mood": mood.to_dict(),
        "extra": extra,
    }
    _atomic_save(payload, path)


def load_snapshot(brain, growth, mood, path: str) -> Dict[str, Any]:
    from collections import defaultdict

    if not os.path.exists(path):
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    if int(data.get("version", 0)) != SNAPSHOT_VERSION:
        print(f"[System] スナップショットの形式が古いため読み飛ばします "
              f"(v{data.get('version')} != v{SNAPSHOT_VERSION})。WAL から再構築します。")
        return {}

    brain.id2word = list(data["id2word"])
    brain.pos = list(data["pos"])
    brain.pos1 = list(data["pos1"])
    brain.conj = list(data.get("conj") or [""] * len(brain.pos))
    brain.word2id = {w: i for i, w in enumerate(brain.id2word)}
    n = len(brain.id2word)
    brain._grow(n + 1)
    brain.state[:n] = data["state"].to(brain.device)
    brain.valence[:n] = data["valence"].to(brain.device)
    content = data["content"]
    brain._content[:n] = content[:n]

    brain.bi = defaultdict(dict, {int(a): dict(r) for a, r in data["bi"].items()})
    brain.bi_tot = defaultdict(float, {int(a): float(v) for a, v in data["bi_tot"].items()})
    brain.tri = defaultdict(dict, {tuple(k): dict(r) for k, r in data["tri"].items()})
    brain.assoc = defaultdict(dict, {int(a): dict(r) for a, r in data["assoc"].items()})
    brain.assoc_tot = defaultdict(
        float, {int(a): float(v) for a, v in data["assoc_tot"].items()})
    brain.edges = int(data.get("edges", 0))

    from .brain import Phrase

    brain.phrases = []
    brain.phrase_by_text = {}
    for i, p in enumerate(data.get("phrases", [])):
        ph = Phrase(p["text"], list(p["ids"]), list(p["keys"]), int(p["turn"]),
                    int(p.get("origin", 1)), bool(p.get("is_question", False)))
        ph.count = float(p.get("count", 1.0))
        ph.next_idx = {int(k): float(v) for k, v in p.get("next_idx", {}).items()}
        brain.phrases.append(ph)
        brain.phrase_by_text[ph.text] = i

    brain.df = defaultdict(float, {int(k): float(v) for k, v in data["df"].items()})
    brain.doc_count = float(data.get("doc_count", 0.0))
    brain.question_tails = {
        tuple(k): float(v) for k, v in data.get("question_tails", [])
    }
    brain.interjections = {int(k): float(v) for k, v in data.get("interjections", {}).items()}
    brain.turns = int(data.get("turns", 0))
    brain.total_tokens = int(data.get("total_tokens", 0))
    brain.web_tokens = int(data.get("web_tokens", 0))

    growth.load(data.get("growth", {}))
    mood.load(data.get("mood", {}))
    return data.get("extra", {}) or {}
