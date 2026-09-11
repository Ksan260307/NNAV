"""ネットナビの脳。

構造:
  * 単語ニューロン      … 32bit にビットパックされた状態テンソル (dense, torch)
  * シナプス結合        … 2-gram / 3-gram の疎な遷移頻度マップ
  * 連想野 (assoc)      … ターンをまたいだ共起。話題の連続性を担う
  * 模倣記憶 (phrases)  … オペレーターの実発話と「その次に来た発話」のリンク

元記事では [max_vocab, max_vocab] の密行列でシナプスを保持していたが、
語彙 1 万語で 400MB、Web 学習で語彙が伸びると破綻する。ここでは
  - O(V) の状態テンソルは密なまま(忘却の一括テンソル演算はそのまま活かす)
  - O(V^2) のシナプスだけを疎構造
に分離し、語彙 10 万語規模でもノート PC で動くようにしている。
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import (
    FLAG_CONTENT,
    FLAG_PINNED,
    ORIGIN_OPERATOR,
    ORIGIN_SYSTEM,
    ORIGIN_WEB,
    Config,
)
from .jp import JPTokenizer, Tok, is_content, is_noise, is_terminator

# --- ハードウェア最適化(元記事より継承) ---------------------------------
torch.set_num_threads(os.cpu_count() or 4)
torch.set_flush_denormal(True)
if torch.cuda.is_available():  # pragma: no cover
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

BOS, EOS, UNK = 0, 1, 2
N_SPECIAL = 3


# ---------------------------------------------------------------------------
# 32bit ビットパック [vA(8) | vB(8) | vC(8) | origin(3) | flags(3)]
# ---------------------------------------------------------------------------
def pack_state(va, vb, vc, origin, flags):
    return ((va & 0xFF) | ((vb & 0xFF) << 8) | ((vc & 0xFF) << 16)
            | ((origin & 0x07) << 24) | ((flags & 0x07) << 27))


def unpack_state(s):
    return (s & 0xFF, (s >> 8) & 0xFF, (s >> 16) & 0xFF,
            (s >> 24) & 0x07, (s >> 27) & 0x07)


class Phrase:
    """オペレーターが実際に口にした一発話の記憶。"""

    __slots__ = ("text", "ids", "keys", "count", "turn", "next_idx", "origin", "is_question")

    def __init__(self, text, ids, keys, turn, origin=ORIGIN_OPERATOR, is_question=False):
        self.text = text
        self.ids = ids
        self.keys = keys
        self.count = 1.0
        self.turn = turn
        self.next_idx: Dict[int, float] = {}
        self.origin = origin
        self.is_question = is_question


class NaviBrain:
    def __init__(self, cfg: Config, tokenizer: Optional[JPTokenizer] = None):
        self.cfg = cfg
        self.tok = tokenizer or JPTokenizer()
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        # --- 語彙 ---------------------------------------------------------
        self.word2id: Dict[str, int] = {}
        self.id2word: List[str] = []
        self.pos: List[str] = []
        self.pos1: List[str] = []
        self.capacity = max(cfg.initial_capacity, 64)
        self.state = torch.zeros(self.capacity, dtype=torch.int32, device=self.device)
        self.valence = torch.zeros(self.capacity, dtype=torch.float32, device=self.device)
        self._content = bytearray(self.capacity)

        # --- シナプス -----------------------------------------------------
        self.bi: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.bi_tot: Dict[int, float] = defaultdict(float)
        self.tri: Dict[Tuple[int, int], Dict[int, float]] = defaultdict(dict)
        self.edges = 0

        # --- 連想野(ターン跨ぎ共起) ---------------------------------------
        self.assoc: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.assoc_tot: Dict[int, float] = defaultdict(float)

        # --- 模倣記憶 -----------------------------------------------------
        self.phrases: List[Phrase] = []
        self.phrase_by_text: Dict[str, int] = {}
        self.df: Dict[int, float] = defaultdict(float)
        self.doc_count = 0.0
        self.question_tails: Dict[Tuple[str, ...], float] = {}
        self.interjections: Dict[int, float] = {}

        # --- カウンタ -----------------------------------------------------
        self.turns = 0
        self.total_tokens = 0
        self.web_tokens = 0

        # 意味層(navi.py が差し込む)。無くても動く。
        self.embed = None

        for w in ("__BOS__", "__EOS__", "__UNK__"):
            self._alloc(w, "__SPECIAL__", "*", ORIGIN_SYSTEM, va=255, vb=255)

    # ------------------------------------------------------------------
    # 語彙
    # ------------------------------------------------------------------
    def _grow(self, need: int) -> None:
        if need <= self.capacity:
            return
        new_cap = self.capacity
        while new_cap < need:
            new_cap *= 2
        new_cap = min(new_cap, self.cfg.max_vocab + 16)
        pad = new_cap - self.capacity
        if pad <= 0:
            return
        self.state = torch.cat(
            [self.state, torch.zeros(pad, dtype=torch.int32, device=self.device)])
        self.valence = torch.cat(
            [self.valence, torch.zeros(pad, dtype=torch.float32, device=self.device)])
        self._content.extend(bytes(pad))
        self.capacity = new_cap

    def _alloc(self, surface, pos, pos1, origin, va=100, vb=30, flags=0) -> int:
        idx = len(self.id2word)
        self._grow(idx + 1)
        self.word2id[surface] = idx
        self.id2word.append(surface)
        self.pos.append(pos)
        self.pos1.append(pos1)
        self.state[idx] = pack_state(va, vb, 0, origin, flags)
        self._content[idx] = 1 if (flags & FLAG_CONTENT) else 0
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self.id2word)

    @property
    def learned_vocab(self) -> int:
        return max(0, self.vocab_size - N_SPECIAL)

    def ensure_word(self, tok: Tok, origin: int) -> int:
        wid = self.word2id.get(tok.surface)
        if wid is not None:
            return wid
        if self.vocab_size >= self.cfg.max_vocab:
            return UNK
        flags = FLAG_CONTENT if is_content(tok) else 0
        # Web 由来の言葉は「聞きかじり」なので初期重要度を低く置く
        va = 100 if origin == ORIGIN_OPERATOR else 55
        return self._alloc(tok.surface, tok.pos, tok.pos1, origin, va=va, flags=flags)

    def tok_of(self, wid: int) -> Tok:
        s = self.id2word[wid]
        return Tok(s, self.pos[wid], self.pos1[wid], s)

    def is_content_id(self, wid: int) -> bool:
        return bool(self._content[wid])

    # ------------------------------------------------------------------
    # シナプス
    # ------------------------------------------------------------------
    def _reinforce(self, a: int, b: int, w: float) -> None:
        row = self.bi[a]
        if b in row:
            row[b] += w
        else:
            row[b] = w
            self.edges += 1
        self.bi_tot[a] += w

    def _reinforce_tri(self, a: int, b: int, c: int, w: float) -> None:
        row = self.tri[(a, b)]
        if c in row:
            row[c] += w
        else:
            row[c] = w
            self.edges += 1

    def observe(self, ids: Sequence[int], weight: float) -> None:
        """単語列をシナプスに刻む。前後に BOS/EOS を補う。"""
        if not ids:
            return
        seq = [BOS] + list(ids) + [EOS]
        for i in range(len(seq) - 1):
            self._reinforce(seq[i], seq[i + 1], weight)
            if i + 2 < len(seq):
                self._reinforce_tri(seq[i], seq[i + 1], seq[i + 2], weight)
        self.total_tokens += len(ids)

    # ------------------------------------------------------------------
    # 刺激と忘却(元記事どおりの密テンソル一括演算)
    # ------------------------------------------------------------------
    def stimulate(self, ids: Iterable[int], scale: float = 1.0) -> None:
        uniq = sorted({i for i in ids if i >= N_SPECIAL})
        if not uniq:
            return
        idx = torch.tensor(uniq, dtype=torch.long, device=self.device)
        va, vb, vc, origin, flags = unpack_state(self.state[idx])
        va = torch.clamp(va + int(self.cfg.stim_va * scale), max=255)
        vb = torch.clamp(vb + int(self.cfg.stim_vb * scale), max=255)
        vc = torch.zeros_like(vc)  # 想起されたので減衰カウンタをリセット
        self.state[idx] = pack_state(va, vb, vc, origin, flags).to(torch.int32)

    def decay(self) -> None:
        """テンソル一括演算による忘却プロセス。"""
        n = self.vocab_size
        if n > N_SPECIAL:
            sl = slice(N_SPECIAL, n)
            va, vb, vc, origin, flags = unpack_state(self.state[sl])
            vc = torch.clamp(vc + self.cfg.decay_vc_step, max=255)
            hit = vc > self.cfg.decay_vc_threshold
            pinned = (flags & FLAG_PINNED) > 0
            damage = torch.where(hit & ~pinned,
                                 torch.full_like(va, self.cfg.decay_va_damage),
                                 torch.zeros_like(va))
            va = torch.clamp(va - damage, min=1)
            vc = torch.where(hit, torch.zeros_like(vc), vc)
            self.state[sl] = pack_state(va, vb, vc, origin, flags).to(torch.int32)
        self._decay_synapses()

    def _decay_synapses(self) -> None:
        r = self.cfg.synapse_decay
        prune = self.cfg.synapse_prune
        removed = 0
        for a, row in list(self.bi.items()):
            tot = 0.0
            for b in list(row.keys()):
                v = row[b] * r
                if v < prune:
                    del row[b]
                    removed += 1
                else:
                    row[b] = v
                    tot += v
            if row:
                self.bi_tot[a] = tot
            else:
                del self.bi[a]
                self.bi_tot.pop(a, None)
        for key, row in list(self.tri.items()):
            for c in list(row.keys()):
                v = row[c] * r
                if v < prune:
                    del row[c]
                    removed += 1
                else:
                    row[c] = v
            if not row:
                del self.tri[key]
        self.edges = max(0, self.edges - removed)
        if self.edges > self.cfg.max_edges:
            self._hard_prune()

    def _hard_prune(self) -> None:
        """容量上限。弱い結合から順に切り捨てる(記憶の飽和)。"""
        vals = sorted(v for row in self.tri.values() for v in row.values())
        if not vals:
            return
        thr = vals[len(vals) // 3]
        for key, row in list(self.tri.items()):
            for c in list(row.keys()):
                if row[c] <= thr:
                    del row[c]
                    self.edges -= 1
            if not row:
                del self.tri[key]

    def importance(self, ids: Optional[Sequence[int]] = None) -> torch.Tensor:
        """重要度 vA を 0..1 に正規化して返す。"""
        if ids is None:
            s = self.state[: self.vocab_size]
        else:
            s = self.state[torch.tensor(list(ids), dtype=torch.long, device=self.device)]
        return (s & 0xFF).to(torch.float32) / 255.0

    # ------------------------------------------------------------------
    # 連想野
    # ------------------------------------------------------------------
    def associate(self, src_ids: Iterable[int], dst_ids: Iterable[int], w: float) -> None:
        src = {i for i in src_ids if self.is_content_id(i)}
        dst = {i for i in dst_ids if self.is_content_id(i)}
        if not src or not dst:
            return
        for a in src:
            row = self.assoc[a]
            added = 0.0
            for b in dst:
                if a == b:
                    continue
                row[b] = row.get(b, 0.0) + w
                added += w
            self.assoc_tot[a] += added

    def related(self, ids: Iterable[int], topn: int = 24) -> List[Tuple[int, float]]:
        agg: Dict[int, float] = {}
        for a in ids:
            row = self.assoc.get(a)
            if not row:
                continue
            tot = self.assoc_tot.get(a, 1.0) or 1.0
            for b, v in row.items():
                agg[b] = agg.get(b, 0.0) + v / tot
        return sorted(agg.items(), key=lambda kv: -kv[1])[:topn]

    def relevance(self, src_ids: Sequence[int], cand_ids: Sequence[int]) -> float:
        """入力キーワードと候補語群の連想的な近さ。"""
        if not src_ids or not cand_ids:
            return 0.0
        cand = set(cand_ids)
        score = 0.0
        for a in src_ids:
            row = self.assoc.get(a)
            if not row:
                continue
            tot = self.assoc_tot.get(a, 1.0) or 1.0
            score += sum(row.get(b, 0.0) for b in cand) / tot
        return score / len(src_ids)

    # ------------------------------------------------------------------
    # IDF と模倣記憶
    # ------------------------------------------------------------------
    def idf(self, wid: int) -> float:
        return math.log((self.doc_count + 1.0) / (self.df.get(wid, 0.0) + 1.0)) + 1.0

    def remember_phrase(self, text, ids, keys, origin, is_question) -> int:
        key = text.strip()
        idx = self.phrase_by_text.get(key)
        if idx is not None:
            ph = self.phrases[idx]
            ph.count += 1.0
            ph.turn = self.turns
            return idx
        idx = len(self.phrases)
        self.phrases.append(Phrase(key, ids, keys, self.turns, origin, is_question))
        self.phrase_by_text[key] = idx
        return idx

    def link_phrases(self, prev_idx: int, next_idx: int) -> None:
        if prev_idx < 0 or next_idx < 0 or prev_idx == next_idx:
            return
        nx = self.phrases[prev_idx].next_idx
        nx[next_idx] = nx.get(next_idx, 0.0) + 1.0

    def search_phrases(self, keys, topn=5, exclude=None) -> List[Tuple[int, float]]:
        """入力キーワードに近い過去の発話を検索(IDF 重み付きコサイン風)。"""
        if not keys or not self.phrases:
            return []
        kset = set(keys)
        qnorm = math.sqrt(sum(self.idf(k) ** 2 for k in kset)) or 1.0
        scored: List[Tuple[int, float]] = []
        for i, ph in enumerate(self.phrases):
            if exclude and i in exclude:
                continue
            if not ph.keys:
                continue
            shared = kset.intersection(ph.keys)
            if not shared:
                continue
            num = sum(self.idf(k) ** 2 for k in shared)
            dnorm = math.sqrt(sum(self.idf(k) ** 2 for k in set(ph.keys))) or 1.0
            scored.append((i, num / (qnorm * dnorm)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:topn]

    # ------------------------------------------------------------------
    # 学習本体
    # ------------------------------------------------------------------
    def learn_text(self, text, weight, origin, as_phrase=False):
        """1 発話を学習し (全token id, 内容語 id, 模倣記憶index) を返す。"""
        toks = self.tok.tokenize(text)
        if not toks:
            return [], [], -1

        ids: List[int] = []
        keys: List[int] = []
        sent: List[int] = []
        for t in toks:
            if is_noise(t.surface):
                continue
            wid = self.ensure_word(t, origin)
            if wid == UNK:
                continue
            ids.append(wid)
            if self.is_content_id(wid):
                keys.append(wid)
            sent.append(wid)
            if is_terminator(t.surface):
                # 終止記号も列に含めて学習する。こうすることで
                # 「…だ → 。 → EOS」という文の閉じ方そのものを覚えられる。
                self.observe(sent, weight)
                sent = []
            if t.pos == "感動詞" and origin == ORIGIN_OPERATOR:
                self.interjections[wid] = self.interjections.get(wid, 0.0) + 1.0
        if sent:
            self.observe(sent, weight)

        self.stimulate(ids, scale=min(1.0, weight))
        if origin == ORIGIN_WEB:
            self.web_tokens += len(ids)

        self.doc_count += 1.0
        for k in set(keys):
            self.df[k] = self.df.get(k, 0.0) + 1.0

        ph_idx = -1
        if as_phrase and ids:
            is_q = any(ch in "？?" for ch in text)
            ph_idx = self.remember_phrase(text.strip(), ids, keys, origin, is_q)
            if is_q:
                self._learn_question_tail(toks)
        return ids, keys, ph_idx

    def encode(self, text: str):
        """脳を一切書き換えずに、既知語だけで (ids, keys) へ変換する。

        評価や自発発話で「学習せずに考えるだけ」をしたいときに使う。
        """
        ids: List[int] = []
        keys: List[int] = []
        for t in self.tok.tokenize(text):
            if is_noise(t.surface):
                continue
            wid = self.word2id.get(t.surface)
            if wid is None or wid == UNK:
                continue
            ids.append(wid)
            if self.is_content_id(wid):
                keys.append(wid)
        return ids, keys

    def _learn_question_tail(self, toks: List[Tok]) -> None:
        """疑問文の語尾パターンを盗む。

        質問の作り方をハードコードせず、オペレーターの言い回しから学ぶ。
        「名前は？」を教わると、語尾パターン (は, ？) が記憶される。
        """
        tail: List[str] = []
        for t in reversed(toks):
            if is_content(t) or t.pos == "感動詞":
                break
            tail.append(t.surface)
            if len(tail) >= 3:
                break
        tail.reverse()
        if tail and any(ch in "？?" for ch in tail[-1]):
            key = tuple(tail)
            self.question_tails[key] = self.question_tails.get(key, 0.0) + 1.0

    def reinforce_pair(self, ids: Sequence[int], delta: float) -> None:
        """フィードバックによる強化/弱化。"""
        if not ids:
            return
        seq = [BOS] + list(ids) + [EOS]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            row = self.bi.get(a)
            if row is not None and b in row:
                row[b] = max(0.0, row[b] + delta)
                self.bi_tot[a] = max(0.0, self.bi_tot.get(a, 0.0) + delta)
            elif delta > 0:
                self._reinforce(a, b, delta)
            if i + 2 < len(seq):
                c = seq[i + 2]
                trow = self.tri.get((a, b))
                if trow is not None and c in trow:
                    trow[c] = max(0.0, trow[c] + delta)
                elif delta > 0:
                    self._reinforce_tri(a, b, c, delta)

    def shift_valence(self, ids: Iterable[int], delta: float) -> None:
        uniq = sorted({i for i in ids if i >= N_SPECIAL})
        if not uniq:
            return
        idx = torch.tensor(uniq, dtype=torch.long, device=self.device)
        self.valence[idx] = torch.clamp(self.valence[idx] + delta, -5.0, 5.0)

    def mood_of(self, ids: Iterable[int]) -> float:
        uniq = [i for i in ids if 0 <= i < self.vocab_size]
        if not uniq:
            return 0.0
        idx = torch.tensor(uniq, dtype=torch.long, device=self.device)
        return float(self.valence[idx].mean().item())

    def set_flag(self, wid: int, flag: int, on: bool = True) -> None:
        va, vb, vc, origin, flags = unpack_state(int(self.state[wid].item()))
        flags = (flags | flag) if on else (flags & ~flag)
        self.state[wid] = pack_state(va, vb, vc, origin, flags)

    def has_flag(self, wid: int, flag: int) -> bool:
        return bool((int(self.state[wid].item()) >> 27) & flag)

    def origin_of(self, wid: int) -> int:
        return (int(self.state[wid].item()) >> 24) & 0x07

    # ------------------------------------------------------------------
    # 遷移分布
    # ------------------------------------------------------------------
    def next_candidates(self, prev: int, prev2: Optional[int]) -> Dict[int, float]:
        """3-gram を優先し 2-gram でバックオフした生スコア表。

        意味ベクトルがあるときは、さらに「似た語がどう続いたか」で補間する
        (クラスベース・バックオフ)。未観測の遷移が埋まるので、語彙が薄い
        時期の行き止まりが大きく減る。
        """
        out: Dict[int, float] = {}
        if prev2 is not None:
            tri_row = self.tri.get((prev2, prev))
            if tri_row:
                w = self.cfg.tri_weight
                for c, v in tri_row.items():
                    out[c] = out.get(c, 0.0) + w * math.log1p(v)
        bi_row = self.bi.get(prev)
        if bi_row:
            w = self.cfg.bi_weight
            for c, v in bi_row.items():
                out[c] = out.get(c, 0.0) + w * math.log1p(v)

        emb = self.embed
        if emb is not None and emb.ready() and self.cfg.class_backoff > 0:
            for near, sim in emb.neighbors(prev, 4):
                row = self.bi.get(near)
                if not row:
                    continue
                w = self.cfg.class_backoff * sim * self.cfg.bi_weight
                for c, v in row.items():
                    out[c] = out.get(c, 0.0) + w * math.log1p(v)
        return out

    def out_degree(self, wid: int) -> int:
        return len(self.bi.get(wid, ()))

    def knowledge_score(self, wid: int) -> float:
        """その語について「どれだけ喋れるか」。低いほど好奇心の対象。"""
        return math.log1p(self.out_degree(wid)) + 0.5 * math.log1p(
            len(self.assoc.get(wid, ())))

    # ------------------------------------------------------------------
    def top_words(self, n=12, origin=None) -> List[Tuple[str, int]]:
        if self.vocab_size <= N_SPECIAL:
            return []
        va, _, _, org, _ = unpack_state(self.state[N_SPECIAL: self.vocab_size])
        va = va.to(torch.float32)
        if origin is not None:
            va = torch.where(org == origin, va, torch.zeros_like(va))
        k = min(n, va.numel())
        vals, idx = torch.topk(va, k)
        out = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            if v <= 0:
                continue
            out.append((self.id2word[i + N_SPECIAL], int(v)))
        return out

    def origin_counts(self) -> Dict[int, int]:
        if self.vocab_size <= N_SPECIAL:
            return {}
        _, _, _, org, _ = unpack_state(self.state[N_SPECIAL: self.vocab_size])
        vals, counts = torch.unique(org, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())}

    def stats(self) -> Dict[str, float]:
        return {
            "vocab": self.learned_vocab,
            "edges": self.edges,
            "bi_rows": len(self.bi),
            "tri_rows": len(self.tri),
            "assoc_rows": len(self.assoc),
            "phrases": len(self.phrases),
            "turns": self.turns,
            "tokens": self.total_tokens,
            "web_tokens": self.web_tokens,
        }
