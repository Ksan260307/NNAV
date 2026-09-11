"""ネットナビ本体。脳・成長・気分・ネット学習・永続化を束ねる。"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .brain import NaviBrain
from .comprehension import Comprehension
from .config import FLAG_PINNED, ORIGIN_NAVI, ORIGIN_OPERATOR, Config
from .embed import Embedding
from .emotion import Mood
from .facts import Fact, FactStore
from .frames import CaseFrames
from .parse import parse
from .generate import SILENCE, Context, GenParams, fluency, generate, soliloquy
from .growth import Growth, Stage
from .store import Journal, load_snapshot, save_snapshot
from .web import StudyResult, WebLearner


@dataclass
class Reply:
    text: str
    annotated: str
    strategy: str = "silence"
    score: float = 0.0
    is_question: bool = False
    stage_up: Optional[Stage] = None
    new_words: int = 0
    n_tokens: int = 0
    facts: List[Fact] = field(default_factory=list)       # この発話から覚えた事実
    clashes: List[tuple] = field(default_factory=list)    # 見つかった矛盾
    detail: Dict[str, float] = field(default_factory=dict)


class NetNavi:
    def __init__(self, data_dir: str = "navi_data", cfg: Optional[Config] = None):
        self.cfg = cfg or Config.load(data_dir)
        self.cfg.save()
        self.rng = random.Random(self.cfg.seed or None)
        self.lock = threading.RLock()

        self.brain = NaviBrain(self.cfg)
        self.growth = Growth()
        self.mood = Mood()

        # --- 意味層 -------------------------------------------------------
        self.frames = CaseFrames()
        self.facts = FactStore()
        self.embed = Embedding(self.cfg.embed_dim)
        self.comp = Comprehension()
        if self.cfg.semantics_enabled:
            self.brain.embed = self.embed
        self.web = WebLearner(self.cfg, self.brain, self.mood,
                              should_stop=lambda: self._stopping)
        self._stopping = False

        self.journal = Journal(self.cfg.file("wal.jsonl"), self.cfg.wal_flush_every)
        self.recent_replies: List[str] = []
        self.last_reply_ids: List[int] = []
        self.last_reply_text: str = ""
        self.last_phrase_idx: int = -1
        self.prev_input_keys: List[int] = []
        self._last_learn = ([], [], -1)
        self._last_preds: List = []
        self._last_facts: List[Fact] = []
        self._last_clashes: List[tuple] = []
        self._last_embed_turn = 0
        self.pending: List[str] = []          # 自発発話のキュー
        self.session_started = time.time()
        self.dream_count = 0
        self._dream_hour = time.time()

        self._restore()

    # ------------------------------------------------------------------
    # 起動: スナップショット + WAL 再生
    # ------------------------------------------------------------------
    def _restore(self) -> None:
        snap = self.cfg.file("snapshot.pt")
        extra: Dict[str, Any] = {}
        saved_at = 0.0
        if os.path.exists(snap):
            extra = load_snapshot(self.brain, self.growth, self.mood, snap)
            if extra or self.brain.vocab_size > 3:
                saved_at = extra.get("saved_at", 0.0) if isinstance(extra, dict) else 0.0
        if extra:
            self.web.load(extra.get("web", {}))
            self.prev_input_keys = list(extra.get("prev_input_keys", []))
            self.last_phrase_idx = int(extra.get("last_phrase_idx", -1))
            self.frames.load(extra.get("frames", {}) or {})
            self.facts.load(extra.get("facts", {}) or {})
            self.embed.load(extra.get("embed", {}) or {})
            self.comp.load(extra.get("comprehension", {}) or {})
            self._last_embed_turn = int(extra.get("last_embed_turn", 0))

        def apply(rec: Dict[str, Any]) -> None:
            if saved_at and rec.get("ts", 0.0) <= saved_at:
                return  # スナップショットに既に含まれている
            self._apply(rec)

        n = self.journal.replay(apply)
        if self.brain.vocab_size > 3 or n:
            print(f"[System] 電脳メモリを復元しました。"
                  f"語彙 {self.brain.learned_vocab} 語 / 対話 {self.brain.turns} ターン"
                  f"{f' / 差分 {n} 件を再生' if n else ''}")
        else:
            print("[System] 新規ニューラルネットワークを初期化。ゼロから学習を開始します。")

    # ------------------------------------------------------------------
    # 出来事の適用(ライブ実行と WAL 再生で同じ経路を通す)
    # ------------------------------------------------------------------
    def _apply(self, rec: Dict[str, Any]) -> None:
        op = rec.get("op")
        b = self.brain
        if op == "learn":
            text, w, origin = rec["x"], rec["w"], rec["o"]
            preds = parse(b.tok.tokenize(text)) if self.cfg.semantics_enabled else []

            # --- prequential 評価: 学習する前に予測させて当たり具合を測る ---
            if self.cfg.semantics_enabled and origin == ORIGIN_OPERATOR:
                pre_ids, _ = b.encode(text)
                self.comp.observe(b, self.frames, pre_ids, preds)

            self._last_learn = b.learn_text(text, w, origin, bool(rec.get("p", False)))

            if self.cfg.semantics_enabled:
                self.frames.observe(preds, w)
                learned, clashes = self.facts.observe(preds, origin, b.turns)
                self._last_preds = preds
                self._last_facts = learned
                self._last_clashes = clashes
        elif op == "assoc":
            b.associate(rec["a"], rec["b"], rec["w"])
        elif op == "link":
            b.link_phrases(rec["p"], rec["n"])
        elif op == "fb":
            b.reinforce_pair(rec["i"], rec["d"])
        elif op == "val":
            b.shift_valence(rec["i"], rec["d"])
        elif op == "flag":
            b.set_flag(rec["i"], rec["f"], bool(rec.get("on", True)))
        elif op == "coh":
            self.growth.update_coherence(rec["f"])
        elif op == "turn":
            b.turns += 1
            if b.turns % self.cfg.decay_every_turns == 0:
                b.decay()
                if self.cfg.semantics_enabled:
                    self.frames.decay(self.cfg.synapse_decay, self.cfg.synapse_prune)

    def _do(self, op: str, **fields: Any) -> None:
        """出来事を記録してから適用する。"""
        rec = {"op": op}
        rec.update(fields)
        self.journal.record(op, **fields)
        self._apply(rec)

    # ------------------------------------------------------------------
    # 対話
    # ------------------------------------------------------------------
    def talk(self, text: str, learn: bool = True) -> Reply:
        """話しかけられて返事をする。

        learn=False にすると脳を一切書き換えずに「考えるだけ」になる。
        育ち具合を測りたいときの評価用。
        """
        text = text.strip()
        if not text:
            return Reply(SILENCE, SILENCE)
        with self.lock:
            before_vocab = self.brain.vocab_size

            if learn:
                # 1. オペレーターの言葉を学ぶ(最優先・最大の重み)
                self._do("learn", x=text, w=self.cfg.w_operator,
                         o=ORIGIN_OPERATOR, p=True)
                ids, keys, ph_idx = self._last_learn
                new_words = self.brain.vocab_size - before_vocab

                # 2. 会話の流れを覚える: 前の発話 -> 今の発話
                if self.last_phrase_idx >= 0 and ph_idx >= 0:
                    self._do("link", p=self.last_phrase_idx, n=ph_idx)
                # 3. 話題の連続性(人間が書いた文どうしの共起だけを連想野に入れる)
                if self.prev_input_keys and keys:
                    self._do("assoc", a=self.prev_input_keys, b=keys, w=1.0)
                    self._do("assoc", a=keys, b=self.prev_input_keys, w=0.5)

                self.mood.on_input(len(ids), new_words, self.brain.mood_of(keys))
                facts_learned = list(self._last_facts)
                clashes = list(self._last_clashes)
            else:
                ids, keys = self.brain.encode(text)
                ph_idx, new_words = -1, 0
                facts_learned, clashes = [], []

            # 4. 考えてから喋る
            self.growth.understanding = self.comp.understanding()
            stage = self.growth.stage(self.brain.learned_vocab, self.brain.turns)
            params = self._params(stage)
            ctx = Context(
                input_ids=ids, input_keys=keys, input_text=text,
                recent_texts=list(self.recent_replies),
                topic_ids=[t for t, _ in self.brain.related(keys, topn=8)],
                energy=self.mood.energy, valence=self.mood.valence,
                asking=any(ch in "？?" for ch in text),
                input_preds=(parse(self.brain.tok.tokenize(text))
                             if self.cfg.semantics_enabled else []),
                frames=self.frames if self.cfg.semantics_enabled else None,
                embed=self.embed if self.cfg.semantics_enabled else None,
                facts=self.facts if self.cfg.semantics_enabled else None,
                turn=self.brain.turns,
            )
            cand = generate(self.brain, ctx, params, self.rng)

            if not learn:
                return Reply(cand.text, cand.annotated, cand.strategy, cand.score,
                             cand.is_question, None, 0, len(cand.ids), [], [],
                             cand.detail)

            # 5. 自分の発話の流暢さを成長の指標に反映
            if cand.ids:
                self._do("coh", f=fluency(self.brain, cand.ids))
            self._do("turn")

            self.last_reply_ids = list(cand.ids)
            self.last_reply_text = cand.text
            self.recent_replies.append(cand.text)
            del self.recent_replies[: -self.cfg.recent_reply_memory]
            self.prev_input_keys = keys
            self.last_phrase_idx = ph_idx

            stage_up = self.growth.observe(self.brain.learned_vocab, self.brain.turns)
            self._maybe_build_embedding()
            self._maybe_compact()
            return Reply(cand.text, cand.annotated, cand.strategy, cand.score,
                         cand.is_question, stage_up, new_words, len(cand.ids),
                         facts_learned, clashes, cand.detail)

    def _params(self, stage: Stage) -> GenParams:
        p = GenParams(**vars(stage.params))
        p.temperature = max(0.45, p.temperature + self.mood.temperature_bias())
        p.max_len = max(p.min_len + 1, int(p.max_len * self.mood.length_scale()))
        p.mutation_rate = min(0.4, p.mutation_rate * (0.6 + self.mood.curiosity))
        p.repetition_penalty = self.cfg.repetition_penalty
        if self.cfg.semantics_enabled:
            p.frame_weight = self.cfg.frame_weight
            p.semantic_weight = self.cfg.semantic_weight
            p.fact_bias = self.cfg.fact_answer_bias
        else:
            p.frame_weight = p.semantic_weight = 0.0
        return p

    # ------------------------------------------------------------------
    def _maybe_build_embedding(self, force: bool = False) -> bool:
        """意味空間を組み直す。重いので間隔を空ける(通常は夢の中で走る)。"""
        if not self.cfg.semantics_enabled:
            return False
        turns = self.brain.turns
        if not force:
            if turns - self._last_embed_turn < self.cfg.embed_rebuild_turns:
                return False
            if not self.embed.stale(self.brain.vocab_size):
                return False
        self._last_embed_turn = turns
        return self.embed.build(self.brain, self.frames)

    # ------------------------------------------------------------------
    # フィードバック
    # ------------------------------------------------------------------
    def feedback(self, positive: bool) -> str:
        with self.lock:
            if not self.last_reply_ids:
                return "まだ評価できる発話がありません。"
            delta = self.cfg.w_feedback * (1.0 if positive else -1.0)
            self._do("fb", i=self.last_reply_ids, d=delta)
            self._do("val", i=self.last_reply_ids, d=0.35 if positive else -0.35)
            self.mood.on_feedback(positive)
            if positive and self.last_reply_text:
                # 褒められた発話は「自分の言葉」として記憶に残す
                self._do("learn", x=self.last_reply_text, w=self.cfg.w_operator * 0.5,
                         o=ORIGIN_NAVI, p=True)
            return ("シナプスを強化しました。" if positive
                    else "シナプスを弱めました。")

    def pin(self, word: str) -> str:
        with self.lock:
            wid = self.brain.word2id.get(word)
            if wid is None:
                return f"「{word}」はまだ知りません。"
            self._do("flag", i=wid, f=FLAG_PINNED, on=True)
            return f"「{word}」を忘却の対象から外しました。"

    # ------------------------------------------------------------------
    # 自律行動
    # ------------------------------------------------------------------
    def dream(self) -> Optional[str]:
        """擬似睡眠中の自己対話。既存のシナプスをごく弱く再強化する。"""
        with self.lock:
            now = time.time()
            if now - self._dream_hour >= 3600:
                self._dream_hour, self.dream_count = now, 0
            if self.dream_count >= self.cfg.dream_max_per_hour:
                return None
            stage = self.growth.stage(self.brain.learned_vocab, self.brain.turns)
            if stage.level < 1:
                return None
            # 眠っている間に意味空間を編成し直す
            self._maybe_build_embedding()
            cand = soliloquy(self.brain, self._params(stage), self.rng)
            if not cand or not cand.ids or len(cand.ids) < 2:
                return None
            # 自分の出力での自己強化は暴走(モード崩壊)を招くため、ごく弱く
            self._do("fb", i=cand.ids, d=self.cfg.w_dream)
            self.dream_count += 1
            return cand.text

    def study(self, topic: Optional[str] = None) -> StudyResult:
        """インターネットで調べ物をする。

        ネットワーク待ちのあいだロックを握らないので、常駐スレッドが
        調べ物をしていてもオペレーターは会話を続けられる。
        """
        with self.lock:
            stage = self.growth.stage(self.brain.learned_vocab, self.brain.turns)
            if topic is None and stage.level < 3:
                return StudyResult(False, reason="まだ自分で調べる段階ではありません"
                                                 f"(現在: {stage.code})")
            ok, reason = self.web.can_study()
            if not ok:
                return StudyResult(False, topic=topic or "", reason=reason)
            if topic is None:
                topic = self.web.pick_topic()

        fetched = self.web.fetch(topic)          # ← ロックを持たない

        with self.lock:
            result = self.web.absorb(fetched)
            if result.ok:
                # Web 学習の中身は WAL に「出来事」として残さない。
                # 再生のたびに再取得しては本末転倒なので、
                # 学習後の脳をスナップショットとして固める。
                self._snapshot()
            return result

    def speak_up(self) -> Optional[str]:
        """自分から話しかける(成長段階 4 以上)。"""
        with self.lock:
            stage = self.growth.stage(self.brain.learned_vocab, self.brain.turns)
            if stage.level < 4:
                return None
            params = self._params(stage)
            # 最近オペレーターが話していた話題から始める
            ctx = Context(input_keys=self.prev_input_keys,
                          recent_texts=list(self.recent_replies),
                          topic_ids=[t for t, _ in
                                     self.brain.related(self.prev_input_keys, topn=6)])
            cand = generate(self.brain, ctx, params, self.rng)
            if not cand.ids or cand.text == SILENCE:
                return None
            self.recent_replies.append(cand.text)
            del self.recent_replies[: -self.cfg.recent_reply_memory]
            self.last_reply_ids = list(cand.ids)
            self.last_reply_text = cand.text
            return cand.text

    # ------------------------------------------------------------------
    # 一括学習
    # ------------------------------------------------------------------
    def teach_file(self, path: str, origin: int = ORIGIN_OPERATOR) -> int:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        n = 0
        with self.lock, open(path, "r", encoding="utf-8", errors="replace") as f:
            prev_idx = -1
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                self._do("learn", x=line, w=self.cfg.w_operator, o=origin, p=True)
                ids, keys, ph_idx = self._last_learn
                if prev_idx >= 0 and ph_idx >= 0:
                    self._do("link", p=prev_idx, n=ph_idx)
                if self.prev_input_keys and keys:
                    self._do("assoc", a=self.prev_input_keys, b=keys, w=1.0)
                self.prev_input_keys = keys
                prev_idx = ph_idx
                self._do("turn")
                n += 1
            self.growth.observe(self.brain.learned_vocab, self.brain.turns)
            self._maybe_build_embedding(force=True)
            self._snapshot()
        return n

    # ------------------------------------------------------------------
    # 永続化
    # ------------------------------------------------------------------
    def _extra(self) -> Dict[str, Any]:
        return {"web": self.web.to_dict(),
                "prev_input_keys": self.prev_input_keys,
                "last_phrase_idx": self.last_phrase_idx,
                "frames": self.frames.to_dict(),
                "facts": self.facts.to_dict(),
                "embed": self.embed.to_dict(),
                "comprehension": self.comp.to_dict(),
                "last_embed_turn": self._last_embed_turn,
                "saved_at": time.time()}

    def _snapshot(self) -> None:
        save_snapshot(self.brain, self.growth, self.mood, self._extra(),
                      self.cfg.file("snapshot.pt"))
        self.journal.truncate()

    def _maybe_compact(self) -> None:
        if self.journal.lines >= self.cfg.wal_compact_threshold:
            self._snapshot()

    def flush(self) -> None:
        self.journal.flush()

    def save(self) -> None:
        with self.lock:
            self._snapshot()

    def shutdown(self) -> None:
        self._stopping = True
        with self.lock:
            self.journal.flush()
            self._snapshot()

    # ------------------------------------------------------------------
    # 状態表示
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        with self.lock:
            vocab = self.brain.learned_vocab
            turns = self.brain.turns
            self.growth.understanding = self.comp.understanding()
            cur, nxt, ratio = self.growth.progress(vocab, turns)
            st = self.brain.stats()
            return {
                "stage": cur, "next": nxt, "progress": ratio,
                "vocab": vocab, "turns": turns,
                "edges": st["edges"], "phrases": st["phrases"],
                "assoc": st["assoc_rows"],
                "tokens": st["tokens"], "web_tokens": st["web_tokens"],
                "coherence": self.growth.coherence_ema,
                "comprehension": self.comp.detail(),
                "frames": self.frames.stats(),
                "facts": self.facts.stats(),
                "embed_ready": self.embed.ready(),
                "embed_builds": self.embed.builds,
                "conflicts": self.facts.conflicts[-3:],
                "mood": self.mood, "mood_label": self.mood.label(),
                "origins": self.brain.origin_counts(),
                "top_words": self.brain.top_words(10),
                "web_budget": self.web.budget.status(),
                "web_enabled": self.cfg.web_enabled,
                "studied": sorted(self.web.studied.items(), key=lambda kv: -kv[1])[:8],
                "web_log": self.web.log[-5:],
                "wal_lines": self.journal.lines,
                "history": self.growth.history,
            }
