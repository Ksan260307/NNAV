"""ネットナビ本体。脳・成長・気分・ネット学習・永続化を束ねる。"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .brain import NaviBrain
def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


from .comprehension import Comprehension
from .config import FLAG_PINNED, ORIGIN_NAVI, ORIGIN_OPERATOR, ORIGIN_WEB, Config
from .deixis import Deixis
from .discourse import Discourse
from .embed import Embedding
from .emotion import Mood
from .facts import Fact, FactStore
from .frames import CaseFrames
from .intents import Intents
from .lexicon import Lexicon
from .neural import Neural
from .parse import FunctionWords, parse
from .realize import Realizer
from .senses import Senses
from . import facts as _facts_mod
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
        self.realizer = Realizer()
        # 機能語(疑問詞・人称・指示詞)。種を切ると自力で見つけるしかなくなる。
        self.lexicon = Lexicon()
        self.seed_fw = (FunctionWords.seeded() if self.cfg.bootstrap_function_words
                        else FunctionWords.empty())
        self.fw = FunctionWords(*(set(x) for x in
                                  (self.seed_fw.questions, self.seed_fw.first,
                                   self.seed_fw.second, self.seed_fw.deictics)))
        _facts_mod.set_function_words(self.fw)
        self.deixis = Deixis(self.cfg.use_person_words, self.fw)
        self.discourse = Discourse(self.cfg.use_demonstratives, self.fw)
        self.intents = Intents(self.cfg.intent_k)
        self.senses = Senses(self.cfg.senses_enabled)
        self.neural = Neural(self.cfg.neural_dim, self.cfg.neural_hidden)
        self.last_intent = -1
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
        self._last_env = None
        self._last_anaphora = 0
        # 直前のやりとり(返し方の評価に使う)
        # (入力意図, 返答意図, 返答の内容語, 入力テキスト, 入力の内容語)
        self._pending: Optional[tuple] = None
        self.novelty = 0.0               # 新規発話率(記憶の再生でない発話の割合)
        self.replies = 0
        self.debug_pool = False          # True にすると候補一式を last_pool に残す
        self.last_pool: List = []
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
            self.realizer.load(extra.get("realizer", {}) or {})
            self.deixis.load(extra.get("deixis", {}) or {})
            self.discourse.load(extra.get("discourse", {}) or {})
            self.intents.load(extra.get("intents", {}) or {})
            self.senses.load(extra.get("senses", {}) or {})
            self.neural.load(extra.get("neural", {}) or {})
            self.lexicon.load(extra.get("lexicon", {}) or {})
            self.refresh_function_words()
            self.last_intent = int(extra.get("last_intent", -1))
            self.novelty = float(extra.get("novelty", 0.0))
            self.replies = int(extra.get("replies", 0))
            self.growth.novelty = self.novelty
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
    def _grade_last_reply(self, text: str, keys) -> Optional[float]:
        """前回の返し方を、オペレーターの反応から採点する。

        独話の意図遷移を学んでも「問いかけにどう返すべきか」は出てこない。
        本当の教師信号は、返した **後** にオペレーターが取る行動のほう。
        """
        p = self._pending
        self._pending = None
        if not p:
            return None
        in_i, reply_i, reply_keys, in_text, in_keys = p
        if in_i < 0 or reply_i < 0:
            return None
        now = set(keys)
        # 1) さっきの問いかけを言い直した = 答えになっていなかった
        if in_text and (text.strip() == in_text.strip() or _jaccard(now, in_keys) > 0.7):
            r = -1.0
        # 2) こちらが出した語を拾って続けた = 会話が転がった
        elif reply_keys and (now & set(reply_keys)):
            r = 1.0
        # 3) 話題がまるごと切り替わった
        else:
            r = -0.3
        self.intents.observe_reward(in_i, reply_i, r)
        return r

    def _env(self):
        return self.senses.env_now(self.mood.last_input_at)

    def _interpret(self, text: str, speaker: int) -> List:
        """文を述語項構造にし、人称を役割に畳み、指示詞を解決する。

        学習と発話生成の両方がここを通るので、解釈は必ず一致する。
        """
        if not self.cfg.semantics_enabled:
            return []
        toks = self.brain.tok.tokenize(text)
        preds = parse(toks, self.fw)
        asking = (any(ch in "？?" for ch in text)
                  or any(t.surface in self.fw.questions for t in toks))
        if self.cfg.discover_function_words:
            # 人称の観測は畳む前、穴の観測は畳んだ後(事実ストアと鍵を揃える)
            self.lexicon.observe(toks, preds, asking)
        self.deixis.apply(preds, speaker)
        if self.cfg.discover_function_words:
            self.lexicon.observe_slots(preds, asking)
        if preds:
            n = self.discourse.resolve(preds, self.frames, self.embed,
                                       self.brain, self.brain.turns)
            if n:
                self._last_anaphora += n
        return preds

    def _apply(self, rec: Dict[str, Any]) -> None:
        op = rec.get("op")
        b = self.brain
        if op == "learn":
            text, w, origin = rec["x"], rec["w"], rec["o"]
            self._last_anaphora = 0
            preds = self._interpret(text, origin)

            # --- prequential 評価: 学習する前に予測させて当たり具合を測る ---
            if self.cfg.semantics_enabled and origin == ORIGIN_OPERATOR:
                pre_ids, _ = b.encode(text)
                self.comp.observe(b, self.frames, pre_ids, preds)
                if self.cfg.neural_enabled:
                    self.neural.observe(pre_ids)

            self._last_learn = b.learn_text(text, w, origin, bool(rec.get("p", False)))
            ids = self._last_learn[0]

            if self.cfg.semantics_enabled:
                self.frames.observe(preds, w)
                self.frames.observe_surface(b.tok.tokenize(text), w)
                self.realizer.observe(preds, w)
                learned, clashes = self.facts.observe(preds, origin, b.turns)
                for f in learned:
                    self.lexicon.note_fact(f.subj, f.rel, f.obj)
                if learned:
                    self.deixis.learn_names(self.facts)
                self.discourse.observe(preds, b.turns)
                self._last_preds = preds
                self._last_facts = learned
                self._last_clashes = clashes
                # 照応の解決率(理解度の 1 軸)
                if self.discourse.attempted:
                    self.comp.anaphora_rate = self.discourse.rate()
                    self.comp.anaphora_samples = self.discourse.attempted

            if origin in (ORIGIN_OPERATOR, ORIGIN_WEB) and self.cfg.neural_enabled:
                self.neural.corpus.add(ids)

            if origin == ORIGIN_OPERATOR:
                if self.cfg.senses_enabled:
                    env = self._env()
                    self.senses.observe(ids, env, w)
                if self.cfg.intents_enabled and ids:
                    vec = self.intents.featurize(b, ids, preds)
                    cur = self.intents.classify(vec) if self.intents.ready() else -1
                    if cur >= 0:
                        if self.last_intent >= 0:
                            self.intents.evaluate(self.last_intent, cur)
                            self.intents.observe_pair(self.last_intent, cur)
                        self.last_intent = cur
                    self.intents.add_sample(vec, text)
                    self.comp.intent_ready = self.intents.measurable()
                    self.comp.intent_skill = self.intents.appropriateness()
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
                    self.realizer.decay(self.cfg.synapse_decay, self.cfg.synapse_prune)
                if self.cfg.senses_enabled:
                    self.senses.decay(self.cfg.synapse_decay)

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

            if learn and self.cfg.intents_enabled:
                # 0. 前回の返し方を、いまの反応から採点する
                _, pre_keys = self.brain.encode(text)
                self._grade_last_reply(text, pre_keys)

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
            self.growth.novelty = self.novelty
            stage = self.growth.stage(self.brain.learned_vocab, self.brain.turns)
            params = self._params(stage)
            in_preds = (self._last_preds if learn
                        else self._interpret(text, ORIGIN_OPERATOR))
            sem = self.cfg.semantics_enabled
            ground = None
            if self.cfg.senses_enabled and params.ground_weight > 0:
                g = self.senses.score_all(self._env(), self.brain.vocab_size)
                ground = g.tolist() if g is not None else None
            intent_in = -1
            if self.cfg.intents_enabled and self.intents.ready() and ids:
                intent_in = self.intents.classify(
                    self.intents.featurize(self.brain, ids, in_preds))
            ctx = Context(
                input_ids=ids, input_keys=keys, input_text=text,
                recent_texts=list(self.recent_replies),
                topic_ids=[t for t, _ in self.brain.related(keys, topn=8)],
                energy=self.mood.energy, valence=self.mood.valence,
                asking=any(ch in "？?" for ch in text),
                input_preds=in_preds,
                input_surfaces={a.head for p_ in in_preds for a in p_.args if a.head},
                fw=self.fw,
                frames=self.frames if sem else None,
                embed=self.embed if sem else None,
                facts=self.facts if sem else None,
                realizer=self.realizer if sem else None,
                deixis=self.deixis if sem else None,
                intents=self.intents if self.cfg.intents_enabled else None,
                intent_in=intent_in,
                ground=ground,
                turn=self.brain.turns,
            )
            pool = [] if self.debug_pool else None
            cand = generate(self.brain, ctx, params, self.rng,
                            neural=self.neural if self.cfg.neural_enabled else None,
                            pool_out=pool)
            if pool is not None:
                pool.sort(key=lambda c: -c.score)
                self.last_pool = pool

            if not learn:
                return Reply(cand.text, cand.annotated, cand.strategy, cand.score,
                             cand.is_question, None, 0, len(cand.ids), [], [],
                             cand.detail)

            # 5. 自分の発話の流暢さを成長の指標に反映
            if cand.ids:
                self._do("coh", f=fluency(self.brain, cand.ids))
            self._do("turn")

            if self.cfg.intents_enabled and intent_in >= 0:
                self._pending = (intent_in, int(cand.detail.get("intent", -1)),
                                 [i for i in cand.ids if self.brain.is_content_id(i)],
                                 text, list(keys))
            if cand.ids:
                fresh_ = 1.0 if cand.detail.get("new", 0.0) > 0 else 0.0
                self.replies += 1
                a = 0.05 if self.replies > 20 else 1.0 / self.replies
                self.novelty = (1 - a) * self.novelty + a * fresh_
                self.growth.novelty = self.novelty
            self.last_reply_ids = list(cand.ids)
            self.last_reply_text = cand.text
            self.recent_replies.append(cand.text)
            del self.recent_replies[: -self.cfg.recent_reply_memory]
            self.prev_input_keys = keys
            self.last_phrase_idx = ph_idx

            stage_up = self.growth.observe(self.brain.learned_vocab, self.brain.turns)
            self._maybe_build_embedding()
            if (self.cfg.intents_enabled and not self.intents.ready()
                    and self.brain.turns % 50 == 0):
                self.intents.fit(self.rng)
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
            p.plan_bias = self.cfg.plan_bias
            p.plan_candidates = self.cfg.plan_candidates
        else:
            p.frame_weight = p.semantic_weight = 0.0
            p.allow_plan = False
        if self.cfg.senses_enabled:
            p.ground_weight = self.cfg.ground_weight
        if self.cfg.intents_enabled:
            p.intent_weight = self.cfg.intent_weight
        if self.cfg.neural_enabled:
            # 学習前の発話をどちらがよく言い当てたかで、採点の主導権が移る
            p.nn_weight = self.neural.alpha(self.comp.logprob)
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
            if self.cfg.intents_enabled and self._pending:
                in_i, reply_i = self._pending[0], self._pending[1]
                if in_i >= 0 and reply_i >= 0:
                    self.intents.observe_reward(in_i, reply_i,
                                                2.0 if positive else -2.0)
                self._pending = None
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
            # 眠っている間に意味空間・意図クラスタを編成し直す
            self._maybe_build_embedding()
            if self.cfg.intents_enabled and not self.intents.ready():
                self.intents.fit(self.rng)
            cand = soliloquy(self.brain, self._params(stage), self.rng)
            if not cand or not cand.ids or len(cand.ids) < 2:
                return None
            # 自分の出力での自己強化は暴走(モード崩壊)を招くため、ごく弱く
            self._do("fb", i=cand.ids, d=self.cfg.w_dream)
            self.dream_count += 1
            return cand.text

    def deep_sleep(self) -> Dict[str, Any]:
        """深い睡眠。意味空間・意図クラスタを貼り直し、ニューラルを学習する。

        重い処理はすべてここに集めてある。会話中は一切走らない。
        """
        out: Dict[str, Any] = {}
        with self.lock:
            # 脳を読む処理だけロックの内側で済ませる
            if self._maybe_build_embedding(force=True):
                out["embed"] = self.embed.builds
            if self.cfg.intents_enabled and self.intents.fit(self.rng):
                out["intents"] = self.intents.k
            found = self.refresh_function_words()
            if found:
                out["lexicon"] = found
            build = self.cfg.neural_enabled and self.neural.ensure(self.brain,
                                                                   self.embed)
        # 学習はロックの外。ここを握ったままだと、眠っている間に
        # 話しかけられたオペレーターが数十秒待たされる。
        if self.cfg.neural_enabled and self.neural.ready():
            r = self.neural.train(self.cfg.neural_train_sec, rng=self.rng)
            if r.get("steps"):
                out["neural"] = r
            elif build:
                out["neural"] = {"steps": 0, "loss": 0.0, "sec": 0.0}
        return out

    def refresh_function_words(self) -> Dict[str, int]:
        """分布から見つけた機能語を、種に足して有効化する。"""
        if not self.cfg.discover_function_words:
            return {}
        found = self.lexicon.discovered(self.frames, self.embed, self.brain)
        merged = self.seed_fw.merged(found)
        added = {
            "questions": len(merged.questions - self.fw.questions),
            "first": len(merged.first - self.fw.first),
            "second": len(merged.second - self.fw.second),
            "deictics": len(merged.deictics - self.fw.deictics),
        }
        self.fw.questions, self.fw.first = merged.questions, merged.first
        self.fw.second, self.fw.deictics = merged.second, merged.deictics
        _facts_mod.set_function_words(self.fw)
        return {k: v for k, v in added.items() if v}

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
                "realizer": self.realizer.to_dict(),
                "deixis": self.deixis.to_dict(),
                "discourse": self.discourse.to_dict(),
                "intents": self.intents.to_dict(),
                "senses": self.senses.to_dict(),
                "neural": self.neural.to_dict(),
                "lexicon": self.lexicon.to_dict(),
                "last_intent": self.last_intent,
                "novelty": self.novelty, "replies": self.replies,
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
            self.growth.novelty = self.novelty
            cur, nxt, ratio = self.growth.progress(vocab, turns)
            st = self.brain.stats()
            return {
                "stage": cur, "next": nxt, "progress": ratio,
                "vocab": vocab, "turns": turns,
                "edges": st["edges"], "phrases": st["phrases"],
                "assoc": st["assoc_rows"],
                "tokens": st["tokens"], "web_tokens": st["web_tokens"],
                "coherence": self.growth.coherence_ema,
                "novelty": self.novelty,
                "comprehension": self.comp.detail(),
                "frames": self.frames.stats(),
                "facts": self.facts.stats(),
                "embed_ready": self.embed.ready(),
                "embed_builds": self.embed.builds,
                "conflicts": self.facts.conflicts[-3:],
                "realizer": self.realizer.stats(),
                "discourse": self.discourse.stats(),
                "intents": self.intents.stats(),
                "senses": self.senses.stats(),
                "neural": self.neural.stats(),
                "alpha": self.neural.alpha(self.comp.logprob),
                "deixis": {"names": dict(self.deixis.names)},
                "lexicon": self.lexicon.stats(),
                "function_words": self.fw.counts(),
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
