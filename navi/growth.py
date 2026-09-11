"""成長段階。

語彙数・対話ターン数・シナプス数・直近の文脈一貫性から段階を決め、
段階ごとに「できること」を解放していく。いきなり流暢に喋らせないことが
この作品の肝なので、最大文長や利用可能な生成戦略をここで絞る。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .generate import GenParams


@dataclass(frozen=True)
class Stage:
    level: int
    code: str
    name: str
    need_vocab: int
    need_turns: int
    need_understanding: float     # 0..1 (comprehension.understanding)
    need_novelty: float           # 0..1 オペレーターが言っていない文を言う割合
    params: GenParams
    note: str

    def unlocked(self, vocab: int, turns: int, understanding: float,
                 novelty: float) -> bool:
        return (vocab >= self.need_vocab and turns >= self.need_turns
                and understanding >= self.need_understanding
                and novelty >= self.need_novelty)


STAGES: List[Stage] = [
    Stage(0, "EGG", "卵 / 起動直後", 0, 0, 0.0, 0.0,
          GenParams(loo_discount=0.0, novelty_weight=0.0, allow_fact=True, min_len=1, max_len=2, n_candidates=4, temperature=1.15,
                    mutation_rate=0.25, topic_bonus=0.4, allow_mimic=False,
                    allow_echo=True, silence_threshold=-99.0),
          "まだ言葉を知らない。聞いた音を返すのと、教わったことを覚えるだけ。"),
    Stage(1, "LARVA", "幼体 / 単語期", 14, 6, 0.0, 0.0,
          GenParams(loo_discount=0.0, novelty_weight=0.0, allow_fact=True, min_len=1, max_len=4, n_candidates=6, temperature=1.0,
                    mutation_rate=0.16, topic_bonus=0.8, allow_mimic=True,
                    mimic_bias=1.6, allow_echo=True, silence_threshold=-12.0),
          "単語をつなげ始める。オペレーターの口真似が多い。"),
    Stage(2, "CHILD", "幼年 / 文節期", 70, 30, 0.15, 0.0,
          GenParams(modifier_rate=0.15, adverb_rate=0.1, conjunction_rate=0.0, loo_discount=0.3, novelty_weight=0.3, allow_plan=True, allow_fact=True, min_len=2, max_len=8, n_candidates=9, temperature=0.92,
                    mutation_rate=0.10, topic_bonus=1.0, allow_mimic=True,
                    mimic_bias=0.9, mimic_len_penalty=0.06,
                    silence_threshold=-9.5),
          "短い文が形になる。話題を少し引き継げる。"),
    Stage(3, "TEEN", "少年 / 文脈期", 220, 110, 0.30, 0.0,
          GenParams(modifier_rate=0.25, adverb_rate=0.15, conjunction_rate=0.1, loo_discount=0.6, novelty_weight=0.7, allow_plan=True, allow_fact=True, min_len=3, max_len=12, n_candidates=12, temperature=0.86,
                    mutation_rate=0.06, topic_bonus=1.25, allow_mimic=True,
                    mimic_bias=0.45, mimic_len_penalty=0.14,
                    allow_question=True, question_rate=0.16,
                    silence_threshold=-8.5),
          "文脈を追えるようになり、知らない言葉を聞き返す。ネット学習が解禁。"),
    Stage(4, "ADULT", "成体 / 対話期", 650, 320, 0.42, 0.25,
          GenParams(modifier_rate=0.35, adverb_rate=0.25, conjunction_rate=0.2, loo_discount=1.0, novelty_weight=1.0, allow_plan=True, allow_fact=True, min_len=4, max_len=18, n_candidates=14, temperature=0.82,
                    mutation_rate=0.05, topic_bonus=1.4, allow_mimic=True,
                    mimic_bias=0.2, mimic_len_penalty=0.22,
                    allow_question=True, question_rate=0.2,
                    silence_threshold=-8.0),
          "自分から話題を広げる。留守中に調べ物をするようになる。"),
    Stage(5, "NAVI", "ネットナビ / 自律期", 1600, 900, 0.52, 0.4,
          GenParams(modifier_rate=0.4, adverb_rate=0.3, conjunction_rate=0.25, loo_discount=1.0, novelty_weight=1.2, allow_plan=True, allow_fact=True, min_len=4, max_len=22, n_candidates=16, temperature=0.8,
                    mutation_rate=0.045, topic_bonus=1.5, allow_mimic=True,
                    mimic_bias=0.05, mimic_len_penalty=0.28,
                    allow_question=True, question_rate=0.18,
                    silence_threshold=-7.6),
          "自律的に思考し、自分から話しかけてくる。"),
]


class Growth:
    """成長の指標を保持し、現在の段階を決める。"""

    def __init__(self) -> None:
        self.coherence_ema = -9.0     # 直近の発話の流暢さの指数移動平均
        self.understanding = 0.0      # comprehension.py が算出する理解度 0..1
        self.novelty = 0.0            # 新規発話率(記憶の再生ではない発話の割合)
        self.best_level = 0
        self.history: List[tuple] = []  # (turn, level) 昇格ログ

    def update_coherence(self, fluency: float) -> None:
        self.coherence_ema = 0.94 * self.coherence_ema + 0.06 * fluency

    def stage(self, vocab: int, turns: int) -> Stage:
        cur = STAGES[0]
        for st in STAGES:
            if st.unlocked(vocab, turns, self.understanding, self.novelty):
                cur = st
        # 一度上がった段階は下げない(忘却で語彙が減っても人格は退行しない)
        if cur.level < self.best_level:
            cur = STAGES[self.best_level]
        return cur

    def observe(self, vocab: int, turns: int) -> Optional[Stage]:
        """段階が上がったなら、その Stage を返す。"""
        st = self.stage(vocab, turns)
        if st.level > self.best_level:
            self.best_level = st.level
            self.history.append((turns, st.level))
            return st
        return None

    def progress(self, vocab: int, turns: int) -> tuple:
        """(現段階, 次段階, 達成率 0..1) を返す。"""
        cur = self.stage(vocab, turns)
        if cur.level + 1 >= len(STAGES):
            return cur, None, 1.0
        nxt = STAGES[cur.level + 1]
        ratios = [
            min(1.0, vocab / nxt.need_vocab) if nxt.need_vocab else 1.0,
            min(1.0, turns / nxt.need_turns) if nxt.need_turns else 1.0,
        ]
        if nxt.need_understanding > 0:
            ratios.append(min(1.0, self.understanding / nxt.need_understanding))
        if nxt.need_novelty > 0:
            ratios.append(min(1.0, self.novelty / nxt.need_novelty))
        return cur, nxt, sum(ratios) / len(ratios)

    def to_dict(self) -> dict:
        return {"coherence_ema": self.coherence_ema, "best_level": self.best_level,
                "understanding": self.understanding, "novelty": self.novelty,
                "history": self.history}

    def load(self, d: dict) -> None:
        self.coherence_ema = float(d.get("coherence_ema", -9.0))
        self.understanding = float(d.get("understanding", 0.0))
        self.novelty = float(d.get("novelty", 0.0))
        self.best_level = int(d.get("best_level", 0))
        self.history = [tuple(x) for x in d.get("history", [])]
