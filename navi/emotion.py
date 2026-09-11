"""気分。

感情語の辞書は一切持たない。「何が嬉しい言葉か」はオペレーターからの
フィードバック(/good /bad)と、対話のリズムだけから学習する。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class Mood:
    energy: float = 0.55      # 元気(0..1) 対話の頻度で上下する
    curiosity: float = 0.6    # 好奇心(0..1) 知らない語に触れると上がる
    bond: float = 0.0         # 親密度(0..) 会話を重ねるほど蓄積
    valence: float = 0.0      # 機嫌(-1..1) 称賛/叱責で上下
    last_input_at: float = field(default_factory=time.time)

    # ------------------------------------------------------------------
    def on_input(self, n_tokens: int, n_new_words: int, word_valence: float) -> None:
        now = time.time()
        gap = now - self.last_input_at
        self.last_input_at = now

        # 話しかけられる間隔が短いほど元気になる
        if gap < 120:
            self.energy = min(1.0, self.energy + 0.06)
        elif gap > 3600:
            self.energy = max(0.15, self.energy - 0.12)

        # 知らない言葉に出会うと好奇心が上がる
        if n_new_words > 0:
            self.curiosity = min(1.0, self.curiosity + 0.05 * math.log1p(n_new_words))
        else:
            self.curiosity = max(0.1, self.curiosity - 0.01)

        self.bond = min(100.0, self.bond + 0.08 + 0.01 * min(n_tokens, 20))
        # 入力に含まれる語の感情価に引きずられる
        self.valence = max(-1.0, min(1.0, 0.9 * self.valence + 0.1 * word_valence))

    def on_feedback(self, positive: bool) -> None:
        delta = 0.35 if positive else -0.35
        self.valence = max(-1.0, min(1.0, self.valence + delta))
        self.energy = min(1.0, self.energy + (0.05 if positive else -0.02))
        self.bond = min(100.0, self.bond + (0.5 if positive else 0.1))

    def on_study(self, learned_tokens: int) -> None:
        if learned_tokens > 0:
            self.curiosity = max(0.1, self.curiosity - 0.08)
            self.energy = max(0.1, self.energy - 0.02)

    def tick(self, idle_sec: float) -> None:
        """時間経過による自然減衰。放置されると元気が落ち、好奇心が溜まる。"""
        if idle_sec > 600:
            self.energy = max(0.15, self.energy - 0.01)
            self.curiosity = min(1.0, self.curiosity + 0.004)
            self.valence *= 0.995

    # ------------------------------------------------------------------
    def temperature_bias(self) -> float:
        """元気なほど発話が奔放に、落ち込むほど硬くなる。"""
        return (self.energy - 0.5) * 0.22 + (self.valence * 0.06)

    def length_scale(self) -> float:
        """文の長さへの補正。段階ごとの上限を壊さないよう倍率で効かせる。"""
        if self.energy > 0.8:
            return 1.12
        if self.energy < 0.3:
            return 0.85
        return 1.0

    def label(self) -> str:
        if self.valence > 0.45:
            return "ごきげん"
        if self.valence < -0.45:
            return "しょんぼり"
        if self.energy > 0.78:
            return "げんき"
        if self.energy < 0.28:
            return "ねむい"
        if self.curiosity > 0.8:
            return "そわそわ"
        return "ふつう"

    def to_dict(self) -> dict:
        return {"energy": self.energy, "curiosity": self.curiosity,
                "bond": self.bond, "valence": self.valence}

    def load(self, d: dict) -> None:
        self.energy = float(d.get("energy", 0.55))
        self.curiosity = float(d.get("curiosity", 0.6))
        self.bond = float(d.get("bond", 0.0))
        self.valence = float(d.get("valence", 0.0))
        self.last_input_at = time.time()
