"""設定値。navi_data/config.json があればそれで上書きされる。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# ニューロンの出自 (state_buffer の 3bit "state" フィールドに格納)
# ---------------------------------------------------------------------------
ORIGIN_SYSTEM = 0    # BOS / EOS / UNK
ORIGIN_OPERATOR = 1  # オペレーター(あなた)が教えた言葉。最も重い。
ORIGIN_WEB = 2       # ナビが自分でインターネットから拾ってきた言葉。
ORIGIN_DREAM = 3     # 擬似睡眠中の自己対話で強化された言葉。
ORIGIN_NAVI = 4      # ナビ自身の発話がオペレーターに肯定された言葉。

ORIGIN_LABEL = {
    ORIGIN_SYSTEM: "系",
    ORIGIN_OPERATOR: "operator",
    ORIGIN_WEB: "web",
    ORIGIN_DREAM: "dream",
    ORIGIN_NAVI: "self",
}

# state_buffer の 3bit "ruin" フィールドをフラグとして流用する
FLAG_CONTENT = 0b001   # 内容語(名詞/動詞/形容詞/副詞)
FLAG_STUDIED = 0b010   # Web学習のトピックとして調査済み
FLAG_PINNED = 0b100    # 忘却禁止(オペレーターが固定した重要語)


@dataclass
class Config:
    # --- 保存先 -----------------------------------------------------------
    data_dir: str = "navi_data"

    # --- 脳の容量 ---------------------------------------------------------
    initial_capacity: int = 4096      # 語彙テンソルの初期確保数(自動で倍々に拡張)
    max_vocab: int = 120_000          # 語彙ニューロン数の上限
    max_edges: int = 4_000_000        # シナプス結合数の上限(超えたら弱い結合から剪定)

    # --- 学習の重み -------------------------------------------------------
    w_operator: float = 1.0           # オペレーターの言葉の学習係数
    w_web: float = 0.35               # Web から拾った文章の学習係数
    w_dream: float = 0.05             # 夢(自己対話)での再強化係数
    w_feedback: float = 2.0           # /good /bad による強化・弱化の係数

    # --- 忘却(エントロピー減衰) -------------------------------------------
    decay_every_turns: int = 4        # 何ターンごとに忘却プロセスを走らせるか
    decay_vc_step: int = 5            # 減衰カウンタ vC の増分
    decay_vc_threshold: int = 100     # vC がこれを超えたら重要度 vA を削る
    decay_va_damage: int = 2          # 1 回の忘却で削る vA
    synapse_decay: float = 0.995      # シナプス結合の自然減衰率
    synapse_prune: float = 0.08       # この強度を下回った結合は忘れ去られる

    # --- 刺激(想起による強化) ---------------------------------------------
    stim_va: int = 40
    stim_vb: int = 10

    # --- 意味層 -----------------------------------------------------------
    semantics_enabled: bool = True    # 格フレーム・事実・意味ベクトル
    use_question_words: bool = True   # 疑問詞の閉じた語彙を使うか
    class_backoff: float = 0.35       # 未観測の遷移を似た語の経験で埋める強さ
    embed_dim: int = 64
    embed_rebuild_turns: int = 150    # 何ターンごとに意味空間を作り直すか
    frame_weight: float = 0.9         # 選択選好の採点重み
    semantic_weight: float = 1.3      # 意味ベクトルによる話題一致の重み
    fact_answer_bias: float = 2.6     # 知っている事実で答えるときの下駄

    # --- 発話生成 ---------------------------------------------------------
    base_temperature: float = 0.85
    mutation_rate: float = 0.05       # 文脈を無視して脳内で気になっている語が漏れる確率
    repetition_penalty: float = 1.6   # 同じ語の繰り返しへのペナルティ
    tri_weight: float = 1.0           # 3-gram シナプスの寄与
    bi_weight: float = 0.45           # 2-gram シナプスの寄与(バックオフ)
    recent_reply_memory: int = 12     # 直近何発話分を「言ったばかり」として避けるか

    # --- インターネット自己学習 -------------------------------------------
    web_enabled: bool = True
    web_user_agent: str = (
        "NetNaviCore/0.3 (personal hobby AI; contact: set your email in config.json)"
    )
    web_min_interval_sec: float = 4.0     # 全体の最小リクエスト間隔
    web_host_interval_sec: float = 8.0    # 同一ホストへの最小リクエスト間隔
    web_max_per_hour: int = 12            # 1時間あたりの上限
    web_max_per_day: int = 120            # 1日あたりの上限
    web_timeout_sec: float = 10.0
    web_max_bytes: int = 512 * 1024
    web_respect_robots: bool = True
    web_max_failures: int = 5             # 連続失敗でサーキットブレーカ作動
    web_breaker_cooldown_sec: float = 1800.0
    web_sentences_per_fetch: int = 6      # 1回の取得で学習する最大文数
    web_study_interval_sec: float = 900.0 # 自発的な調べ物の最小間隔(15分)
    web_rss_feeds: List[str] = field(default_factory=list)  # 任意で追加

    # --- 常駐デーモン -----------------------------------------------------
    daemon_enabled: bool = True
    daemon_tick_sec: float = 15.0
    dream_idle_sec: float = 90.0          # 無操作がこれを超えたら擬似睡眠に入る
    dream_interval_sec: float = 45.0
    dream_max_per_hour: int = 40          # 自己強化の暴走(モード崩壊)を防ぐ上限
    proactive_idle_sec: float = 240.0     # 自発的に話しかけるまでの無操作時間
    proactive_interrupt: bool = True      # 入力待ち中でも割り込んで喋るか

    # --- 永続化 -----------------------------------------------------------
    wal_flush_every: int = 8              # 何イベントごとに WAL をディスクへ流すか
    wal_compact_threshold: int = 4000     # WAL 行数がこれを超えたらスナップショット化

    # --- その他 -----------------------------------------------------------
    seed: int = 0                          # 0 なら非決定的
    device: str = "auto"                   # "auto" / "cpu" / "cuda"

    # ------------------------------------------------------------------
    @property
    def path(self) -> str:
        return os.path.join(self.data_dir, "config.json")

    @classmethod
    def load(cls, data_dir: str = "navi_data") -> "Config":
        cfg = cls(data_dir=data_dir)
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(cfg.path):
            try:
                with open(cfg.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                known = {f_.name for f_ in cls.__dataclass_fields__.values()}
                for k, v in raw.items():
                    if k in known and k != "data_dir":
                        setattr(cfg, k, v)
            except Exception as exc:  # 壊れた設定で起動不能にはしない
                print(f"[System] config.json の読み込みに失敗 ({exc})。既定値で起動します。")
        return cfg

    def save(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    # 便利メソッド ------------------------------------------------------
    def file(self, name: str) -> str:
        os.makedirs(self.data_dir, exist_ok=True)
        return os.path.join(self.data_dir, name)
