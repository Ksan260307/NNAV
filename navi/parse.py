"""述語項構造・極性・モダリティの抽出。

日本語は格助詞が意味役割を表層に出しているので、係り受け解析器なしでも
「誰が・何を・どうした」がかなりの精度で取れる。Janome の品詞細分類だけを
使い、語彙知識は一切持ち込まない(唯一の例外は疑問詞リスト。理由は下記)。

    「ぼくはカレーが好きだ」   -> Pred(好き, na, [ (ぼく,は), (カレー,が) ], +1)
    「きみの名前はロックマンだよ」-> Pred(ロックマン, copula, [ (名前<-きみ,は) ], +1)
    「カレーは食べない」        -> Pred(食べる, verb, [ (カレー,は) ], -1)

ここで得た構造が、選択選好(frames.py)・事実(facts.py)・
意味ベクトル(embed.py)・理解度評価(comprehension.py)すべての入力になる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .jp import Tok, is_terminator

# --- 格 -------------------------------------------------------------------
# 格助詞(が/を/に/…)に加え、係助詞「は」「も」も役割標識として扱う。
# 「は」は真の格を隠すが、主題として極めて情報量が高いのでそのまま採る。
CASES = ("が", "を", "に", "へ", "と", "で", "から", "より", "まで", "は", "も")
CASE_POS1 = {"格助詞", "係助詞", "副助詞", "副助詞／並立助詞／終助詞"}

# --- 名詞句を構成しうる品詞 ------------------------------------------------
NOUN_POS = {"名詞", "接頭詞"}
NOUN_SKIP_POS1 = {"接尾", "非自立"}   # 単独では句の主辞になれない

# --- 極性・モダリティ ------------------------------------------------------
# 助動詞は閉じた機能語の集合であり、語彙知識ではなく文法の一部として扱う。
NEGATIVE = {"ない", "ぬ", "ん", "まい"}
MODALITY = {
    "たい": "desire", "たがる": "desire",
    "た": "past",
    "う": "guess", "よう": "guess", "だろう": "guess", "まい": "guess",
    "らしい": "hearsay", "そうだ": "hearsay", "ようだ": "hearsay",
    "れる": "passive", "られる": "passive",
    "せる": "causative", "させる": "causative",
    "べき": "should",
    "ます": "polite", "です": "polite",
}

# --- 疑問詞 ---------------------------------------------------------------
# 事実の照会にはどうしても「何を聞かれているか」の判定が要る。
# ここだけは閉じた語彙を持ち込んでいる (Config.use_question_words で無効化可)。
QUESTION_WORDS = {
    "何", "なに", "なん", "誰", "だれ", "どこ", "いつ", "どれ", "どの",
    "どちら", "どっち", "いくつ", "いくら", "なぜ", "どうして", "どう",
}


@dataclass
class Arg:
    """述語の項(名詞句 + 格)。"""
    head: str            # 名詞句の主辞(複合名詞は連結済み)
    owner: str = ""      # 「AのB」の A
    case: str = ""       # が / を / に / … / は / も  (空 = 格標識なし)

    def is_question(self) -> bool:
        return self.head in QUESTION_WORDS


@dataclass
class Pred:
    """述語とその項。"""
    lemma: str                        # 原形
    surface: str
    kind: str                         # verb / adj / na / copula
    args: List[Arg] = field(default_factory=list)
    polarity: int = 1                 # +1 肯定 / -1 否定
    modality: Tuple[str, ...] = ()
    is_question: bool = False         # 述語位置が疑問詞、または項に疑問詞
    asking: bool = False              # 発話そのものが問いかけ(？ や 疑問詞を含む)

    def arg(self, case: str) -> Optional[Arg]:
        for a in self.args:
            if a.case == case:
                return a
        return None

    def triples(self) -> List[Tuple[str, str, str]]:
        """(述語, 格, 名詞) の三つ組。格フレームの学習単位。"""
        return [(self.lemma, a.case, a.head) for a in self.args if a.case and a.head]


# ---------------------------------------------------------------------------
def _is_noun_like(t: Tok) -> bool:
    if t.pos not in NOUN_POS:
        return False
    if t.pos == "名詞" and t.pos1 in {"接尾"}:
        return True          # 接尾辞は直前の名詞に連結する
    return True


def _is_aux(t: Tok) -> bool:
    if t.pos == "助動詞":
        return True
    if t.pos in {"動詞", "形容詞"} and t.pos1 == "非自立":
        return True
    if t.pos == "助詞" and t.pos1 == "接続助詞":
        return True
    return False


def _consume_aux(toks: Sequence[Tok], i: int, p: Pred) -> int:
    """述語の後ろに続く助動詞列から極性とモダリティを読み取る。"""
    mods: List[str] = []
    while i < len(toks):
        t = toks[i]
        if not _is_aux(t):
            # 「好き / じゃ / ない」の「じゃ」のように、助動詞に挟まれた助詞は
            # 述語の一部とみなして読み飛ばす(否定の取りこぼしを防ぐ)。
            if (t.pos == "助詞" and i + 1 < len(toks)
                    and toks[i + 1].pos == "助動詞"):
                i += 1
                continue
            break
        base = t.base or t.surface
        if base in NEGATIVE or t.surface in NEGATIVE:
            p.polarity = -p.polarity
        m = MODALITY.get(base)
        if m and m not in mods:
            mods.append(m)
        i += 1
    p.modality = tuple(mods)
    return i


def parse(toks: Sequence[Tok]) -> List[Pred]:
    """トークン列から述語項構造を抽出する。

    複文は「次に出てきた述語にそれまでの項を束ねる」という近似で処理する。
    取りこぼしも誤結合もあるが、統計を取る用途では十分に働く。
    """
    preds: List[Pred] = []
    pending: List[Arg] = []
    np_buf: List[str] = []
    np_last_pos1 = ""
    np_independent = False   # 形式名詞(こと/もの/ん/ほう)だけの句を弾く
    owner = ""
    i = 0
    n = len(toks)
    asking = any(t.surface in "？?" for t in toks) or has_question_word(toks)

    def resolve_np() -> str:
        """名詞句の主辞を決める。形式名詞だけなら所有者に降りる。"""
        nonlocal owner
        head = "".join(np_buf)
        if np_buf and not np_independent:
            head, owner = owner, ""
        return head

    def flush_np_as_arg(head: str, case: str = "") -> None:
        nonlocal np_buf, owner, np_last_pos1, np_independent
        if head:
            pending.append(Arg(head, owner, case))
        np_buf, owner, np_last_pos1, np_independent = [], "", "", False

    def flush_query() -> None:
        """述語が省略された問いかけ(「きみの名前は？」)の骨格を残す。"""
        nonlocal pending
        if asking and pending:
            q = Pred("", "", "query", list(pending))
            q.is_question = True
            q.asking = True
            preds.append(q)
        pending = []

    def make_pred(lemma: str, surface: str, kind: str) -> Pred:
        nonlocal pending
        p = Pred(lemma, surface, kind, list(pending))
        p.is_question = any(a.is_question() for a in p.args)
        p.asking = asking
        pending = []
        return p

    while i < n:
        t = toks[i]

        # --- 名詞句の蓄積 ------------------------------------------------
        if _is_noun_like(t):
            np_buf.append(t.surface)
            np_last_pos1 = t.pos1
            if t.pos1 not in NOUN_SKIP_POS1:
                np_independent = True
            i += 1
            continue

        head = resolve_np()

        # --- 「A の B」: A を所有者として退避 -----------------------------
        if head and t.pos == "助詞" and t.pos1 == "連体化":
            owner = head if not owner else f"{owner}の{head}"
            np_buf, np_last_pos1, np_independent = [], "", False
            i += 1
            continue

        # --- 名詞句 + 格助詞 ----------------------------------------------
        if head and t.pos == "助詞" and t.surface in CASES and t.pos1 in CASE_POS1:
            flush_np_as_arg(head, t.surface)
            i += 1
            continue

        # --- 名詞句 + 助動詞 → コピュラ述語 --------------------------------
        # 「だ/です」だけに限ると「好きじゃない」「元気でした」を取り落とす。
        # 形容動詞語幹のあとは助詞始まりの否定形(じゃない/ではない)も来る。
        if head and (t.pos == "助動詞"
                     or (np_last_pos1 == "形容動詞語幹" and t.pos == "助詞"
                         and t.surface not in CASES)):
            kind = "na" if np_last_pos1 == "形容動詞語幹" else "copula"
            np_buf, np_last_pos1, np_independent = [], "", False
            saved_owner, owner = owner, ""
            p = make_pred(head, head, kind)
            if saved_owner:
                p.args.append(Arg(head, saved_owner, ""))
            if head in QUESTION_WORDS:
                p.is_question = True
            i = _consume_aux(toks, i, p)
            preds.append(p)
            continue

        # --- 形容動詞語幹が単独で文を締める(「カレーが好き」) ---------------
        if head and np_last_pos1 == "形容動詞語幹" and (
                is_terminator(t.surface) or t.pos == "助詞" and t.pos1 == "終助詞"):
            np_buf, np_last_pos1, owner, np_independent = [], "", "", False
            p = make_pred(head, head, "na")
            preds.append(p)
            i += 1
            continue

        # --- 宙に浮いた名詞句は格なしの項として保持 -------------------------
        if head or np_buf:
            flush_np_as_arg(head, "")

        # --- 動詞・形容詞 --------------------------------------------------
        if t.pos == "動詞" and t.pos1 == "自立":
            p = make_pred(t.base or t.surface, t.surface, "verb")
            i = _consume_aux(toks, i + 1, p)
            preds.append(p)
            continue
        if t.pos == "形容詞" and t.pos1 == "自立":
            p = make_pred(t.base or t.surface, t.surface, "adj")
            i = _consume_aux(toks, i + 1, p)
            preds.append(p)
            continue

        # --- 文の区切り。問いかけなら骨格を残してから持ち越しを捨てる -------
        if is_terminator(t.surface):
            flush_query()
        i += 1

    # 「きみの名前はロックマン」のように だ が無いまま終わる形
    if np_buf:
        head = resolve_np()
        saved_owner, owner = owner, ""
        np_buf = []
        if head and any(a.case == "は" for a in pending):
            p = make_pred(head, head, "copula")
            if saved_owner:
                p.args.append(Arg(head, saved_owner, ""))
            if head in QUESTION_WORDS:
                p.is_question = True
            preds.append(p)

    flush_query()
    return preds


def has_question_word(toks: Sequence[Tok]) -> bool:
    return any(t.surface in QUESTION_WORDS for t in toks)
