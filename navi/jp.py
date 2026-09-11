"""日本語の形態素解析と、発話を「人間らしく」するための文法合法性ルール。

単純なマルコフ連鎖が不自然に聞こえる最大の原因は、日本語として成立しない
品詞の並び(助詞の連打、文末が助詞で終わる、連体詞の後に動詞が来る 等)を
平気で生成してしまうことにある。ここでは品詞遷移の合法性だけを制約として
与え、「どの単語を選ぶか」は一切ハードコードしない。
(= 語彙も意味も人格もオペレーターから学ぶ、という原則は崩さない)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, NamedTuple, Optional

try:
    from janome.tokenizer import Tokenizer as _JanomeTokenizer
except ImportError:  # pragma: no cover - 環境依存
    _JanomeTokenizer = None


class Tok(NamedTuple):
    surface: str        # 表層形
    pos: str            # 品詞(大分類)
    pos1: str           # 品詞細分類1
    base: str           # 原形
    conj: str = ""      # 活用形(命令形の検出・活用の記憶に使う)


# --- 品詞カテゴリ ---------------------------------------------------------
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}
# 文末に置いても日本語として自然に終われる品詞
ENDABLE_POS = {"名詞", "動詞", "形容詞", "助動詞", "感動詞", "副詞", "フィラー"}
# 文頭に置けない品詞
NON_INITIAL_POS = {"助詞", "助動詞"}
# 文末に置くと明らかに尻切れになる品詞
NON_FINAL_POS = {"助詞", "接続詞", "連体詞", "接頭詞"}

TERMINATORS = "。．.！!？?"
SOFT_PUNCT = "、，,・…ー~〜"
# 読点だけは「間」として学習する価値があるので語彙に残す
KEEP_PUNCT = "、，,"
# 括弧は片方だけ残ると文が壊れるので語彙に入れない
BRACKETS = "「」『』（）()【】《》〈〉[]{}\"'“”‘’"

_SKIP_SURFACES = {"", " ", "\u3000", "\t", "\n", "\r"}
_URL_RE = re.compile(r"https?://\S+")
_BRACKET_RE = re.compile(r"[（(][^）)]{0,60}[）)]")
_REF_RE = re.compile(r"\[\d+\]")
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*")


class JPTokenizer:
    """Janome ラッパー。未導入環境では文字単位にフォールバックする。"""

    def __init__(self) -> None:
        self._t = _JanomeTokenizer() if _JanomeTokenizer else None
        self.available = self._t is not None
        self._cache: dict[str, List[Tok]] = {}
        self._cache_limit = 4096

    def tokenize(self, text: str) -> List[Tok]:
        text = normalize(text)
        if not text:
            return []
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        toks = self._tokenize_uncached(text)
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[text] = toks
        return toks

    def _tokenize_uncached(self, text: str) -> List[Tok]:
        out: List[Tok] = []
        if self._t is None:
            for ch in text:
                if ch in _SKIP_SURFACES:
                    continue
                pos = "記号" if _is_punct(ch) else "名詞"
                out.append(Tok(ch, pos, "*", ch))
            return out
        for t in self._t.tokenize(text):
            surface = t.surface
            if surface in _SKIP_SURFACES:
                continue
            parts = t.part_of_speech.split(",")
            pos = parts[0] if parts else "その他"
            pos1 = parts[1] if len(parts) > 1 else "*"
            base = t.base_form if t.base_form and t.base_form != "*" else surface
            conj = getattr(t, "infl_form", "") or ""
            out.append(Tok(surface, pos, pos1, base, conj))
        return out


def _is_punct(ch: str) -> bool:
    return ch in TERMINATORS or ch in SOFT_PUNCT or unicodedata.category(ch).startswith("P")


def normalize(text: str) -> str:
    """全角英数の統一と空白の正規化。絵文字や顔文字は壊さない。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    # NFKC は全角の ？ ！ を半角に潰してしまう。日本語の文としては
    # 全角のほうが自然なので戻す(終止記号の判定は両方を受け付ける)。
    text = text.replace("?", "？").replace("!", "！")
    return re.sub(r"[ \t]+", " ", text).strip()


def is_terminator(surface: str) -> bool:
    return bool(surface) and all(c in TERMINATORS for c in surface)


def is_soft_punct(surface: str) -> bool:
    return bool(surface) and all(c in SOFT_PUNCT for c in surface)


def is_comma(surface: str) -> bool:
    return bool(surface) and all(c in KEEP_PUNCT for c in surface)


def is_bracket(surface: str) -> bool:
    return bool(surface) and all(c in BRACKETS for c in surface)


def is_noise(surface: str) -> bool:
    """語彙に入れる価値のない記号(中黒・三点リーダ・括弧 など)。"""
    if is_bracket(surface):
        return True
    return is_soft_punct(surface) and not is_comma(surface)


def is_content(tok: Tok) -> bool:
    if tok.pos not in CONTENT_POS:
        return False
    if tok.pos == "名詞" and tok.pos1 in {"代名詞", "非自立", "接尾", "数"}:
        return False
    if tok.pos == "動詞" and tok.pos1 == "非自立":
        return False
    if len(tok.surface) == 1 and _is_punct(tok.surface):
        return False
    return True


# ---------------------------------------------------------------------------
# 品詞遷移の合法性
# ---------------------------------------------------------------------------
def can_start(pos: str, surface: str) -> bool:
    if pos in NON_INITIAL_POS:
        return False
    if pos == "記号":
        return False
    if pos == "名詞" and surface in {"の", "こと"}:
        return False
    return True


def can_end(pos: str, surface: str) -> bool:
    if is_terminator(surface):
        return True
    if is_soft_punct(surface):
        return False
    if pos in NON_FINAL_POS:
        return False
    if pos == "記号":
        return False
    return pos in ENDABLE_POS


def can_follow(prev: Optional[Tok], nxt: Tok, consecutive_particles: int = 0) -> bool:
    """prev の直後に nxt を置いてよいか(日本語として破綻しないか)。"""
    if prev is None:
        return can_start(nxt.pos, nxt.surface)

    # 文終止記号はここでは扱わない(生成側が文末処理で付与する)
    if is_terminator(nxt.surface):
        return can_end(prev.pos, prev.surface)

    # 読点の直後に読点、文頭直後の読点は禁止
    if is_soft_punct(nxt.surface):
        return not (is_soft_punct(prev.surface) or prev.pos in NON_FINAL_POS)
    if is_soft_punct(prev.surface):
        return can_start(nxt.pos, nxt.surface) or nxt.pos in {"助詞", "助動詞"} is False

    # 連体詞(この/その/大きな…)の後は名詞句しか来られない
    if prev.pos == "連体詞":
        return nxt.pos in {"名詞", "接頭詞", "連体詞", "形容詞"}

    # 接頭詞(お/ご/第…)の後は名詞か動詞
    if prev.pos == "接頭詞":
        return nxt.pos in {"名詞", "動詞", "接頭詞"}

    # 助詞の連打は最大2つまで(「からは」「までに」は許容、3連は破綻)
    if prev.pos == "助詞" and nxt.pos == "助詞":
        return consecutive_particles < 2

    # 接続助詞や格助詞の直後に接続詞が来るのは不自然
    if prev.pos == "助詞" and nxt.pos == "接続詞":
        return False

    # 感動詞の後に助詞・助動詞は付かない
    if prev.pos == "感動詞" and nxt.pos in {"助詞", "助動詞"}:
        return False

    # 接続詞の直後に助詞・助動詞は付かない
    if prev.pos == "接続詞" and nxt.pos in {"助詞", "助動詞"}:
        return False

    return True


def trim_to_valid_end(toks: List[Tok]) -> List[Tok]:
    """末尾から、文末に置けない品詞を削って尻切れを防ぐ。"""
    out = list(toks)
    while out and not can_end(out[-1].pos, out[-1].surface):
        out.pop()
    return out


def split_sentences(text: str, max_len: int = 160) -> List[str]:
    """文章を文単位に分割する(Web学習用)。"""
    text = _URL_RE.sub(" ", text)
    text = _REF_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    out: List[str] = []
    for chunk in _SENT_SPLIT_RE.split(text):
        s = chunk.strip()
        if not s:
            continue
        if len(s) > max_len:
            s = s[:max_len]
        out.append(s)
    return out


def join_surfaces(toks: Iterable[Tok]) -> str:
    return "".join(t.surface for t in toks)
