"""活用の記憶。

活用エンジンは持たない。「実際に聞いた活用形だけを再現できる」という方針で、
述語の原形と (極性, モダリティ) の組に対して、口にされた表層の並びを数える。

    食べる + 否定        <- ("食べ", "ない")
    買う   + 過去        <- ("買っ", "て", "き", "た") / ("買っ", "た", "よ")
    コピュラ + 否定      <- ("じゃ", "ない")          ※ 名詞は呼び出し側が付ける

教わっていない活用は作れない。これは制約ではなく、この作品の原則そのもの
(オペレーターが口にしたことのない喋り方はしない)。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from .parse import Pred

Form = Tuple[int, Tuple[str, ...]]        # (極性, モダリティ)
Span = Tuple[str, ...]

# 名詞に付く語尾は述語ごとではなく汎用なので、種類ごとにまとめて覚える
GENERIC = {"na": "@na", "copula": "@copula"}


class Realizer:
    def __init__(self) -> None:
        self.forms: Dict[str, Dict[Form, Dict[Span, float]]] = defaultdict(
            lambda: defaultdict(dict))
        # 接続助詞で終わる形(「食べて」「行くけど」)。節をつなぐときに使う。
        self.conn: Dict[str, Dict[Form, Dict[Span, float]]] = defaultdict(
            lambda: defaultdict(dict))
        self.observed = 0

    # ------------------------------------------------------------------
    @staticmethod
    def key_of(pred: Pred) -> Tuple[str, Span]:
        """(記憶キー, 覚える表層列) を決める。

        名詞述語(コピュラ・形容動詞)は頭が任意の名詞なので、語尾だけを
        汎用キーに貯める。動詞・形容詞は語幹ごと活用するので丸ごと貯める。
        """
        if pred.kind in GENERIC:
            return GENERIC[pred.kind], tuple(pred.span[1:])
        return pred.lemma, tuple(pred.span)

    def observe(self, preds: Sequence[Pred], w: float = 1.0) -> int:
        n = 0
        for p in preds:
            if not p.lemma or p.kind == "query" or not p.span:
                continue
            key, span = self.key_of(p)
            if not key:
                continue
            if p.kind in GENERIC and not span:
                span = ()          # 「ぼくはケイ」のように語尾が無い形も 1 つの形
            store = self.conn if p.connective else self.forms
            table = store[key][(p.polarity, tuple(p.modality))]
            table[span] = table.get(span, 0.0) + w
            n += 1
        self.observed += n
        return n

    # ------------------------------------------------------------------
    def _table(self, key: str, form: Form,
               connective: bool = False) -> Optional[Dict[Span, float]]:
        by_form = (self.conn if connective else self.forms).get(key)
        if not by_form:
            return None
        if form in by_form and by_form[form]:
            return by_form[form]
        # 同じ極性で、モダリティ無しへ後退
        plain = (form[0], ())
        if plain in by_form and by_form[plain]:
            return by_form[plain]
        # 同じ極性のどれか
        merged: Dict[Span, float] = {}
        for (pol, _), table in by_form.items():
            if pol == form[0]:
                for span, v in table.items():
                    merged[span] = merged.get(span, 0.0) + v
        return merged or None

    def realize(self, lemma: str, kind: str, polarity: int,
                modality: Sequence[str], rng,
                connective: bool = False) -> Optional[List[str]]:
        """述語を表層に戻す。聞いたことのない形は None。

        connective=True なら「〜して」「〜だけど」のように次の節へ続く形を返す。
        """
        key = GENERIC.get(kind, lemma)
        table = self._table(key, (polarity, tuple(modality)), connective)
        if not table:
            return None
        spans = list(table.keys())
        span = rng.choices(spans, weights=[table[s] for s in spans], k=1)[0]
        if kind in GENERIC:
            return [lemma] + list(span)
        return list(span)

    def sample_form(self, lemma: str, kind: str, rng) -> Optional[Form]:
        """その述語で実際に使われた (極性, モダリティ) を 1 つ引く。"""
        key = GENERIC.get(kind, lemma)
        by_form = self.forms.get(key)
        if not by_form:
            return None
        forms = [f for f, t in by_form.items() if t]
        if not forms:
            return None
        weights = [sum(by_form[f].values()) for f in forms]
        return rng.choices(forms, weights=weights, k=1)[0]

    def coverage(self, lemma: str, kind: str) -> int:
        key = GENERIC.get(kind, lemma)
        by_form = self.forms.get(key)
        return sum(len(t) for t in by_form.values()) if by_form else 0

    def can_connect(self, lemma: str, kind: str) -> bool:
        """次の節に続く形を聞いたことがあるか。"""
        key = GENERIC.get(kind, lemma)
        by_form = self.conn.get(key)
        return bool(by_form and any(by_form.values()))

    def knows(self, lemma: str, kind: str) -> bool:
        return self.coverage(lemma, kind) > 0

    # ------------------------------------------------------------------
    def decay(self, rate: float, prune: float) -> None:
        for store in (self.forms, self.conn):
            for key, by_form in list(store.items()):
                for form, table in list(by_form.items()):
                    for span in list(table.keys()):
                        v = table[span] * rate
                        if v < prune:
                            del table[span]
                        else:
                            table[span] = v
                    if not table:
                        del by_form[form]
                if not by_form:
                    del store[key]

    def stats(self) -> Dict[str, float]:
        return {"preds": len(self.forms),
                "forms": sum(len(t) for f in self.forms.values() for t in f.values()),
                "conn": sum(len(t) for f in self.conn.values() for t in f.values()),
                "observed": self.observed}

    def to_dict(self) -> dict:
        return {
            "forms": [[key, form[0], list(form[1]), list(span), v]
                      for key, by_form in self.forms.items()
                      for form, table in by_form.items()
                      for span, v in table.items()],
            "conn": [[key, form[0], list(form[1]), list(span), v]
                     for key, by_form in self.conn.items()
                     for form, table in by_form.items()
                     for span, v in table.items()],
            "observed": self.observed,
        }

    def load(self, d: dict) -> None:
        self.forms = defaultdict(lambda: defaultdict(dict))
        self.conn = defaultdict(lambda: defaultdict(dict))
        for key, pol, mods, span, v in (d.get("forms") or []):
            self.forms[key][(int(pol), tuple(mods))][tuple(span)] = float(v)
        for key, pol, mods, span, v in (d.get("conn") or []):
            self.conn[key][(int(pol), tuple(mods))][tuple(span)] = float(v)
        self.observed = int(d.get("observed", 0))
