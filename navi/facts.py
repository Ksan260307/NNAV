"""事実ストア。

述語項構造から「言い切られた内容」を抜き出して保持する。

    「きみの名前はロックマンだよ」 -> (きみ, 名前, ロックマン)
    「ぼくはケイだ」               -> (ぼく, =, ケイ)
    「ぼくはカレーが好きだ」       -> (ぼく, 好き, カレー)
    「カレーは好きじゃない」       -> (…, 好き, カレー) 極性 -1

問いかけは事実として取り込まない(質問は主張ではない)。
オペレーターの言葉はネット由来より常に強く、矛盾は消さずに両方持って
「前はこう言っていた」と提示できるようにしてある。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ORIGIN_NAVI, ORIGIN_OPERATOR, ORIGIN_WEB
from .parse import FunctionWords, Pred

ORIGIN_WEIGHT = {ORIGIN_OPERATOR: 1.0, ORIGIN_NAVI: 0.5, ORIGIN_WEB: 0.3}
MAX_LEN = 24


@dataclass
class Fact:
    subj: str
    rel: str           # "=" (そのものズバリ) / 属性名 / 述語の原形
    obj: str
    polarity: int = 1
    origin: int = ORIGIN_OPERATOR
    turn: int = 0
    ts: float = field(default_factory=time.time)
    count: float = 1.0
    functional: bool = False   # 値が 1 つに定まる関係か(名前は 1 つ、好物は複数)

    @property
    def key(self) -> Tuple[str, str]:
        return (self.subj, self.rel)

    def confidence(self, now_turn: int) -> float:
        """確信度 = 出自 x 言われた回数 x 新しさ。

        回数を対数にして新しさを強めに効かせているので、何度も言われた事実でも
        オペレーターが言い直せばそちらが勝つ(訂正が通る)。
        """
        import math

        w = ORIGIN_WEIGHT.get(self.origin, 0.3)
        recency = 1.0 / (1.0 + max(0, now_turn - self.turn) / 60.0)
        return (1.0 + math.log1p(self.count)) * w * (0.30 + 0.70 * recency)

    def text(self, deixis=None) -> str:
        subj = deixis.display(self.subj) if deixis is not None else self.subj
        obj = deixis.display(self.obj) if deixis is not None else self.obj
        rel = "" if self.rel == "=" else f"の{self.rel}"
        neg = "ではない" if self.polarity < 0 else ""
        return f"{subj}{rel} = {obj}{neg}"


_FW = FunctionWords.seeded()


def _ok(s: str) -> bool:
    return bool(s) and len(s) <= MAX_LEN and s not in _FW.questions


def set_function_words(fw: FunctionWords) -> None:
    """発見された機能語を反映する(FactStore は文字列しか持たないため)。"""
    global _FW
    _FW = fw


class FactStore:
    def __init__(self) -> None:
        self.facts: Dict[Tuple[str, str], List[Fact]] = {}
        self.conflicts: List[Tuple[Fact, Fact]] = []
        self.learned = 0

    # ------------------------------------------------------------------
    # 抽出
    # ------------------------------------------------------------------
    @staticmethod
    def extract(preds: Sequence[Pred]) -> List[Fact]:
        out: List[Fact] = []
        for p in preds:
            if p.asking or p.kind == "query" or p.is_question:
                continue          # 質問は主張ではない
            topic = p.arg("は") or p.arg("も")
            if p.kind == "copula":
                if not _ok(p.lemma):
                    continue
                if topic and _ok(topic.head):
                    if topic.owner and _ok(topic.owner):
                        # 「きみの名前はロックマンだ」
                        # 「A の B は C」は属性の値を言い切る形なので、
                        # 別の値が来たら矛盾とみなしてよい(名前は 1 つ)。
                        out.append(Fact(topic.owner, topic.head, p.lemma,
                                        p.polarity, functional=True))
                    else:
                        # 「ぼくはケイだ」
                        out.append(Fact(topic.head, "=", p.lemma, p.polarity))
            elif p.kind in ("na", "adj"):
                if not topic or not _ok(topic.head):
                    continue
                subj = topic.owner if (topic.owner and _ok(topic.owner)) else topic.head
                obj_arg = p.arg("が")
                if obj_arg and _ok(obj_arg.head):
                    # 「ぼくはカレーが好きだ」
                    out.append(Fact(subj, p.lemma, obj_arg.head, p.polarity))
                elif _ok(p.lemma):
                    # 「きみは元気だ」
                    out.append(Fact(subj, "=", p.lemma, p.polarity))
        return out

    # ------------------------------------------------------------------
    # 取り込み
    # ------------------------------------------------------------------
    def observe(self, preds: Sequence[Pred], origin: int, turn: int
                ) -> Tuple[List[Fact], List[Tuple[Fact, Fact]]]:
        """新しく覚えた事実と、そこで起きた矛盾を返す。"""
        learned: List[Fact] = []
        clashes: List[Tuple[Fact, Fact]] = []
        for f in self.extract(preds):
            f.origin = origin
            f.turn = turn
            bucket = self.facts.setdefault(f.key, [])
            same = next((g for g in bucket
                         if g.obj == f.obj and g.polarity == f.polarity), None)
            if same is not None:
                same.count += 1.0
                same.turn = turn
                same.ts = time.time()
                if ORIGIN_WEIGHT.get(origin, 0) > ORIGIN_WEIGHT.get(same.origin, 0):
                    same.origin = origin
                continue
            # 同じ (主語, 関係) に別の値がすでにある = 矛盾
            for g in bucket:
                if not (f.functional and g.functional):
                    continue     # 好物や属性は複数あってよい。矛盾ではない。
                if g.polarity == f.polarity and g.obj != f.obj:
                    if origin == ORIGIN_OPERATOR and g.origin == ORIGIN_OPERATOR:
                        clashes.append((g, f))
                        # 言い直された = 訂正。古い値の信用を落とす。
                        # こうしないと「何度も言われた古い事実」が訂正に勝つ。
                        g.count = max(1.0, g.count * 0.4)
            bucket.append(f)
            learned.append(f)
            self.learned += 1
        self.conflicts.extend(clashes)
        del self.conflicts[:-40]
        return learned, clashes

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------
    def lookup(self, subj: str, rel: str, turn: int = 0) -> Optional[Fact]:
        bucket = self.facts.get((subj, rel))
        if not bucket:
            return None
        positive = [f for f in bucket if f.polarity > 0]
        if not positive:
            return None
        return max(positive, key=lambda f: f.confidence(turn))

    def query(self, preds: Sequence[Pred], turn: int = 0) -> Optional[Fact]:
        """問いかけに答えられる事実を探す。

        当てずっぽうで答えるより黙るほうがマシなので、条件は狭く取る。
        主語と関係の両方が特定できたときだけ答える。
        """
        for p in preds:
            if not (p.asking or p.is_question or p.kind == "query"):
                continue
            topic = p.arg("は") or p.arg("も")
            if not (topic and _ok(topic.head)):
                continue
            has_qword = any(a.is_question() for a in p.args)
            wants_value = (p.kind == "query" or p.is_question
                           or p.lemma in _FW.questions)

            # 1) 「きみの名前は？」 主語も関係も明示されている
            if topic.owner and _ok(topic.owner):
                hit = self.lookup(topic.owner, topic.head, turn)
                if hit:
                    return hit

            # 2) 「ぼくは何が好き？」 述語が関係、疑問詞が空欄
            if has_qword and p.lemma and p.kind in ("na", "adj", "verb"):
                hit = self.lookup(topic.head, p.lemma, turn)
                if hit:
                    return hit

            # 3) 「きみは？」「きみは何？」 主語そのものを聞かれている
            if wants_value:
                hit = self.lookup(topic.head, "=", turn)
                if hit:
                    return hit
        return None

    def find_by_rel_obj(self, rel: str, obj: str, turn: int = 0) -> Optional[Fact]:
        best = None
        for bucket in self.facts.values():
            for f in bucket:
                if f.rel == rel and f.obj == obj and f.polarity > 0:
                    if best is None or f.confidence(turn) > best.confidence(turn):
                        best = f
        return best

    def find_by_rel(self, rel: str, turn: int = 0) -> Optional[Fact]:
        best = None
        for bucket in self.facts.values():
            for f in bucket:
                if f.rel == rel and f.polarity > 0:
                    if best is None or f.confidence(turn) > best.confidence(turn):
                        best = f
        return best

    def about(self, subj: str, turn: int = 0) -> List[Fact]:
        out = [f for (s, _), bucket in self.facts.items() if s == subj
               for f in bucket]
        out.sort(key=lambda f: -f.confidence(turn))
        return out

    def all_facts(self) -> List[Fact]:
        return [f for bucket in self.facts.values() for f in bucket]

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        return {"keys": len(self.facts), "facts": len(self.all_facts()),
                "conflicts": len(self.conflicts), "learned": self.learned}

    def to_dict(self) -> dict:
        return {
            "facts": [[f.subj, f.rel, f.obj, f.polarity, f.origin, f.turn,
                       f.ts, f.count, f.functional] for f in self.all_facts()],
            "learned": self.learned,
        }

    def load(self, d: dict) -> None:
        self.facts = {}
        self.conflicts = []
        for row in d.get("facts") or []:
            f = Fact(row[0], row[1], row[2], int(row[3]), int(row[4]),
                     int(row[5]), float(row[6]), float(row[7]),
                     bool(row[8]) if len(row) > 8 else False)
            self.facts.setdefault(f.key, []).append(f)
        self.learned = int(d.get("learned", 0))
