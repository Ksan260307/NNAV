"""話者役割の正規化。

「ぼく」「きみ」は、誰が言ったかで指す相手が入れ替わる。表層のまま記憶すると
ナビが自分をオペレーターだと言い出す(実際にそうなっていた)。そこで人称は
表層ではなく **役割** に畳んで持つ。

    オペレーターが「ぼく」  -> @OP      ナビが「ぼく」  -> @NAVI
    オペレーターが「きみ」  -> @NAVI    ナビが「きみ」  -> @OP

さらに (@NAVI, 名前, ロックマン) を教わったら「ロックマン」も @NAVI の別名に
なるので、呼びかけ「ロックマン、おはよう」も同じ実体に落ちる。

人称は疑問詞と同じく閉じた機能語の集合として扱う
(Config.use_person_words で無効化できる)。
"""

from __future__ import annotations

from typing import Dict, Optional

from .config import ORIGIN_DREAM, ORIGIN_NAVI, ORIGIN_OPERATOR
from .parse import SEED_FIRST, SEED_SECOND, FunctionWords

ROLE_OP = "@OP"
ROLE_NAVI = "@NAVI"
ROLES = (ROLE_OP, ROLE_NAVI)

FIRST_PERSON = SEED_FIRST      # 後方互換
SECOND_PERSON = SEED_SECOND

# ナビ側として扱う話者
_NAVI_SPEAKERS = {ORIGIN_NAVI, ORIGIN_DREAM}


def is_role(s: str) -> bool:
    return s in ROLES


class Deixis:
    def __init__(self, enabled: bool = True,
                 fw: Optional[FunctionWords] = None) -> None:
        self.enabled = enabled
        self.fw = fw if fw is not None else FunctionWords.seeded()
        self.aliases: Dict[str, str] = {}     # 固有名 -> 役割
        self.names: Dict[str, str] = {}       # 役割 -> 名前
        # オペレーターが実際に使った人称。ナビが喋るときはこれを借りる。
        self.forms: Dict[str, Dict[str, float]] = {"first": {}, "second": {}}

    # ------------------------------------------------------------------
    def normalize(self, surface: str, speaker: int) -> str:
        """発話者から見た人称を役割に畳む。"""
        if not self.enabled or not surface:
            return surface
        role = self.aliases.get(surface)
        if role:
            return role
        if speaker == ORIGIN_OPERATOR:
            if surface in self.fw.first:
                self.forms["first"][surface] = self.forms["first"].get(surface, 0.0) + 1
                return ROLE_OP
            if surface in self.fw.second:
                self.forms["second"][surface] = self.forms["second"].get(surface, 0.0) + 1
                return ROLE_NAVI
        elif speaker in _NAVI_SPEAKERS:
            if surface in self.fw.first:
                return ROLE_NAVI
            if surface in self.fw.second:
                return ROLE_OP
        # Web 由来の一人称は誰を指すか分からないので畳まない
        return surface

    def apply(self, preds, speaker: int) -> None:
        """述語項構造の中の人称をまとめて役割に置き換える。"""
        if not self.enabled:
            return
        for p in preds:
            for a in p.args:
                a.head = self.normalize(a.head, speaker)
                if a.owner:
                    a.owner = self.normalize(a.owner, speaker)
            # 述語(値)の位置は畳まない。「きみの名前はロックマン」の
            # ロックマンまで @NAVI にしてしまうと、名前そのものが消える。

    # ------------------------------------------------------------------
    def learn_names(self, facts) -> None:
        """(役割, 名前, X) を見つけたら X を別名として登録する。"""
        if not self.enabled:
            return
        for role in ROLES:
            hit = facts.lookup(role, "名前")
            if hit and hit.obj and not is_role(hit.obj):
                self.names[role] = hit.obj
                self.aliases[hit.obj] = role

    def surface_for(self, role: str) -> str:
        """ナビが口に出すときの表層。自分は一人称、相手は二人称。"""
        group = "first" if role == ROLE_NAVI else "second"
        table = self.forms.get(group) or {}
        if table:
            return max(table.items(), key=lambda kv: kv[1])[0]
        return "ぼく" if role == ROLE_NAVI else "きみ"

    def display(self, s: str) -> str:
        """記憶の中の役割を、オペレーターから見た呼び方に戻す。

        名前で置き換えると「ケイの名前 = ケイ」のような表示になるので、
        あくまで関係(あなた/きみ)で出す。
        """
        if s == ROLE_OP:
            return "あなた"
        if s == ROLE_NAVI:
            return "きみ"
        return s

    def name_of(self, role: str) -> Optional[str]:
        return self.names.get(role)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"aliases": dict(self.aliases), "names": dict(self.names),
                "forms": {k: dict(v) for k, v in self.forms.items()}}

    def load(self, d: dict) -> None:
        self.aliases = dict(d.get("aliases") or {})
        self.names = dict(d.get("names") or {})
        f = d.get("forms") or {}
        self.forms = {"first": dict(f.get("first") or {}),
                      "second": dict(f.get("second") or {})}
