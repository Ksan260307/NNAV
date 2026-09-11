"""NetNavi Core Engine.

ロックマンエグゼのネットナビを「ゼロから育つ電脳生命体」として実装するための
自律成長型・対話コアエンジン。

設計原則:
  1. 事前学習済みの巨大言語モデルを一切使わない（知能を借りない）。
  2. 言葉・文法・人格はすべてオペレーターとの対話から自己組織化する。
  3. インターネットは「語彙と知識の栄養源」であり、「声（言い回し）」は
     オペレーターからしか学ばない。
"""

__version__ = "0.3.0"
__all__ = ["NetNavi"]


def __getattr__(name):
    if name == "NetNavi":
        from .navi import NetNavi

        return NetNavi
    raise AttributeError(name)
