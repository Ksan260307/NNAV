"""ネットナビ起動スクリプト。

    python navi_main.py                # 対話を始める
    python navi_main.py --no-web       # ネット学習を切って起動
    python navi_main.py --teach log.txt  # テキストを一気に教えてから対話
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from navi.cli import run, setup_console  # noqa: E402
from navi.config import Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="NetNavi Core Engine")
    ap.add_argument("--data-dir", default="navi_data", help="電脳メモリの保存先")
    ap.add_argument("--no-web", action="store_true", help="インターネット学習を無効化")
    ap.add_argument("--no-daemon", action="store_true", help="常駐スレッドを無効化")
    ap.add_argument("--teach", metavar="FILE", help="起動時にテキストを一括学習")
    ap.add_argument("--seed", type=int, default=0, help="乱数シード(0 で非決定的)")
    args = ap.parse_args()

    setup_console()
    cfg = Config.load(args.data_dir)
    if args.no_web:
        cfg.web_enabled = False
    if args.no_daemon:
        cfg.daemon_enabled = False
    if args.seed:
        cfg.seed = args.seed
    cfg.save()

    if args.teach:
        from navi.navi import NetNavi

        navi = NetNavi(data_dir=args.data_dir, cfg=cfg)
        n = navi.teach_file(args.teach)
        print(f"[System] {n} 行を学習しました。語彙 {navi.brain.learned_vocab} 語")
        navi.shutdown()

    run(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
