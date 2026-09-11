"""機能語の再発見テスト。

疑問詞・人称・指示詞の組み込みリスト(種)を丸ごと外して起動し、
ナビが分布から何ターンで自力で見つけ直すかを測る。

    python tools/rediscover.py --turns 3000

「消せます」と主張するより、消して測ったほうが誠実なので。
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navi.cli import setup_console  # noqa: E402
from navi.config import Config  # noqa: E402
from navi.navi import NetNavi  # noqa: E402
from navi.parse import (SEED_DEICTIC, SEED_FIRST, SEED_QUESTION,  # noqa: E402
                        SEED_SECOND)

TRUTH = {"疑問詞": SEED_QUESTION, "一人称": SEED_FIRST,
         "二人称": SEED_SECOND, "指示詞": SEED_DEICTIC}
MARKS = (200, 400, 800, 1200, 2000, 3000, 5000)


def reachable(navi, truth):
    """一度でも耳にした語だけを分母にする(聞いていない語は見つけようがない)。"""
    return {w for w in truth if w in navi.brain.word2id}


def report(navi, turn: int):
    fw = navi.lexicon.discovered(navi.frames, navi.embed, navi.brain)
    found = {"疑問詞": fw.questions, "一人称": fw.first,
             "二人称": fw.second, "指示詞": fw.deictics}
    parts = []
    for name, truth in TRUTH.items():
        got, can = found[name], reachable(navi, truth)
        hit, fp = got & can, got - truth
        parts.append(f"{name} {len(hit)}/{len(can)}"
                     + (f"(誤{len(fp)})" if fp else ""))
    print(f"  turn {turn:>5}  " + "  ".join(parts))
    return found


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="navi_data_rediscover")
    ap.add_argument("--corpus", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corpus_sample.txt"))
    ap.add_argument("--with-seeds", action="store_true",
                    help="種を与えたまま走らせる(比較用)")
    args = ap.parse_args()

    if os.path.isdir(args.data_dir):
        shutil.rmtree(args.data_dir)
    cfg = Config.load(args.data_dir)
    cfg.web_enabled = False
    cfg.daemon_enabled = False
    cfg.seed = args.seed
    cfg.bootstrap_function_words = args.with_seeds
    cfg.save()

    corpus = [l.strip() for l in open(args.corpus, encoding="utf-8")
              if l.strip() and not l.startswith("#")]
    rng = random.Random(args.seed)
    navi = NetNavi(data_dir=args.data_dir, cfg=cfg)

    print(f"\n=== 機能語の再発見 ({'種あり' if args.with_seeds else '種なし'}) ===")
    print(f"    {args.turns} ターン / 語彙源 {len(corpus)} 発話\n")
    i = 0
    found = {}
    for turn in range(1, args.turns + 1):
        if rng.random() < 0.25:
            line = corpus[rng.randrange(len(corpus))]
        else:
            line = corpus[i % len(corpus)]
            i += 1
        navi.talk(line)
        if turn % 200 == 0:
            navi._maybe_build_embedding(force=True)
        if turn in MARKS or turn == args.turns:
            found = report(navi, turn)

    print("")
    print("--- 見つけたもの(耳にした語だけを分母にする) ---")
    for name, truth in TRUTH.items():
        got = found.get(name, set())
        can = reachable(navi, truth)
        print(f"  {name}: {len(got & can)}/{len(can)}  {sorted(got & can)}"
              + (f"   誤検出 {sorted(got - truth)}" if got - truth else ""))
        missed = sorted(w for w in can if w not in got)
        if missed:
            print(f"    見つからず: {missed}")
    navi.shutdown()


if __name__ == "__main__":
    main()
