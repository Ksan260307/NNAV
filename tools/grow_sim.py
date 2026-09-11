"""成長シミュレータ。

ゼロの状態から N ターン会話させて、ナビがどう育つかを観測する。
ネット学習と常駐スレッドは切って、純粋に対話だけの成長を見る。

    python tools/grow_sim.py --turns 900
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navi.cli import setup_console  # noqa: E402
from navi.config import Config  # noqa: E402
from navi.navi import NetNavi  # noqa: E402

CHECKPOINTS = (1, 3, 6, 12, 25, 50, 100, 200, 350, 550, 800, 1200, 1800, 2500)


def load_corpus(path: str):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="navi_data_sim")
    ap.add_argument("--corpus", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "corpus_sample.txt"))
    ap.add_argument("--keep", action="store_true", help="既存の電脳メモリを消さない")
    ap.add_argument("--sleep-every", type=int, default=300,
                    help="何ターンごとに深い睡眠(重い学習)を挟むか")
    args = ap.parse_args()

    if not args.keep and os.path.isdir(args.data_dir):
        shutil.rmtree(args.data_dir)

    cfg = Config.load(args.data_dir)
    cfg.web_enabled = False
    cfg.daemon_enabled = False
    cfg.seed = args.seed
    cfg.save()

    corpus = load_corpus(args.corpus)
    rng = random.Random(args.seed)
    navi = NetNavi(data_dir=args.data_dir, cfg=cfg)

    print(f"\n=== 成長シミュレーション: {args.turns} ターン / 語彙源 {len(corpus)} 発話 ===\n")
    t0 = time.time()
    i = 0
    strategies = collections.Counter()
    # 戦略ごとの品質を測る(採用されたかどうかではなく、候補そのものの出来)
    navi.debug_pool = True
    quality = collections.defaultdict(lambda: {"n": 0, "flu": 0.0, "pmi": 0.0,
                                               "judged": 0, "broken": 0})
    novel = collections.Counter()   # 思い出せる発話が無いターンでの採用戦略
    for turn in range(1, args.turns + 1):
        # 人間の会話らしく、たまに同じ話題を繰り返す
        if rng.random() < 0.25:
            line = corpus[rng.randrange(len(corpus))]
        else:
            line = corpus[i % len(corpus)]
            i += 1
        reply = navi.talk(line)
        strategies[reply.strategy] += 1
        if not any(c.strategy.startswith("mimic") for c in navi.last_pool):
            novel[reply.strategy] += 1
        for c in navi.last_pool:
            q = quality[c.strategy]
            q["n"] += 1
            q["flu"] += c.detail.get("flu", 0.0)
            f = c.detail.get("frame", 0.0)
            if f:
                q["pmi"] += f
                q["judged"] += 1
                if f < 0:
                    q["broken"] += 1
        # 常駐スレッドの代わりに、一定間隔で「深い睡眠」を挟む
        if turn % args.sleep_every == 0:
            navi.deep_sleep()
        if turn in CHECKPOINTS or turn == args.turns:
            st = navi.status()
            c = st["comprehension"]
            print(f"--- turn {turn:>5}  [{st['stage'].code}] "
                  f"語彙{st['vocab']:>5}  シナプス{st['edges']:>6}  "
                  f"理解度{c['understanding']:.3f}/{c['core']:.3f} "
                  f"(ppl{c['perplexity']:.0f} 穴埋め{c['cloze_mrr']:.2f} "
                  f"格{c['slot_precision']:.2f}) "
                  f"事実{int(st['facts']['facts'])} ---")
            print(f"  あなた: {line}")
            print(f"  Navi  : {reply.text}   ({reply.strategy})")
            # 同じ問いかけへの応答も見る
            for probe in ("おはよう。", "きみの名前は。", "きみは元気かい。",
                          "今日はいい天気だね。"):
                r = navi.talk(probe, learn=False)
                print(f"  > {probe:<12} -> {r.text}   ({r.strategy})")
            print()

    dt = time.time() - t0
    st = navi.status()
    print(f"=== 完了: {args.turns} ターン / {dt:.1f} 秒 "
          f"({args.turns / max(dt, 1e-9):.1f} turn/s) ===")
    print(f"  段階      : {st['stage'].code} ({st['stage'].name})")
    print(f"  語彙      : {st['vocab']}   シナプス: {st['edges']}")
    print(f"  記憶発話  : {st['phrases']}  連想: {st['assoc']}")
    print(f"  文脈一貫性: {st['coherence']:.2f}")
    print(f"  新規発話率: {st['novelty'] * 100:.1f}% "
          f"(オペレーターが言っていない文を口にした割合)")
    c = st["comprehension"]
    print(f"  理解度    : {c['understanding']:.3f} "
          f"(perplexity {c['perplexity']:.1f} / 穴埋めMRR {c['cloze_mrr']:.3f} "
          f"/ 格スロット {c['slot_precision']:.3f})")
    print(f"  格フレーム: 述語{int(st['frames']['preds'])} "
          f"スロット{int(st['frames']['slots'])} 項{int(st['frames']['triples'])}")
    print(f"  事実      : {int(st['facts']['facts'])}件 "
          f"(矛盾{int(st['facts']['conflicts'])}件)")
    print(f"  意味ベクトル: {'あり' if st['embed_ready'] else 'なし'} "
          f"(再編成{st['embed_builds']}回)")
    print(f"  活用の記憶: 述語{int(st['realizer']['preds'])}種 "
          f"/ {int(st['realizer']['forms'])}通り")
    print(f"  指示詞    : {int(st['discourse']['resolved'])}"
          f"/{int(st['discourse']['attempted'])} 解決")
    ic = st["intents"]
    print(f"  意図      : k={int(ic['k'])} / 返し方の適切さ "
          f"{ic['appropriateness']:.2f} (平均報酬 {ic['mean_reward']:+.2f} / "
          f"{int(ic['rewarded'])}回)")
    print(f"  接地      : {int(st['senses']['grounded'])}語")
    nn = st["neural"]
    print(f"  神経回路  : {'あり' if nn['ready'] else 'なし'} "
          f"{int(nn['params'])}パラメータ / {int(nn['steps'])}歩 "
          f"/ ppl {nn['perplexity']:.1f} / 移行度 {st['alpha']:.2f}")
    total = sum(strategies.values()) or 1
    mix = " ".join(f"{k}:{v * 100 // total}%" for k, v in strategies.most_common(6))
    print(f"  採用された戦略: {mix}")
    nt = sum(novel.values())
    if nt:
        nmix = " ".join(f"{k}:{v * 100 // nt}%" for k, v in novel.most_common(5))
        print(f"  思い出せない時({nt}ターン): {nmix}")
    print("  候補の品質(採用の有無によらず):")
    print(f"    {'戦略':<12}{'本数':>7}{'流暢さ':>9}{'選択選好':>10}{'破綻率':>9}")
    for name, q in sorted(quality.items(), key=lambda kv: -kv[1]["n"]):
        if q["n"] < 10:
            continue
        flu = q["flu"] / q["n"]
        pmi = q["pmi"] / q["judged"] if q["judged"] else 0.0
        broken = q["broken"] / q["judged"] * 100 if q["judged"] else 0.0
        print(f"    {name:<12}{q['n']:>7}{flu:>9.2f}{pmi:>+10.2f}{broken:>8.1f}%")
    print(f"  昇格ログ  : {st['history']}")
    navi.shutdown()


if __name__ == "__main__":
    main()
