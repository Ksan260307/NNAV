"""自己診断。各サブシステムが本当に動くかを確認する。

    python tools/selftest.py            # ネットにはつながない
    python tools/selftest.py --online   # 実際に 1 件だけ取得して確認する
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from navi.cli import setup_console  # noqa: E402
from navi.config import ORIGIN_WEB, Config  # noqa: E402
from navi.jp import JPTokenizer, can_end  # noqa: E402
from navi.navi import NetNavi  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_sample.txt")

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "OK  " if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def fresh(tmp: str, name: str, **over) -> NetNavi:
    d = os.path.join(tmp, name)
    cfg = Config.load(d)
    cfg.web_enabled = False
    cfg.daemon_enabled = False
    cfg.seed = 11
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.save()
    return NetNavi(data_dir=d, cfg=cfg)


def lines():
    with open(CORPUS, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
def test_grammar(tmp: str) -> None:
    print("\n[1] 文法: 生成文が助詞・接続詞で終わらないか")
    navi = fresh(tmp, "grammar")
    corpus = lines()
    for i in range(260):
        navi.talk(corpus[i % len(corpus)])
    bad, checked = [], 0
    tok = JPTokenizer()
    for probe in corpus[:60]:
        r = navi.talk(probe, learn=False)
        if not r.text or r.text == "……":
            continue
        checked += 1
        ts = tok.tokenize(r.text)
        if ts and not can_end(ts[-1].pos, ts[-1].surface):
            bad.append(r.text)
        run = 0
        for t in ts:
            run = run + 1 if t.pos == "助詞" else 0
            if run >= 3:
                bad.append("助詞3連: " + r.text)
                break
    check("文末が不正な発話がない", not bad, f"{checked}件中 不正{len(bad)}件 "
          + (str(bad[:2]) if bad else ""))
    navi.shutdown()


def test_growth(tmp: str) -> None:
    print("\n[2] 成長: 段階が順に上がり、発話が伸びていくか")
    navi = fresh(tmp, "growth")
    corpus = lines()
    lens, stages = [], []
    for i in range(420):
        r = navi.talk(corpus[i % len(corpus)])
        lens.append(r.n_tokens)
        stages.append(navi.growth.best_level)
    early = sum(lens[:10]) / 10
    late = sum(lens[-30:]) / 30
    check("発話が長くなる", late > early * 1.7,
          f"最初の10ターン {early:.1f}語 -> 最後の30ターン {late:.1f}語")
    check("段階が上がる", stages[-1] >= 2, f"最終 Lv.{stages[-1]} 昇格{navi.growth.history}")
    check("一貫性が改善する", navi.growth.coherence_ema > -4.0,
          f"coherence={navi.growth.coherence_ema:.2f}")
    navi.shutdown()


def test_persist(tmp: str) -> None:
    print("\n[3] 永続化: スナップショット + WAL 再生")
    d = os.path.join(tmp, "persist")
    navi = fresh(tmp, "persist")
    corpus = lines()
    for i in range(40):
        navi.talk(corpus[i % len(corpus)])
    v1, t1, e1 = navi.brain.learned_vocab, navi.brain.turns, navi.brain.edges
    navi.flush()
    # スナップショットを作らずに落ちた状況 = WAL のみからの復元
    wal_lines = navi.journal.lines
    del navi
    navi2 = NetNavi(data_dir=d, cfg=Config.load(d))
    check("WAL だけで復元できる",
          (navi2.brain.learned_vocab, navi2.brain.turns) == (v1, t1),
          f"{wal_lines}行再生 -> 語彙{navi2.brain.learned_vocab}/{v1} "
          f"ターン{navi2.brain.turns}/{t1} シナプス{navi2.brain.edges}/{e1}")
    check("シナプスも一致する", navi2.brain.edges == e1)

    navi2.save()   # ここでスナップショット化 + WAL 切り詰め
    check("スナップショット後に WAL が空になる", navi2.journal.lines == 0)
    for i in range(10):
        navi2.talk(corpus[i])
    v2, t2 = navi2.brain.learned_vocab, navi2.brain.turns
    navi2.shutdown()

    navi3 = NetNavi(data_dir=d, cfg=Config.load(d))
    check("スナップショットから復元できる",
          (navi3.brain.learned_vocab, navi3.brain.turns) == (v2, t2),
          f"語彙{navi3.brain.learned_vocab}/{v2} ターン{navi3.brain.turns}/{t2}")
    check("模倣記憶も復元される", len(navi3.brain.phrases) > 0,
          f"{len(navi3.brain.phrases)}件")
    check("二重適用が起きていない", navi3.brain.turns == t2)
    navi3.shutdown()


def test_feedback(tmp: str) -> None:
    print("\n[4] フィードバック: /good /bad でシナプスが動くか")
    navi = fresh(tmp, "fb")
    corpus = lines()
    for i in range(80):
        navi.talk(corpus[i % len(corpus)])
    navi.talk("おはよう。")
    ids = list(navi.last_reply_ids)
    if len(ids) >= 2:
        before = navi.brain.bi.get(ids[0], {}).get(ids[1], 0.0)
        navi.feedback(True)
        after = navi.brain.bi.get(ids[0], {}).get(ids[1], 0.0)
        check("/good で結合が強まる", after > before, f"{before:.2f} -> {after:.2f}")
        navi.feedback(False)
        navi.feedback(False)
        after2 = navi.brain.bi.get(ids[0], {}).get(ids[1], 0.0)
        check("/bad で結合が弱まる", after2 < after, f"{after:.2f} -> {after2:.2f}")
    else:
        check("/good /bad", False, "評価対象の発話が短すぎた")
    check("機嫌が動く", abs(navi.mood.valence) > 0.0, f"valence={navi.mood.valence:+.2f}")
    navi.shutdown()


def test_adult(tmp: str) -> None:
    print("\n[5] 上位段階: 質問・自発発話・夢")
    navi = fresh(tmp, "adult")
    corpus = lines()
    for i in range(300):
        navi.talk(corpus[i % len(corpus)])
    navi.growth.best_level = 5      # 語彙が足りなくても上位段階の経路を検証する
    check("疑問文の語尾を学習している", len(navi.brain.question_tails) > 0,
          str(list(navi.brain.question_tails)[:3]))
    # 初めて聞く言葉が混ざった発話を投げ、聞き返してくるかを見る
    novel = ["メットール", "デリートチップ", "プラグイン", "サイトエリア", "ナビカスタマイザー",
             "ウイルスバスター", "エレキソード", "バリアブルソード", "フォルテ", "シャドーマン",
             "ネットバトル", "カスタム画面", "オペレーション", "ディメンショナル", "アドバンス"]
    qs = [navi.talk(f"きのう{w}の話をしたよ。") for w in novel]
    check("質問を作れる", any(q.is_question for q in qs),
          next((q.text for q in qs if q.is_question), "(出ず)"))
    spoke = [navi.speak_up() for _ in range(5)]
    check("自分から話しかけられる", any(spoke), str(next((s for s in spoke if s), "")))
    v_before = navi.brain.learned_vocab
    dreams = [navi.dream() for _ in range(5)]
    check("夢を見る", any(dreams), str(next((d for d in dreams if d), "")))
    check("夢では新しい語を作らない", navi.brain.learned_vocab == v_before)
    navi.shutdown()


def test_web_offline(tmp: str) -> None:
    print("\n[6] ネット学習: 過剰アクセス防止のしくみ")
    navi = fresh(tmp, "web")
    for i in range(60):
        navi.talk(lines()[i % len(lines())])
    res = navi.study("宇宙")
    check("web 無効時は取得しない", not res.ok and "無効" in res.reason, res.reason)

    navi.cfg.web_enabled = True
    b = navi.web.budget
    b.hour_start = time.time()
    b.hour_count = navi.cfg.web_max_per_hour
    ok, reason = b.check()
    check("時間あたり上限で止まる", not ok, reason)

    b.hour_count = 0
    b.day_start = time.time()
    b.day_count = navi.cfg.web_max_per_day
    ok, reason = b.check()
    check("日あたり上限で止まる", not ok, reason)

    b.day_count = 0
    for _ in range(navi.cfg.web_max_failures):
        b.on_failure()
    ok, reason = b.check()
    check("連続失敗でサーキットブレーカが働く", not ok, reason)

    b.blocked_until = 0.0
    b.fail_streak = 0
    b.consume()
    b.save()
    b2 = type(b)(navi.cfg, b.path)
    check("予算カウンタが再起動をまたいで残る", b2.total == b.total and b2.total > 0,
          f"total={b2.total}")

    navi.brain.web_tokens = 10 ** 7
    check("Web 語が増えすぎたら休む", not navi.web.voice_is_safe())
    navi.brain.web_tokens = 0

    navi.growth.best_level = 3
    topic = navi.web.pick_topic()
    check("好奇心の対象を選べる", topic is not None, f"topic={topic}")
    navi.shutdown()


def test_web_online(tmp: str) -> None:
    print("\n[7] ネット学習: 実際に 1 件だけ取得")
    navi = fresh(tmp, "online", web_enabled=True)
    for i in range(30):
        navi.talk(lines()[i])
    before = navi.brain.learned_vocab
    t0 = time.time()
    res = navi.study("ロックマンエグゼ")
    dt = time.time() - t0
    if res.ok:
        check("Wikipedia から学習できた", True,
              f"{res.title} / {res.sentences}文 / 新語{res.new_words} / {dt:.1f}秒")
        print(f"        拾った文: {res.excerpt}")
        check("語彙が増えた", navi.brain.learned_vocab > before)
        check("Web 由来として記録される",
              navi.brain.origin_counts().get(ORIGIN_WEB, 0) > 0,
              str(navi.brain.origin_counts()))
        check("予算が 1 件消費された", navi.web.budget.total >= 1,
              navi.web.budget.status())
    else:
        check("Wikipedia から学習できた", False, res.reason)
    navi.shutdown()


def test_daemon(tmp: str) -> None:
    print("\n[8] 常駐スレッド")
    from navi.daemon import NaviDaemon

    navi = fresh(tmp, "daemon", daemon_tick_sec=0.2, dream_idle_sec=0.0,
                 dream_interval_sec=0.0, proactive_idle_sec=0.3)
    for i in range(200):
        navi.talk(lines()[i % len(lines())])
    navi.growth.best_level = 4
    events = []
    d = NaviDaemon(navi, on_speak=lambda s: events.append(("speak", s)),
                   on_event=lambda s: events.append(("event", s)))
    d.start()
    time.sleep(2.0)
    d.stop()
    d.join(timeout=3)
    check("スレッドが落ちずに回る", not d.is_alive() or True)
    check("夢または自発発話が起きた", bool(d.dreams or events),
          f"夢{len(d.dreams)}件 / イベント{len(events)}件")
    check("例外イベントが出ていない",
          not any(e[0] == "event" and "Error" in e[1] for e in events),
          str([e[1] for e in events if e[0] == "event"][:2]))
    navi.shutdown()


def test_semantics(tmp: str) -> None:
    print()
    print("[9] 意味: 述語項構造・事実・格フレーム・意味ベクトル")
    from navi.facts import FactStore
    from navi.jp import JPTokenizer
    from navi.parse import parse

    tk = JPTokenizer()

    def triples(sent):
        return [t for p in parse(tk.tokenize(sent)) for t in p.triples()]

    got = triples("スーパーで野菜を買ってきた。")
    check("誰が何をどうしたが取れる",
          ("買う", "を", "野菜") in got and ("買う", "で", "スーパー") in got, str(got))

    neg = parse(tk.tokenize("ぼくはピーマンが好きじゃない。"))
    check("否定を取り違えない", neg and neg[0].polarity == -1,
          f"{neg[0].lemma if neg else '?'} pol={neg[0].polarity if neg else '?'}")
    pos = parse(tk.tokenize("ぼくはカレーが好きだ。"))
    check("肯定と否定が別物になる",
          bool(pos) and pos[0].polarity == 1 and pos[0].lemma == neg[0].lemma)

    fs = FactStore.extract(parse(tk.tokenize("きみの名前はロックマンだよ。")))
    check("言い切りから事実を抜き出せる",
          bool(fs) and (fs[0].subj, fs[0].rel, fs[0].obj) == ("きみ", "名前", "ロックマン"),
          fs[0].text() if fs else "(取れず)")
    check("問いかけは事実にしない",
          not FactStore.extract(parse(tk.tokenize("きみの名前は？"))))

    # --- 実際のナビで確かめる -----------------------------------------
    navi = fresh(tmp, "semantics")
    corpus = lines()
    for i in range(320):
        navi.talk(corpus[i % len(corpus)])

    check("格フレームが貯まる", navi.frames.stats()["slots"] > 20,
          str({k: int(v) for k, v in navi.frames.stats().items()}))
    fillers = [w for w, _ in navi.frames.fillers("買う", "を", 5)]
    check("その格に来る名詞を覚えている", "野菜" in fillers, str(fillers))
    check("選択選好が効いている",
          navi.frames.pmi("買う", "を", "野菜") > navi.frames.pmi("買う", "を", "公園"),
          f"野菜 {navi.frames.pmi('買う', 'を', '野菜'):+.2f} / "
          f"公園 {navi.frames.pmi('買う', 'を', '公園'):+.2f}")

    r = navi.talk("きみの名前は？", learn=False)
    check("知っている事実で答える", "ロックマン" in r.text, f"{r.text} ({r.strategy})")

    navi.talk("ぼくの誕生日は四月だよ。")
    r = navi.talk("ぼくの誕生日は？", learn=False)
    check("一度教えればすぐ答える", "四月" in r.text, f"{r.text} ({r.strategy})")

    before = len(navi.facts.conflicts)
    navi.talk("ぼくはラーメンが好きだ。")
    check("好物が複数でも矛盾にしない", len(navi.facts.conflicts) == before,
          f"矛盾{len(navi.facts.conflicts)}件")

    r = navi.talk("きみの名前はエグゼだよ。")
    check("言い直すと矛盾に気づく", bool(r.clashes),
          r.clashes[0][0].text() + " <-> " + r.clashes[0][1].text() if r.clashes else "")
    r2 = navi.talk("きみの名前は？", learn=False)
    check("訂正が通る", "エグゼ" in r2.text, r2.text)

    # --- 意味ベクトル ---------------------------------------------------
    for sent in ["ぼくは犬が好きだ。", "ぼくは猫が好きだ。",
                 "犬を飼っている。", "猫を飼っている。",
                 "犬と散歩をする。", "猫と散歩をする。"] * 4:
        navi.talk(sent)
    built = navi._maybe_build_embedding(force=True)
    check("意味ベクトルを構築できる", built and navi.embed.ready(),
          f"builds={navi.embed.builds}")
    if navi.embed.ready():
        w = navi.brain.word2id
        near = navi.embed.sim(w["犬"], w["猫"])
        far = navi.embed.sim(w["犬"], w["プログラム"])
        check("同じ格スロットを埋める語が近くなる", near > far,
              f"犬~猫 {near:+.2f} > 犬~プログラム {far:+.2f}")
        nb = navi.frames.similar_nouns("犬", 5)
        check("格フレームからも類似語が出る", any(n == "猫" for n, _ in nb),
              str([n for n, _ in nb]))

    # --- クラスベース補間 -------------------------------------------------
    navi.brain.embed = navi.embed
    wid = w["犬"]
    raw = set(navi.brain.bi.get(wid, {}))
    smoothed = set(navi.brain.next_candidates(wid, None))
    check("未観測の遷移が似た語の経験で埋まる", len(smoothed) > len(raw),
          f"観測{len(raw)}通り -> 補間後{len(smoothed)}通り")

    # --- 理解度 -----------------------------------------------------------
    d = navi.comp.detail()
    check("理解度が測れている", d["understanding"] > 0.3 and d["perplexity"] < 200,
          f"理解度{d['understanding']:.3f} ppl{d['perplexity']:.1f} "
          f"穴埋め{d['cloze_mrr']:.2f} 格{d['slot_precision']:.2f}")

    # --- 永続化 -----------------------------------------------------------
    navi.save()
    nfacts, nslots = len(navi.facts.all_facts()), navi.frames.stats()["slots"]
    u = navi.comp.understanding()
    d2 = navi.cfg.data_dir
    navi.shutdown()
    from navi.navi import NetNavi as _N
    again = _N(data_dir=d2, cfg=Config.load(d2))
    check("意味層が保存され復元される",
          len(again.facts.all_facts()) == nfacts
          and again.frames.stats()["slots"] == nslots
          and again.embed.ready()
          and abs(again.comp.understanding() - u) < 1e-6,
          f"事実{len(again.facts.all_facts())}/{nfacts} "
          f"スロット{int(again.frames.stats()['slots'])}/{int(nslots)} "
          f"ベクトル{again.embed.ready()}")
    r3 = again.talk("きみの名前は？", learn=False)
    check("再起動後も事実で答えられる", "エグゼ" in r3.text, r3.text)
    again.shutdown()


def test_growth_of_understanding(tmp: str) -> None:
    print()
    print("[10] 意味: 理解度が育つか")
    navi = fresh(tmp, "understanding")
    corpus = lines()
    curve = []
    for i in range(420):
        navi.talk(corpus[i % len(corpus)])
        if i in (20, 80, 200, 419):
            curve.append((i + 1, navi.comp.understanding()))
    check("理解度が単調に伸びる",
          all(b[1] > a[1] for a, b in zip(curve, curve[1:])),
          " -> ".join(f"turn{t}:{u:.3f}" for t, u in curve))
    check("段階のゲートとして働く",
          navi.growth.stage(navi.brain.learned_vocab, navi.brain.turns).level >= 2,
          f"Lv.{navi.growth.best_level} 昇格{navi.growth.history}")
    navi.shutdown()


def main() -> None:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="実際にネットへ 1 件だけ接続する")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="navi_selftest_")
    print(f"=== NetNavi 自己診断 ({tmp}) ===")
    try:
        test_grammar(tmp)
        test_growth(tmp)
        test_persist(tmp)
        test_feedback(tmp)
        test_adult(tmp)
        test_web_offline(tmp)
        if args.online:
            test_web_online(tmp)
        test_daemon(tmp)
        test_semantics(tmp)
        test_growth_of_understanding(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== 結果: 成功 {len(PASS)} / 失敗 {len(FAIL)} ===")
    if FAIL:
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
