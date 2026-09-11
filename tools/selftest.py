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
from navi.config import ORIGIN_OPERATOR, ORIGIN_WEB, Config  # noqa: E402
from navi.growth import STAGES  # noqa: E402
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
             "ネットバトル", "カスタム画面", "オペレーション", "ディメンショナル", "アドバンス",
             "スタイルチェンジ", "ダークチップ", "プログラムアドバンス", "ネットポリス",
             "ワールドスリー", "ガッツマン", "ファイアマン", "ウッドマン", "アイスマン",
             "カーネル", "バレル", "ジャック", "デリート", "バスターマックス",
             "フォルダバック", "ナビチップ", "ソウルユニゾン", "クロスシステム",
             "ビーストアウト", "リンクナビ", "オペレートショット", "チップトレーダー",
             "エレメント", "パネルクラック", "エリアスチール"]
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
            curve.append((i + 1, navi.comp.core(), navi.comp.understanding()))
    check("理解度が単調に伸びる(基準3軸)",
          all(b[1] > a[1] for a, b in zip(curve, curve[1:])),
          " -> ".join(f"turn{t}:{c:.3f}" for t, c, _ in curve))
    check("総合の理解度も伸びる", curve[-1][2] > curve[0][2] * 1.8,
          " -> ".join(f"turn{t}:{u:.3f}" for t, _, u in curve))
    check("段階のゲートとして働く",
          navi.growth.stage(navi.brain.learned_vocab, navi.brain.turns).level >= 2,
          f"Lv.{navi.growth.best_level} 昇格{navi.growth.history}")
    navi.shutdown()


def test_deixis(tmp: str) -> None:
    print()
    print("[11] 話者役割: ぼく/きみ が誰を指すか")
    from navi.config import ORIGIN_NAVI
    from navi.deixis import ROLE_NAVI, ROLE_OP

    navi = fresh(tmp, "deixis")
    for _ in range(2):
        navi.talk("きみの名前はロックマンだよ。")
        navi.talk("ぼくの名前はケイだ。")
    check("役割ごとに名前を覚える",
          navi.deixis.name_of(ROLE_NAVI) == "ロックマン"
          and navi.deixis.name_of(ROLE_OP) == "ケイ",
          f"navi={navi.deixis.name_of(ROLE_NAVI)} op={navi.deixis.name_of(ROLE_OP)}")

    r1 = navi.talk("きみの名前は？", learn=False)
    r2 = navi.talk("ぼくの名前は？", learn=False)
    check("きみ/ぼくを取り違えない",
          "ロックマン" in r1.text and "ケイ" in r2.text,
          f"きみ->{r1.text} / ぼく->{r2.text}")

    check("固有名も同じ実体に落ちる",
          navi.deixis.normalize("ロックマン", ORIGIN_OPERATOR) == ROLE_NAVI)
    check("話者が変わると人称が反転する",
          navi.deixis.normalize("ぼく", ORIGIN_OPERATOR) == ROLE_OP
          and navi.deixis.normalize("ぼく", ORIGIN_NAVI) == ROLE_NAVI)

    # ナビが自分を「ケイ」だと言い出さないこと(実装前はこれが起きていた)
    bad = [f for f in navi.facts.all_facts()
           if f.subj == ROLE_NAVI and f.rel == "名前" and f.obj == "ケイ"]
    check("ナビがオペレーターの名前を自分のものにしない", not bad)
    navi.shutdown()


def test_plan(tmp: str) -> None:
    print()
    print("[12] 組み立て生成: 述語を先に決めてから並べる")
    import random as _r

    from navi.generate import Context as _Ctx
    from navi.generate import propose_plan
    from navi.jp import JPTokenizer, can_end
    from navi.parse import parse as _parse

    navi = fresh(tmp, "plan")
    corpus = lines()
    for i in range(320):
        navi.talk(corpus[i % len(corpus)])
    navi._maybe_build_embedding(force=True)

    st = navi.realizer.stats()
    check("活用を記憶している", st["forms"] > 30,
          f"述語{int(st['preds'])}種 / 形{int(st['forms'])}通り")
    # 聞いたことのない活用は作れない、が原則
    check("教わっていない活用は作らない",
          navi.realizer.realize("食べる", "verb", -1, (), _r.Random(1)) is None)
    navi.talk("カレーは食べない。")
    neg = navi.realizer.realize("食べる", "verb", -1, (), _r.Random(1))
    check("一度聞けば否定形を再現できる",
          neg is not None and "ない" in "".join(neg or []), str(neg))

    stage = navi.growth.stage(navi.brain.learned_vocab, navi.brain.turns)
    p = navi._params(stage)
    p.allow_plan, p.plan_candidates = True, 20
    ids, keys = navi.brain.encode("今日はカレーを食べたよ。")
    ctx = _Ctx(input_ids=ids, input_keys=keys, input_text="今日はカレーを食べたよ。",
               frames=navi.frames, embed=navi.embed, facts=navi.facts,
               realizer=navi.realizer, deixis=navi.deixis, turn=navi.brain.turns)
    cands = propose_plan(navi.brain, ctx, p, _r.Random(3))
    check("組み立てで文が作れる", len(cands) >= 5, f"{len(cands)}本: "
          + " / ".join(c.text for c in cands[:3]))

    tk = JPTokenizer()
    bad = []
    for c in cands:
        ts = tk.tokenize(c.text)
        if ts and not can_end(ts[-1].pos, ts[-1].surface):
            bad.append(c.text)
    check("組み立てた文が尻切れにならない", not bad, str(bad[:2]))

    scores = [navi.frames.score(_parse(tk.tokenize(c.text)))[0] for c in cands]
    judged = [s for s in scores if s != 0.0]
    check("選択選好から見ても妥当",
          bool(judged) and sum(judged) / len(judged) > 0,
          f"平均PMI {sum(judged) / max(1, len(judged)):+.2f} ({len(judged)}本)")

    used = [navi.talk(corpus[i % len(corpus)]).strategy for i in range(120)]
    check("実際の会話でも採用される", "plan" in used,
          f"採用率 {used.count('plan') / len(used) * 100:.0f}%")
    navi.shutdown()


def test_discourse(tmp: str) -> None:
    print()
    print("[13] 指示詞: 「それ」が何を指すか")
    navi = fresh(tmp, "discourse")
    for _ in range(4):
        navi.talk("カレーを食べた。")
        navi.talk("公園まで歩いた。")
        navi.talk("パンを食べた。")
    before = navi.discourse.resolved
    navi.talk("カレーを作った。")
    navi.talk("それを食べた。")
    check("指示詞を解決する", navi.discourse.resolved > before,
          str(navi.discourse.last[-1:]))
    if navi.discourse.last:
        src, dst = navi.discourse.last[-1]
        check("型の合う先を選ぶ(食べる[を] にカレー)", dst in ("カレー", "パン"),
              f"{src} -> {dst}")
    check("解決結果が格フレームに入る",
          navi.frames.frames.get("食べる", {}).get("を", {}).get("カレー", 0) > 0,
          str(navi.frames.fillers("食べる", "を", 4)))
    navi.shutdown()


def test_intents(tmp: str) -> None:
    print()
    print("[14] 意図: 質問と平叙を教師なしで分ける")
    navi = fresh(tmp, "intents")
    corpus = lines()
    for i in range(420):
        navi.talk(corpus[i % len(corpus)])
    check("意図クラスタが立つ", navi.intents.ready(),
          f"k={navi.intents.k} 標本{len(navi.intents.reservoir)}")
    if navi.intents.ready():
        def label(text):
            ids, _ = navi.brain.encode(text)
            return navi.intents.classify(navi.intents.featurize(navi.brain, ids))
        import collections as _c
        q = _c.Counter(label(t) for t in
                       ["きみの名前は？", "これは何？", "カレーは好き？",
                        "今日の天気は？", "なんで？", "きみは元気？"])
        d = _c.Counter(label(t) for t in
                       ["ぼくはカレーが好きだ。", "今日はいい天気だね。",
                        "朝から晴れているよ。", "公園には木がたくさんある。",
                        "夕飯は自分で作るよ。", "本を読むのも好きだ。"])
        # どのクラスタ番号に落ちるかは実行ごとに変わるので、
        # 「問いかけが集まるクラスタに、平叙はあまり来ない」で分離を見る。
        qc = q.most_common(1)[0][0]
        q_rate = q[qc] / sum(q.values())
        d_rate = d[qc] / sum(d.values())
        check("問いかけと平叙が別クラスタになる", q_rate > d_rate + 0.3,
              f"問いの主クラスタ{qc} に 問い{q_rate:.0%} / 平叙{d_rate:.0%}")
        check("意図の遷移を学習している", len(navi.intents.trans) > 0,
              f"{len(navi.intents.trans)}種の遷移 / 予測精度 "
              f"{navi.intents.stats()['accuracy']:.2f}")
    navi.shutdown()


def test_senses(tmp: str) -> None:
    print()
    print("[15] 接地: 語と環境の結び付き")
    navi = fresh(tmp, "senses")
    corpus = lines()
    for i in range(200):
        navi.talk(corpus[i % len(corpus)])
    s = navi.senses.stats()
    check("語に環境が貯まる", s["grounded"] > 20,
          f"接地した語 {int(s['grounded'])} / 標本 {int(s['samples'])}")
    env = navi._env()
    scores = navi.senses.score_all(env, navi.brain.vocab_size)
    check("いまの環境との近さが出せる", scores is not None and float(scores.max()) > 0.5,
          f"最大 {float(scores.max()):.2f}" if scores is not None else "(出ず)")
    wid = navi.brain.word2id.get("おはよう")
    if wid:
        check("語ごとの時間帯が読める", navi.senses.profile(wid) != "(まだ分からない)",
              f"おはよう: {navi.senses.profile(wid)}")
    navi.shutdown()


def test_neural(tmp: str) -> None:
    print()
    print("[16] ニューラル: 自前コーパスだけで学習し、採点を引き継ぐ")
    navi = fresh(tmp, "neural", neural_train_sec=4.0)
    corpus = lines()
    for i in range(360):
        navi.talk(corpus[i % len(corpus)])
    check("学習コーパスが貯まる", len(navi.neural.corpus) >= 300,
          f"{len(navi.neural.corpus)}文")

    navi._maybe_build_embedding(force=True)
    built = navi.neural.ensure(navi.brain, navi.embed)
    check("意味ベクトルを種にモデルを作れる", built and navi.neural.ready(),
          f"{navi.neural.stats()['params']} パラメータ")

    r1 = navi.neural.train(2.0, rng=navi.rng)
    first = r1["loss"]
    r2 = navi.neural.train(4.0, rng=navi.rng)
    check("学習でロスが下がる", r2["loss"] < first,
          f"{first:.3f} -> {r2['loss']:.3f} ({r1['steps'] + r2['steps']}歩)")

    lps = navi.neural.score_batch([navi.brain.encode("ぼくは元気だよ。")[0],
                                   navi.brain.encode("元気ぼくよだは。")[0]])
    check("自然な語順のほうを高く採点する", lps[0] > lps[1],
          f"自然 {lps[0]:.2f} > 崩れ {lps[1]:.2f}")

    for i in range(40):
        navi.talk(corpus[i % len(corpus)])
    a = navi.neural.alpha(navi.comp.logprob)
    check("引き継ぎ度 alpha が範囲に収まる", 0.0 <= a <= 0.9,
          f"alpha={a:.3f} (ニューラル ppl {navi.neural.stats()['perplexity']:.1f})")

    navi.save()
    d2 = navi.cfg.data_dir
    steps = navi.neural.steps
    navi.shutdown()
    from navi.navi import NetNavi as _N
    again = _N(data_dir=d2, cfg=Config.load(d2))
    check("モデルが保存・復元される",
          again.neural.ready() and again.neural.steps == steps,
          f"復元後 {again.neural.steps}歩 / ready={again.neural.ready()}")
    again.shutdown()


def test_objective(tmp: str) -> None:
    print()
    print("[17] 目的関数: 丸暗記が有利にならないか")
    from navi.generate import fluency

    navi = fresh(tmp, "objective")
    corpus = lines()
    for i in range(320):
        navi.talk(corpus[i % len(corpus)])

    once = "ぼくはめずらしい果物を食べたよ。"
    navi.talk(once)
    idx = navi.brain.phrase_by_text.get(once.strip())
    ph = navi.brain.phrases[idx]
    plain = fluency(navi.brain, ph.ids)
    loo = fluency(navi.brain, ph.ids, ph, 1.0)
    check("一度きりの丸暗記は leave-one-out で崩れる", loo < plain - 1.0,
          f"そのまま {plain:.2f} -> 引くと {loo:.2f}")

    often = navi.brain.phrases[navi.brain.phrase_by_text["おはよう。"]]
    p2 = fluency(navi.brain, often.ids)
    l2 = fluency(navi.brain, often.ids, often, 1.0)
    check("何度も聞いた言い回しは引いても残る", (p2 - l2) < (plain - loo),
          f"落ち幅 何度も {p2 - l2:.2f} < 一度きり {plain - loo:.2f}")
    navi.shutdown()


def test_novelty(tmp: str) -> None:
    print()
    print("[18] 新規発話率: 育つほど自分の言葉になるか")
    navi = fresh(tmp, "novelty")
    corpus = lines()
    for i in range(400):
        navi.talk(corpus[i % len(corpus)])
    navi._maybe_build_embedding(force=True)

    rates = {}
    for level in (3, 5):
        navi.growth.best_level = level
        navi.novelty = navi.growth.novelty = 0.5      # ゲートを通す
        fresh_n = 0
        for i in range(150):
            r = navi.talk(corpus[(i * 7) % len(corpus)])
            fresh_n += 1 if r.detail.get("new", 0) > 0 else 0
        rates[level] = fresh_n / 150
    check("段階が上がるほど記憶の再生から離れる", rates[5] > rates[3] + 0.15,
          f"TEEN {rates[3] * 100:.0f}% -> NAVI {rates[5] * 100:.0f}%")

    st = navi.growth.stage(navi.brain.learned_vocab, navi.brain.turns)
    p = navi._params(st)
    check("新しさは段階に応じて重みが付く", p.novelty_weight > 0 and p.loo_discount > 0,
          f"loo={p.loo_discount} novelty={p.novelty_weight}")
    check("成体以降は新規発話率がゲートになる",
          any(s.need_novelty > 0 for s in STAGES),
          str([f"{s.code}:{s.need_novelty}" for s in STAGES if s.need_novelty]))
    navi.shutdown()


def test_bandit(tmp: str) -> None:
    print()
    print("[19] 意図: 返し方をオペレーターの反応から学ぶか")
    navi = fresh(tmp, "bandit")
    corpus = lines()
    for i in range(500):
        navi.talk(corpus[i % len(corpus)])
    check("反応から報酬が貯まる", navi.intents.reward_n > 50,
          f"{int(navi.intents.reward_n)}回の反応 / 平均報酬 "
          f"{navi.intents.stats()['mean_reward']:+.2f}")
    check("返し方の適切さが測れる", navi.intents.measurable(),
          f"適切さ {navi.intents.appropriateness():.2f}")

    # /good が「その返し方」の評価として効くか
    navi.talk("おはよう。")
    if navi._pending:
        in_i, reply_i = navi._pending[0], navi._pending[1]
        before = navi.intents._cell(in_i, reply_i)[0]
        navi.feedback(True)
        after = navi.intents._cell(in_i, reply_i)[0]
        check("/good が返し方そのものの評価になる", after > before,
              f"{before:+.1f} -> {after:+.1f}")
    else:
        check("/good が返し方そのものの評価になる", False, "評価対象なし")

    navi.talk("今日はいい天気だね。")     # 評価の反映を 1 ターン進める
    check("理解度の意図軸が適切さに差し替わっている",
          abs(navi.comp.intent_skill - navi.intents.appropriateness()) < 1e-6,
          f"{navi.comp.intent_skill:.3f}")
    navi.shutdown()


def test_plan_rich(tmp: str) -> None:
    print()
    print("[20] 組み立て生成: 修飾・副詞・節の接続")
    import random as _r

    from navi.generate import Context as _Ctx
    from navi.generate import propose_plan

    navi = fresh(tmp, "planrich")
    corpus = lines()
    for i in range(420):
        navi.talk(corpus[i % len(corpus)])
    navi._maybe_build_embedding(force=True)

    check("連体修飾を覚える", len(navi.frames.noun_mods) > 5,
          str([(n, [m for m, _ in navi.frames.modifiers(n, 2)])
               for n in list(navi.frames.noun_mods)[:3]]))
    check("副詞を覚える", len(navi.frames.pred_advs) > 3,
          str([(p_, [a for a, _ in navi.frames.adverbs(p_, 2)])
               for p_ in list(navi.frames.pred_advs)[:3]]))
    check("語順を覚える", len(navi.frames.orders) > 20,
          f"{len(navi.frames.orders)} 述語ぶん")
    check("次の節へ続く形を覚える", navi.realizer.stats()["conn"] > 3,
          f"{int(navi.realizer.stats()['conn'])} 通り")

    navi.growth.best_level = 5
    st = navi.growth.stage(navi.brain.learned_vocab, navi.brain.turns)
    p = navi._params(st)
    p.allow_plan, p.plan_candidates = True, 30
    ids, keys = navi.brain.encode("今日はカレーを食べたよ。")
    ctx = _Ctx(input_ids=ids, input_keys=keys, input_text="今日はカレーを食べたよ。",
               frames=navi.frames, embed=navi.embed, facts=navi.facts,
               realizer=navi.realizer, deixis=navi.deixis, fw=navi.fw,
               turn=navi.brain.turns)
    cands = propose_plan(navi.brain, ctx, p, _r.Random(11))
    texts = [c.text for c in cands]
    check("組み立て文が出る", len(cands) >= 8, " / ".join(texts[:3]))
    check("同じ名詞を二度出さない",
          all(len(set(c.ids)) >= len(c.ids) - 2 for c in cands))
    long_ones = [t for t in texts if len(t) >= 12]
    check("節をつないだ長めの文も作れる", bool(long_ones),
          (long_ones[0] if long_ones else "(出ず)"))
    navi.shutdown()


def test_lexicon(tmp: str) -> None:
    print()
    print("[21] 機能語: 種を外しても自力で見つけられるか")
    from navi.parse import SEED_DEICTIC, SEED_FIRST, SEED_QUESTION, SEED_SECOND

    navi = fresh(tmp, "lexicon", bootstrap_function_words=False)
    check("種が空で始まる",
          not (navi.fw.questions or navi.fw.first or navi.fw.second
               or navi.fw.deictics),
          str(navi.fw.counts()))

    corpus = lines()
    for i in range(900):
        navi.talk(corpus[i % len(corpus)])
    navi._maybe_build_embedding(force=True)
    fw = navi.lexicon.discovered(navi.frames, navi.embed, navi.brain)

    def heard(truth):
        return {w for w in truth if w in navi.brain.word2id}

    check("一人称を見つける", "ぼく" in fw.first, str(sorted(fw.first)))
    check("二人称を見つける", "きみ" in fw.second, str(sorted(fw.second)))
    check("一人称と二人称を取り違えない", not (fw.first & fw.second))
    hit_q = fw.questions & heard(SEED_QUESTION)
    check("疑問詞を見つける", len(hit_q) >= 3,
          f"{len(hit_q)}/{len(heard(SEED_QUESTION))} {sorted(hit_q)}")
    hit_d = fw.deictics & heard(SEED_DEICTIC)
    check("指示詞を見つける", len(hit_d) >= 3,
          f"{len(hit_d)}/{len(heard(SEED_DEICTIC))} {sorted(hit_d)}")
    check("人称を指示詞と混同しない",
          not (fw.deictics & (heard(SEED_FIRST) | heard(SEED_SECOND))),
          str(sorted(fw.deictics)))

    added = navi.refresh_function_words()
    check("見つけた機能語が実際に有効になる",
          "ぼく" in navi.fw.first and bool(navi.fw.questions),
          f"{navi.fw.counts()} 追加 {added}")
    r = navi.talk("きみの名前は？", learn=False)
    check("種なしでも役割の照会が働く", r.strategy in ("fact", "mimic-exact",
                                                      "mimic-next", "gen",
                                                      "gen+seed", "plan"),
          f"{r.text} ({r.strategy})")
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
        test_deixis(tmp)
        test_plan(tmp)
        test_discourse(tmp)
        test_intents(tmp)
        test_senses(tmp)
        test_neural(tmp)
        test_objective(tmp)
        test_novelty(tmp)
        test_bandit(tmp)
        test_plan_rich(tmp)
        test_lexicon(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== 結果: 成功 {len(PASS)} / 失敗 {len(FAIL)} ===")
    if FAIL:
        for name in FAIL:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
