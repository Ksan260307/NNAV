"""対話用のコンソール。"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from .config import ORIGIN_LABEL
from .daemon import NaviDaemon
from .navi import NetNavi

C_RESET = "\033[0m"
C_NAVI = "\033[96m"
C_SYS = "\033[90m"
C_WARN = "\033[93m"
C_OK = "\033[92m"
C_BAR = "\033[95m"


def setup_console() -> None:
    """Windows コンソールを UTF-8 + ANSI 対応にする。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if os.name == "nt":
        try:
            import ctypes

            k = ctypes.windll.kernel32
            for handle in (-11, -12):  # STDOUT, STDERR
                h = k.GetStdHandle(handle)
                mode = ctypes.c_uint32()
                if k.GetConsoleMode(h, ctypes.byref(mode)):
                    k.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            pass


HELP = """\
  /status          育ち具合・気分・シナプス数を表示
  /good  /bad      直前の発話を評価する(シナプスを強化/弱化)
  /study [語]      インターネットで調べさせる(語を省略すると好奇心まかせ)
  /web on|off      ネット学習の有効/無効
  /teach <ファイル> テキストファイルを1行1発話として一気に教える
  /pin <語>        その語を忘却の対象から外す
  /facts [語]      覚えている事実を見る
  /frames <述語>   その述語がどんな格を取ると思っているか
  /near <語>       意味ベクトル上で近い語
  /top             いま脳内で重要度の高い語
  /dream           いますぐ夢(自己対話)を見せる
  /diary [n]       ナビが経験したことの記録(WAL)の末尾を見る
  /raw             メタ認知表示([語:品詞])の ON/OFF
  /save            いま保存する
  /help  /exit"""


def banner(navi: NetNavi) -> None:
    st = navi.status()
    print("=" * 62)
    print("   NetNavi Core Engine   -   自律成長型 電脳生命体")
    print("=" * 62)
    print(f"{C_SYS}段階 {st['stage'].code} ({st['stage'].name}) / "
          f"語彙 {st['vocab']} / 対話 {st['turns']}ターン{C_RESET}")
    print(f"{C_SYS}{st['stage'].note}{C_RESET}")
    print(f"{C_SYS}「/help」でコマンド一覧。「exit」で終了します。{C_RESET}")
    if navi.cfg.web_enabled and "set your email" in navi.cfg.web_user_agent:
        print(f"{C_WARN}[注意] ネット学習の User-Agent が既定のままです。"
              f"{os.path.join(navi.cfg.data_dir, 'config.json')} の "
              f"web_user_agent に連絡先を書いてください。{C_RESET}")


def show_status(navi: NetNavi) -> None:
    s = navi.status()
    cur, nxt, ratio = s["stage"], s["next"], s["progress"]
    bar_len = 28
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\n{C_BAR}== 成長 =={C_RESET}")
    print(f"  段階      : {cur.code} / {cur.name}  (Lv.{cur.level})")
    print(f"  {cur.note}")
    if nxt:
        print(f"  次の段階  : {nxt.code} [{bar}] {ratio*100:5.1f}%")
        print(f"              必要: 語彙 {nxt.need_vocab} / 対話 {nxt.need_turns}ターン"
              + (f" / 理解度 {nxt.need_understanding:.2f}"
                 if nxt.need_understanding > 0 else ""))
    else:
        print("  最終段階に到達しています。")
    print(f"\n{C_BAR}== 電脳 =={C_RESET}")
    print(f"  語彙      : {s['vocab']} 語   (うち内訳 "
          + ", ".join(f"{ORIGIN_LABEL.get(k, k)}:{v}" for k, v in sorted(s["origins"].items()))
          + ")")
    print(f"  シナプス  : {s['edges']} 結合 / 連想 {s['assoc']} 語 / 記憶発話 {s['phrases']} 件")
    print(f"  学習量    : {s['tokens']} トークン (ネット由来 {s['web_tokens']})")
    print(f"  文脈一貫性: {s['coherence']:.2f}  (高いほど流暢)")
    print(f"  未書出差分: {s['wal_lines']} 行")

    c = s["comprehension"]
    fr, fa = s["frames"], s["facts"]
    print()
    print(f"{C_BAR}== 理解 =={C_RESET}")
    ubar = int(24 * min(1.0, c["understanding"]))
    print(f"  理解度    : {c['understanding']:.3f} [{'#' * ubar}{'-' * (24 - ubar)}]")
    print(f"    次の語の予測  : perplexity {c['perplexity']:.1f} "
          f"(logP {c['logprob']:.2f} / {int(c['samples'])}発話で測定)")
    print(f"    穴埋め(MRR)   : {c['cloze_mrr']:.3f}")
    print(f"    格スロット予測: {c['slot_precision']:.3f}")
    print(f"  格フレーム: 述語{int(fr['preds'])} / スロット{int(fr['slots'])} "
          f"/ 項{int(fr['triples'])}")
    print(f"  事実      : {int(fa['facts'])}件 (矛盾 {int(fa['conflicts'])}件)")
    print(f"  意味ベクトル: {'構築済み' if s['embed_ready'] else '未構築'} "
          f"(再編成 {s['embed_builds']}回)")
    for old, new in s["conflicts"]:
        print(f"  {C_WARN}? {old.text()}  <->  {new.text()}{C_RESET}")
    m = s["mood"]
    print(f"\n{C_BAR}== 気分 =={C_RESET}")
    print(f"  {s['mood_label']}  元気{m.energy:.2f} 好奇心{m.curiosity:.2f} "
          f"親密度{m.bond:.1f} 機嫌{m.valence:+.2f}")
    print(f"\n{C_BAR}== ネット学習 =={C_RESET}")
    print(f"  状態      : {'有効' if s['web_enabled'] else '無効'} / {s['web_budget']}")
    if s["studied"]:
        print("  調べた語  : " + ", ".join(w for w, _ in s["studied"]))
    for line in s["web_log"]:
        print(f"  {C_SYS}{line}{C_RESET}")
    if s["top_words"]:
        print(f"\n{C_BAR}== いま気になっている言葉 =={C_RESET}")
        print("  " + " ".join(f"{w}({v})" for w, v in s["top_words"]))
    print()


def show_diary(navi: NetNavi, n: int = 15) -> None:
    """ナビが経験したことの記録(WAL)を読む。"""
    import json

    navi.flush()
    path = navi.journal.path
    if not os.path.exists(path):
        print(f"{C_SYS}まだ記録がありません。{C_RESET}")
        return
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("op") in ("learn", "fb"):
                rows.append(r)
    if not rows:
        print(f"{C_SYS}まだ記録がありません。{C_RESET}")
        return
    for r in rows[-n:]:
        ts = time.strftime("%m/%d %H:%M", time.localtime(r.get("ts", 0)))
        if r["op"] == "learn":
            src = ORIGIN_LABEL.get(r.get("o"), "?")
            print(f"{C_SYS}{ts} 学習 <{src}> {r.get('x', '')[:60]}{C_RESET}")
        else:
            d = r.get("d", 0.0)
            print(f"{C_SYS}{ts} 評価 {d:+.1f} "
                  f"({'ほめられた' if d > 0 else 'しかられた'}){C_RESET}")


def run(data_dir: str = "navi_data") -> None:
    setup_console()
    navi = NetNavi(data_dir=data_dir)
    show_raw = True

    def speak(text: str) -> None:
        # 入力待ちに割り込んで喋る
        sys.stdout.write(f"\r\033[K\n{C_NAVI}[Navi]{C_RESET} {text}\n\nあなた: ")
        sys.stdout.flush()

    def event(text: str) -> None:
        sys.stdout.write(f"\r\033[K{C_SYS}{text}{C_RESET}\n\nあなた: ")
        sys.stdout.flush()

    daemon: Optional[NaviDaemon] = None
    if navi.cfg.daemon_enabled:
        daemon = NaviDaemon(navi, on_speak=speak, on_event=event)
        daemon.start()

    banner(navi)
    if daemon:
        for line in daemon.report():
            print(f"{C_SYS}{line}{C_RESET}")

    try:
        while True:
            try:
                raw = input("\nあなた: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            text = raw.strip()
            if not text:
                continue

            # --- コマンド -----------------------------------------------
            low = text.lower()
            if low in ("exit", "quit", "終了", "/exit", "/quit"):
                break
            if low in ("/help", "help", "?"):
                print(HELP)
                continue
            if low in ("/status", "/s"):
                show_status(navi)
                continue
            if low == "/good":
                print(f"{C_OK}{navi.feedback(True)}{C_RESET}")
                continue
            if low == "/bad":
                print(f"{C_WARN}{navi.feedback(False)}{C_RESET}")
                continue
            if low == "/raw":
                show_raw = not show_raw
                print(f"{C_SYS}メタ認知表示: {'ON' if show_raw else 'OFF'}{C_RESET}")
                continue
            if low == "/save":
                navi.save()
                print(f"{C_SYS}電脳メモリを保存しました。{C_RESET}")
                continue
            if low == "/top":
                print("  " + " ".join(f"{w}({v})" for w, v in navi.brain.top_words(20)))
                continue
            if low == "/dream":
                t = navi.dream()
                print(f"{C_SYS}[夢] {t or '……(まだ夢を見られません)'}{C_RESET}")
                continue
            if low.startswith("/diary"):
                parts = text.split()
                show_diary(navi, int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15)
                continue
            if low.startswith("/web"):
                parts = low.split()
                if len(parts) > 1 and parts[1] in ("on", "off"):
                    navi.cfg.web_enabled = parts[1] == "on"
                    navi.cfg.save()
                print(f"{C_SYS}ネット学習: "
                      f"{'有効' if navi.cfg.web_enabled else '無効'} / "
                      f"{navi.web.budget.status()}{C_RESET}")
                continue
            if low.startswith("/study"):
                topic = text[len("/study"):].strip() or None
                print(f"{C_SYS}……調べています{C_RESET}", flush=True)
                res = navi.study(topic)
                if res.ok:
                    print(f"{C_OK}[学習] 「{res.topic}」<- {res.title} / "
                          f"{res.sentences}文 / 新しい言葉 {res.new_words}語{C_RESET}")
                    print(f"{C_SYS}  拾った文: {res.excerpt}…{C_RESET}")
                else:
                    print(f"{C_WARN}[学習] できませんでした: {res.reason}{C_RESET}")
                continue
            if low.startswith("/teach"):
                path = text[len("/teach"):].strip().strip('"')
                try:
                    n = navi.teach_file(path)
                    print(f"{C_OK}{n} 行を学習しました。"
                          f"語彙 {navi.brain.learned_vocab} 語{C_RESET}")
                except FileNotFoundError:
                    print(f"{C_WARN}ファイルが見つかりません: {path}{C_RESET}")
                continue
            if low.startswith("/facts"):
                key = text[len("/facts"):].strip()
                fs = (navi.facts.about(key, navi.brain.turns) if key
                      else sorted(navi.facts.all_facts(),
                                  key=lambda f: -f.confidence(navi.brain.turns))[:20])
                if not fs:
                    print(f"{C_SYS}まだ何も覚えていません。{C_RESET}")
                for f in fs[:20]:
                    src = ORIGIN_LABEL.get(f.origin, "?")
                    print(f"  {f.text()}   {C_SYS}<{src}> x{f.count:.0f}{C_RESET}")
                continue
            if low.startswith("/frames"):
                pred = text[len("/frames"):].strip()
                cases = navi.frames.cases_of(pred)
                if not cases:
                    print(f"{C_SYS}「{pred}」の使われ方をまだ知りません。{C_RESET}")
                for case, tot in cases:
                    fillers = navi.frames.fillers(pred, case, 6)
                    body = " ".join(f"{w}({v:.0f})" for w, v in fillers)
                    print(f"  {pred}[{case}] x{tot:.0f}  <- {body}")
                continue
            if low.startswith("/near"):
                word = text[len("/near"):].strip()
                wid = navi.brain.word2id.get(word)
                if wid is None:
                    print(f"{C_SYS}「{word}」はまだ知りません。{C_RESET}")
                elif not navi.embed.ready():
                    print(f"{C_SYS}意味ベクトルがまだ構築されていません"
                          f"(会話を重ねるか /dream で作られます)。{C_RESET}")
                else:
                    nb = navi.embed.neighbors(wid, 8)
                    if nb:
                        print("  " + "  ".join(
                            f"{navi.brain.id2word[i]}({v:.2f})" for i, v in nb))
                    else:
                        sim = navi.frames.similar_nouns(word, 8)
                        print("  " + ("  ".join(f"{w}({v:.2f})" for w, v in sim)
                                      or "近い語が見つかりません"))
                continue
            if low.startswith("/pin"):
                word = text[len("/pin"):].strip()
                print(f"{C_SYS}{navi.pin(word)}{C_RESET}")
                continue

            # --- 対話 ---------------------------------------------------
            if navi.pending:
                navi.pending.clear()
            reply = navi.talk(text)
            body = reply.annotated if show_raw else reply.text
            print(f"\n{C_NAVI}[Navi]{C_RESET} {body}")
            if reply.new_words:
                print(f"{C_SYS}       (新しい言葉を {reply.new_words} 語おぼえた){C_RESET}")
            for f in reply.facts:
                print(f"{C_SYS}       (おぼえた: {f.text()}){C_RESET}")
            for old, new in reply.clashes:
                print(f"{C_WARN}       (あれ? 前は「{old.text()}」だったけど"
                      f"「{new.text()}」になった){C_RESET}")
            if reply.stage_up:
                st = reply.stage_up
                print(f"\n{C_OK}*** 成長: {st.code} / {st.name} に到達 ***{C_RESET}")
                print(f"{C_OK}    {st.note}{C_RESET}")
    finally:
        if daemon:
            daemon.stop()
        print(f"\n{C_SYS}[System] 電脳メモリを保存しています……{C_RESET}")
        navi.shutdown()
        print(f"{C_NAVI}[Navi]{C_RESET} （ログアウトしました）")
