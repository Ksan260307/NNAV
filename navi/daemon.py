"""常駐スレッド。

オペレーターが入力した瞬間だけ脳が動く「受動的」な構造から抜け出し、
放置されている間もナビが裏で動き続けるようにする(元記事の課題①)。

  * 擬似睡眠と自己対話 … 無操作が続くと、脳内でひとりごとを生成して
                          既存のシナプスをごく弱く再強化する(夢)
  * 自律的な調べ物     … 好奇心の対象をネットで調べる(厳しいレート制限つき)
  * 自発発話           … 十分に育つと、自分から話しかけてくる
  * 非同期の永続化     … 差分(WAL)を定期的にディスクへ流す
"""

from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional


class NaviDaemon(threading.Thread):
    def __init__(self, navi, on_speak: Optional[Callable[[str], None]] = None,
                 on_event: Optional[Callable[[str], None]] = None):
        super().__init__(name="navi-daemon", daemon=True)
        self.navi = navi
        self.cfg = navi.cfg
        self.on_speak = on_speak or (lambda s: None)
        self.on_event = on_event or (lambda s: None)
        self._stopev = threading.Event()
        self.last_dream = 0.0
        self.last_speak = 0.0
        self.dreams: List[str] = []
        self.studies: List[str] = []
        self.sleeps: List[str] = []
        self.last_deep = 0.0

    def stop(self) -> None:
        self._stopev.set()

    def run(self) -> None:  # pragma: no cover - 時間依存
        while not self._stopev.wait(self.cfg.daemon_tick_sec):
            try:
                self._tick()
            except Exception as exc:
                self.on_event(f"[daemon] {type(exc).__name__}: {exc}")

    def _tick(self) -> None:
        now = time.time()
        idle = now - self.navi.mood.last_input_at
        self.navi.mood.tick(idle)

        # --- 擬似睡眠(夢) ------------------------------------------------
        if (idle >= self.cfg.dream_idle_sec
                and now - self.last_dream >= self.cfg.dream_interval_sec):
            self.last_dream = now
            text = self.navi.dream()
            if text:
                self.dreams.append(text)
                del self.dreams[:-30]

        # --- 深い睡眠(重い学習はすべてここ) --------------------------------
        if (idle >= self.cfg.neural_idle_sec
                and now - self.last_deep >= self.cfg.neural_idle_sec):
            self.last_deep = now
            r = self.navi.deep_sleep()
            if r:
                parts = []
                if "embed" in r:
                    parts.append("意味空間を編成し直した")
                if "intents" in r:
                    parts.append(f"意図を{r['intents']}種類に整理した")
                if "neural" in r:
                    parts.append(f"神経回路を{r['neural']['steps']}歩ぶん鍛えた "
                                 f"(loss {r['neural']['loss']:.3f})")
                if parts:
                    msg = " / ".join(parts)
                    self.sleeps.append(msg)
                    del self.sleeps[:-20]
                    self.on_event("[睡眠] " + msg)

        # --- 自律的な調べ物 ----------------------------------------------
        if self.cfg.web_enabled and self.navi.web.due():
            stage = self.navi.growth.stage(self.navi.brain.learned_vocab,
                                           self.navi.brain.turns)
            if stage.level >= 3:
                res = self.navi.study()
                if res.ok:
                    msg = (f"「{res.topic}」について調べた "
                           f"({res.sentences}文 / 新しい言葉 {res.new_words}語)")
                    self.studies.append(msg)
                    del self.studies[:-30]
                    self.on_event("[学習] " + msg)
                elif res.reason and "上限" in res.reason:
                    self.navi.web.last_study = now  # 上限中は静かに待つ

        # --- 自発発話 ----------------------------------------------------
        if (idle >= self.cfg.proactive_idle_sec
                and now - self.last_speak >= self.cfg.proactive_idle_sec):
            text = self.navi.speak_up()
            if text:
                self.last_speak = now
                self.navi.pending.append(text)
                if self.cfg.proactive_interrupt:
                    self.on_speak(text)

        # --- 非同期の永続化 ------------------------------------------------
        self.navi.flush()

    # ------------------------------------------------------------------
    def report(self) -> List[str]:
        """留守中に何をしていたかを返す。"""
        out: List[str] = []
        if self.dreams:
            out.append("[夢] " + " / ".join(self.dreams[-3:]))
        if self.studies:
            out.append("[調べ物] " + " / ".join(self.studies[-3:]))
        if self.sleeps:
            out.append("[睡眠] " + " / ".join(self.sleeps[-2:]))
        return out
