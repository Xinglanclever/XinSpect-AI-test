#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一層「機器專屬問題」判斷的守門：正反兩面都要對。

為什麼要單獨測這一層：它決定「要不要叫模型」。誤判成機器問題 → 文件真的有的題目被擋掉；
漏判 → 模型撿到一段講「我的電腦」頁面的文件就回答，讀起來像它真的看過那台機器。

用法：
  python eval/test_machine_gate.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, os.pardir, "server", "xinspect_ai_server.py")

# (問題, 應該是機器專屬嗎, 分類應是 config 嗎；None = 不管)
CASES = [
    # ── 要攔：配置／型號 ─────────────────────────────────────────────
    ("我的電腦配置是什麼？", True, True),
    ("這台電腦的顯示卡是哪一張？", True, True),
    ("我这台电脑的内存有多少？", True, True),
    ("幫我看一下我的記憶體有多少", True, True),
    ("What are my PC specs?", True, True),
    ("what gpu do i have?", True, True),
    # ── 要攔：讀值 ───────────────────────────────────────────────────
    ("我這台電腦現在 CPU 幾度？", True, False),
    ("我的cpu幾度", True, False),
    ("幫我看一下我的風扇轉速", True, False),
    ("How hot is my CPU right now?", True, False),
    # ── 不該攔：這些是文件問題或一般問答 ─────────────────────────────
    ("曦覽會不會上傳我的資料？", False, None),
    ("曦覽的系統需求是什麼？", False, None),
    ("曦覽在我的電腦上要多少空間？", False, None),
    ("我的電腦跑得動曦覽嗎？", False, None),
    ("什麼是六態？", False, None),
    ("如何安裝曦覽？", False, None),
    ("How does XinSpect read MSR?", False, None),
    ("What is a PCIe root complex?", False, None),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    spec = importlib.util.spec_from_file_location("xai", SERVER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    bad = 0
    for q, want_gate, want_config in CASES:
        got = m.looks_machine_specific(q)
        ok = got == want_gate
        if ok and want_gate and want_config is not None:
            # 分類只看內部旗標：走的是不是 config 那一段
            import re
            is_reading = bool(re.search(
                r"幾度|几度|溫度|温度|多熱|多热|幾瓦|几瓦|跑幾分|跑几分|風扇|风扇"
                r"|how hot|temperature|celsius|fan speed", q, re.I))
            ok = is_reading != want_config
        if not ok:
            bad += 1
        print(("  ok  " if ok else "  !!  ") + f"{q}  → 攔={got}（預期 {want_gate}）")

    print(f"\n{len(CASES) - bad}/{len(CASES)} 通過")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())