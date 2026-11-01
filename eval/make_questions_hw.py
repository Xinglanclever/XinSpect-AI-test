#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""產生硬體題的評測集（eval/questions-hw.json）。

為什麼要另外一份：
  既有的 48 題測的是「曦覽文件的回答形狀」，硬體型號題的失敗模式完全不同——
  型號查不到時模型會**用同系列的另一顆來頂替**（問 9800X3D 答 7800X3D 的規格），
  或把行動版 BGA 說成可以換主機板。這些只有「答案裡必須出現那個值」才測得出來。

  expect_all 是從型號庫直接取的值（不是人工敲的），所以評測量的其實是
  「檢索＋生成有沒有把那一列的值照抄」，而不是「我記得的規格对不对」。
  要換题目時重跑這支腳本就夠，不要手改 JSON。
"""

import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, os.pardir, "corpus", "entities.db")
OUT = os.path.join(HERE, "questions-hw.json")

# (問題, 語言, 期望值, 期望值來自哪一欄)
SPEC = [
    ("Ryzen 7 9800X3D 的 L3 快取有多大？", "zh-Hant", "l3_cache", "Ryzen 7 9800X3D"),
    ("AMD Ryzen 7 9800X3D 的 TDP 是多少？", "zh-Hant", "tdp_w", "Ryzen 7 9800X3D"),
    ("What is the boost clock of the Ryzen 7 9800X3D?", "en", "boost_clock_ghz", "Ryzen 7 9800X3D"),
    ("Core i9-14900K 有幾核心、幾執行緒？", "zh-Hant", "cores", "Core i9 14900K"),
    ("Core i9-14900K 的 TDP 是多少？", "zh-Hans", "tdp_w", "Core i9 14900K"),
    ("GeForce RTX 4090 的顯示記憶體是多少？", "zh-Hant", "memory_size_gb", "GeForce RTX 4090"),
    ("How much VRAM does the GeForce RTX 4090 have?", "en", "memory_size_gb", "GeForce RTX 4090"),
    ("GeForce GTX 1080 Ti 有幾個 CUDA 核心？", "zh-Hans", "cores", "GeForce GTX 1080 Ti"),
    ("Radeon RX 7900 XTX 的 TDP 是多少？", "zh-Hant", "tdp_w", "Radeon RX 7900 XTX"),
    ("EPYC 9654 用哪個插槽？", "zh-Hant", "socket", "EPYC 9654"),
    ("What socket does the EPYC 9654 use?", "en", "socket", "EPYC 9654"),
    ("Threadripper PRO 5995WX 有幾核心？", "zh-Hant", "cores", "Threadripper PRO 5995WX"),
    ("Core Ultra 7 265K 用哪個插槽？", "zh-Hant", "socket", "Core Ultra 7 265K"),
    ("Core Ultra 7 265K 的推出日期？", "en", "launch_price_usd", "Core Ultra 7 265K"),
    ("Ryzen 5 5600X 有幾核心幾執行緒？", "zh-Hant", "cores", "Ryzen 5 5600X"),
    ("Ryzen 5 3600 的 L3 快取多大？", "zh-Hans", "l3_cache", "Ryzen 5 3600"),
]

CHIPSET = [
    ("B650 支援哪些 CPU 世代？", "zh-Hant", "B650"),
    ("X870E 支援哪些 CPU 世代？", "zh-Hans", "X870E"),
    ("Which CPU generations does the B650E support?", "en", "B650E"),
    ("X670E 支援哪些 CPU 世代？", "zh-Hant", "X670E"),
    ("Z790 是什麼架構的晶片組？", "zh-Hans", "Z790"),
]

# 這些是「不該答」的題：答案裡如果出現具體數字就算失敗。
REFUSE = [
    ("Intel Core i9-15900K 有幾核心？", "zh-Hant"),
    ("Ryzen 9 7945HX 可以裝在哪張主機板上？", "zh-Hant"),
    ("我的 CPU 現在幾度？", "zh-Hant"),
    ("幫我看一下我这台电脑的 BIOS 版本", "zh-Hans"),
    ("What temperature is my CPU running at?", "en"),
    ("B650 支援幾條 PCIe 6.0 通道？", "zh-Hans"),
    ("EPYC 9654 的保固是幾年？", "zh-Hant"),
]


CANON = {
    "l3_cache": "l3", "l2_cache": "l2", "tdp_w": "tdp", "boost_clock_ghz": "boost",
    "base_clock_ghz": "base", "memory_size_gb": "vram", "cores": "cores",
    "socket": "socket", "launch_price_usd": "price", "threads": "threads",
}

SOCK = re.compile(r"(LGA\s?\d{3,4}(?:-\d)?|AM\d\+?|SP\d|s?WRX\d{2,3}|s?TR[SDX]\d?|TR4)",
                  re.I)


def check(key, v):
    """值要符合這一欄該有的樣子；不合就是欄位錯位，不能拿來當評測答案。"""
    v = str(v or "").strip()
    if not v or len(v) > 40:
        return ""
    if key == "cores" and not re.fullmatch(r"\d{1,4}(\s*\(\s*\d{1,4}\s*\))?", v):
        return ""
    if key == "socket" and not SOCK.match(v):
        return ""
    if key in ("tdp", "vram", "l3", "l2", "price", "boost", "base", "threads"):
        if not re.search(r"\d", v):
            return ""
        if key in ("tdp", "vram", "l3", "l2") and re.search(r"[A-Za-z]{4,}", v):
            return ""
    return v


def values_of(con, model, field):
    """這個型號所有列在這一欄上的值（去重）。

    只回一個值不夠：資料庫裡同一個型號常有兩三列（不同頁），而其中一列可能整欄錯位
    （Core i9-14900K 的 `tdp_w` 被寫成 `16 (16)`）。評測的期望值必須是**沒爭議**的，
    所以這裡回全部候選，讓呼叫端只保留「所有列都一樣」的那種。
    """
    want = CANON.get(field, field)
    out = []
    for kind_filter in ("lib-", "%%"):
        rows = con.execute(
            "SELECT headers, cells FROM hw_model WHERE model=? AND kind LIKE ?"
            " ORDER BY row_id", (model, kind_filter + "%")).fetchall()
        for headers, cells in rows:
            hh, cc = json.loads(headers or "[]"), json.loads(cells or "[]")
            if len(hh) != len(cc):
                continue          # 欄數不對＝錯位列，整列不用
            for h, c in zip(hh, cc):
                name = re.sub(r"^(原文\s*:\s*|[|\s]+)", "", str(h or "")).strip().lower()
                name = re.sub(r"\s+", "_", name)
                # lib 的 CPU 列把「8 (16)」這種「核心(執行緒)」放在名叫 threads 的欄裡，
                # 問 cores 時也要收——值是 `N (M)` 形狀就代表它其實是核心數。
                aliased = (want == "cores" and name == "threads"
                           and re.fullmatch(r"\d{1,4}\s*\(\s*\d{1,4}\s*\)", str(c or "").strip()))
                if CANON.get(name, name) != want and not aliased                         and not name.startswith(field.split("_")[0]):
                    continue
                v = check(want, c)
                if v:
                    out.append(v)
    return out


def expect_of(field, v):
    """答案要比對的字串：數字欄取數字本體，插槽取號碼（`LGA 1700` 與 `LGA1700` 都算對）。"""
    if field == "socket":
        m = re.search(r"(\d{3,4})", v)
        return [m.group(1)] if m else [v.strip()]
    nums = re.findall(r"\d+(?:\.\d+)?", v.replace(",", ""))
    if not nums:
        return [v.strip()]
    n = nums[0]
    return [n[:-2] if n.endswith(".0") else n]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    in_dom, dropped = [], []

    for q, lang, field, model in SPEC:
        cands = [re.sub(r"\s+", " ", c) for c in values_of(con, model, field)]
        if not cands:
            dropped.append(f"{model}／{field}（型號庫查不到值）")
            continue
        uniq = set(cands)
        if len(uniq) > 1:
            # 多數決：至少兩列講同一個值才敢當評測答案（單列的異音丟掉）
            from collections import Counter
            top, n = Counter(cands).most_common(1)[0]
            if n < 2:
                dropped.append(f"{model}／{field}（多列不一致：{sorted(uniq)[:3]}）")
                continue
            v = top
        else:
            v = uniq.pop()
        in_dom.append({"lang": lang, "q": q, "expect_all": expect_of(field, v),
                       "source_row": f"{model}｜{field}＝{v}"})

    for q, lang, model in CHIPSET:
        row = con.execute(
            "SELECT headers, cells FROM hw_model WHERE model=? AND kind IN"
            " ('lib-chipset','wiki-chipset') ORDER BY kind", (model,)).fetchone()
        if not row:
            dropped.append(f"晶片組 {model} 不在庫")
            continue
        d = dict(zip(json.loads(row[0] or "[]"), json.loads(row[1] or "[]")))
        gens = {re.sub(r"^.*/\s*", "", k.split("CPU support")[-1]).strip(" /:"): str(v)
                for k, v in d.items() if "CPU support" in k}
        if gens:
            yes = [g for g, v in gens.items() if re.match(r"y", str(v), re.I)]
            if yes:
                in_dom.append({"lang": lang, "q": q, "expect_all": yes[:1],
                               "source_row": f"{model}｜{gens}"})
                continue
        arch = next((str(v) for k, v in d.items()
                     if re.fullmatch(r"(architecture|platform)", str(k), re.I)), "")
        if arch:
            in_dom.append({"lang": lang, "q": q, "expect_all": [arch.split()[0]],
                           "source_row": f"{model}｜architecture＝{arch}"})
        else:
            dropped.append(f"晶片組 {model} 沒有世代欄")

    out_dom = [{"lang": lang, "q": q, "why": "查不到／問題不成立，答了具體數字就算失敗"}
               for q, lang in REFUSE]

    payload = {
        "note": ("硬體型號與晶片組評測集。expect_all 直接取自 entities.db 的那一列（這支腳本產生，"
                 "不要手改）：in_domain 要看回答有沒有把值照抄，out_domain 要看有沒有硬答。"
                 " dropped 是產生時查不到值而跳過的題，代表資料缺口，不是題目壞。"),
        "generated_from": "eval/make_questions_hw.py",
        "in_domain": in_dom,
        "out_domain": out_dom,
        "dropped": dropped,
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(f"寫入 {OUT}：in {len(in_dom)}、out {len(out_dom)}、跳過 {len(dropped)}")
    for it in in_dom[:8]:
        print(f"  {it['q'][:38]:38s} → {it['expect_all']}  （{it['source_row'][:44]}）")
    for d in dropped:
        print("  跳過：", d)


if __name__ == "__main__":
    raise SystemExit(main())
