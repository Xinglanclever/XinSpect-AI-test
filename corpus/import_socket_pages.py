#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從已抓下來的 Wikipedia 世代頁，抽出「這個世代用哪個插槽」，寫成對照表。

為什麼需要這一支：
  清單頁的表格裡，Intel 的 Socket 欄常常是「整個世代共用一個 rowspan 儲存格」，
  我們的列展開在少數頁面會把它吃掉，於是 12700K 這種列查不到插槽。
  但同一個世代頁（Alder Lake、Zen 4…）的正文／資訊框一定寫著
  「LGA 1700 socket」「Socket AM5」——那是頁面級事實，正好補上。

輸出表 hw_socket_page(page, socket, sockets, evidence)：
  page     = 檔名去副檔名（和 hw_model.page 同一組字串）
  socket   = 出現次數最多的那個插槽
  sockets  = 該頁出現過的所有插槽（JSON）
  evidence = 原文片段（人工核對用）

用法：
  python corpus/import_socket_pages.py --docs corpus/raw-fetched --db corpus/entities.db
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter

# 一定要有「Socket」在旁邊才算：光看到 SP5／AM2 這種字串，可能只是引用別頁或別代。
# 兩種寫法都收：`Socket: LGA 1700`、`LGA 1700 socket`。
_SOCK = r"(LGA\s?\d{3,4}(?:-\d)?|AM\d\+?|FM\d\+?|s?WRX\d{1,2}|SP\d|TR[SD]\d)"
# 從一小段文字裡找插槽（資訊框用）
SOCK_ONLY = re.compile(r"(?:Socket\s+)?" + _SOCK, re.I)
SOCKET_RE = re.compile(
    r"(?:Socket\s*:?\s*(?:\[\[\s*)?(?:Socket\s+)?" + _SOCK + r")"
    r"|(?:" + _SOCK + r")\s*(?:socket|插槽)", re.I)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "raw-fetched"))
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "entities.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("""CREATE TABLE IF NOT EXISTS hw_socket_page (
        page TEXT PRIMARY KEY, socket TEXT, sockets TEXT, count INTEGER, evidence TEXT,
        segments TEXT)""")
    if "segments" not in {r[1] for r in con.execute("PRAGMA table_info(hw_socket_page)")}:
        con.execute("ALTER TABLE hw_socket_page ADD COLUMN segments TEXT")
    con.execute("DELETE FROM hw_socket_page")

    written = 0
    # 一個世代的資訊框會把插槽分時段寫（Zen 5：Desktop=AM5、Server=SP5、HEDT=sTR5）。
    # 只取「最常見的那一個」會把 AM5 說成整個世代，所以要留分段。
    APOS = chr(39) * 3
    INFOBOX = re.compile(
        r"\|\s*sock\d+\s*=\s*" + APOS + r"?([A-Za-z][A-Za-z /-]*?)" + APOS +
        r"?\s*(.{0,90})$", re.M)
    for fn in sorted(os.listdir(args.docs)):
        if not fn.endswith(".wiki"):
            continue
        page = fn[:-5]
        text = open(os.path.join(args.docs, fn), encoding="utf-8", errors="replace").read()
        hits = Counter()
        first = {}
        segments = {}
        for m in INFOBOX.finditer(text):
            label = m.group(1).strip().lower()
            mm = SOCK_ONLY.search(m.group(2))
            if mm and label:
                segments[label] = re.sub(r"\s+", " ", mm.group(1)).strip()
        for m in SOCKET_RE.finditer(text):
            s = re.sub(r"\s+", " ", m.group(1) or m.group(2) or "").strip()
            # 正規化大小寫，但保留 LGA/AM 的寫法
            s = re.sub(r"^lga\s*", "LGA ", s, flags=re.I)
            s = re.sub(r"^(am|fm|sp|tr|swrx)", lambda mm: mm.group(1).upper(), s)
            hits[s] += 1
            first.setdefault(s, text[max(0, m.start() - 60):m.end() + 40].replace("\n", " "))
        if not hits:
            continue
        top, n = hits.most_common(1)[0]
        con.execute("INSERT OR REPLACE INTO hw_socket_page"
                    " (page,socket,sockets,count,evidence,segments)"
                    " VALUES (?,?,?,?,?,?)",
                    (page, top, json.dumps(list(hits), ensure_ascii=False), n,
                     first[top][-160:], json.dumps(segments, ensure_ascii=False)))
        written += 1
    con.commit()
    con.close()
    print(f"寫入 {written} 個頁面的插槽對照表")

    # 抽樣核對
    con = sqlite3.connect(args.db)
    for page, sock in con.execute(
            "select page,socket from hw_socket_page order by page limit 12"):
        print(f"   {page[:44]:44s} → {sock}")


if __name__ == "__main__":
    raise SystemExit(main())
