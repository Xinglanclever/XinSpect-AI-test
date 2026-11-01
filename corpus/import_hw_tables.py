#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把外部語料裡的 Wikipedia 規格表，抽成可以「按型號查」的資料列。

為什麼需要這一支：
  xlsx 那份資料庫只有 147 顆 CPU、161 張 GPU，問到 2010 年代的型號就查不到，
  只能回頭讓模型憑記憶講——那正是我們最不想發生的事。Wiki 的「List of … processors」
  這類頁面裡有上萬列規格，而且我們的語料裡已經有（tables.jsonl），只是還沒被拆成資料列。

為什麼不逐頁寫死欄位對應：
  這些表格是三層表頭 ＋ rowspan/colspan 混在一起的，逐頁硬對一定會在某幾頁錯欄。
  錯欄的規格比沒有規格更糟（「16 核心」被讀成「16 執行緒」）。所以這裡只做兩件有把握的事：
    1. 從列裡認出**型號**（用各家族型號的樣式去比對，這是可靠的）；
    2. 把整列原樣留著，回答時照抄，不重新解釋。
  代價是回答會長得像一列表格；好處是不可能把欄位講錯。

用法：
  python corpus/import_hw_tables.py --tables <corpus/jsonl/tables.jsonl>
"""

import argparse
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "entities.db")

# 只收「型號清單」類的頁面。chipsets／interface bit rates／CPUID 這種不是產品型號。
PAGE_OK = re.compile(
    r"(list of .*processor|list of .*graphics|list of .*gpu|table of .*processor"
    r"|^intel core$|^ryzen$|^xeon$|^geforce$|^radeon$|geforce .*series|radeon .*series"
    r"|intel graphics technology|arrow lake|panther lake)", re.I)
PAGE_BAD = re.compile(r"(chipset|interface bit|cpuid|comparison of intel chipsets)", re.I)

# 型號樣式。寧可漏收（不影響正確性），不要亂收（會把標題列當成型號）。
MODEL_PATTERNS = [
    r"\b(?:core\s+)?(?:i[3579]|ultra\s*[3579])[-\s]?\d{3,5}[a-z]{0,3}\b",
    r"\bryzen\s+(?:ai\s+)?\d\s*\d{4}[a-z]{0,2}\b",
    r"\bthreadripper(?:\s+pro)?\s+\d{4}[a-z]{0,3}\b",
    r"\bepyc\s+\d{4}[a-z]{0,4}\b",
    r"\bxeon\s+(?:[a-z]+\s+)?(?:[epwx]?-?\d{4,5}[a-z]{0,3}|[a-z0-9]{4,8})\b",
    r"\batom\s+[a-z]?\d{3,4}[a-z]{0,3}\b",
    r"\bceleron\s+[a-z]?\d{3,4}[a-z]{0,3}\b",
    r"\bpentium\s+(?:gold|silver|dual-core)?\s*[a-z]?\d{3,4}[a-z]{0,3}\b",
    r"\bathlon(?:\s+\d{2,4})?\s+[a-z]?\d{3,4}[a-z]{0,2}\b",
    r"\bphenom\s+[ivx]+\s*\d{4}[a-z]{0,3}\b",
    r"\bfx-\d{4}[a-z]{0,2}\b",
    r"\b(?:geforce\s+)?(?:gtx|rtx)\s*\d{3,4}\s*(?:ti|super|max-q)?\b",
    r"\bradeon\s+(?:rx\s*)?(?:vega\s*)?\d{3,4}[a-z]{0,2}\b",
    r"\barc\s+[ab]\d{3}[a-z]{0,2}\b",
    r"\b(?:quadro|tesla)\s+[a-z]?\d{3,4}[a-z]{0,3}\b",
]
MODEL_RE = re.compile("|".join("(?:%s)" % p for p in MODEL_PATTERNS), re.I)

GPU_PAGE = re.compile(r"(graphics|geforce|radeon|gpu|rtx|mali|adreno|arc\b)", re.I)


def norm(s):
    s = str(s or "").lower()
    s = re.sub(r"\[[^\]]*\]", " ", s)               # 吃掉 [12] 這種註腳
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    return re.sub(r"[^0-9a-z]+", "", s)


def clean_cell(s):
    s = str(s or "")
    s = re.sub(r"^\s*(?:[a-z-]+=\"[^\"]*\"\s*\|)+", "", s)   # class="…" rowspan="2" |
    s = re.sub(r"<ref[^>]*>.*?</ref>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", s)
    s = re.sub(r"\[\[([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", s)
    s = re.sub(r"\{\{[^}]*\}\}", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def find_model(cells):
    """回傳 (model, 起始 cell 索引)。取第一個「像型號」的最短字串。"""
    best = None
    for idx, c in enumerate(cells):
        for m in MODEL_RE.finditer(c or ""):
            cand = m.group(0).strip()
            if len(norm(cand)) < 4:
                continue
            if best is None or len(cand) < len(best[0]):
                best = (cand, idx)
    return best if best else (None, -1)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("""CREATE TABLE IF NOT EXISTS hw_model (
        row_id INTEGER PRIMARY KEY, model TEXT, norm TEXT, kind TEXT, page TEXT,
        table_index INTEGER, row_index INTEGER, cells TEXT, headers TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_hw_model_norm ON hw_model(norm)")
    con.execute("DELETE FROM hw_model")
    con.execute("DELETE FROM hw_alias WHERE kind IN ('wiki-cpu','wiki-gpu')")

    seen = set()
    n = 0
    by_page = {}
    with open(args.tables, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            page = rec.get("page") or ""
            if not PAGE_OK.search(page) or PAGE_BAD.search(page):
                continue
            headers = [clean_cell(h) for h in (rec.get("headers") or [])]
            rows = rec.get("rows") or []
            kind = "wiki-gpu" if GPU_PAGE.search(page) else "wiki-cpu"

            for ridx, raw in enumerate(rows):
                cells = [clean_cell(c) for c in raw]
                if not any(cells):
                    continue
                model, at = find_model(cells)
                if not model:
                    continue
                key = (kind, norm(model))
                if key in seen:            # 同一型號可能出現在好幾張重複的表
                    continue
                seen.add(key)

                n += 1
                con.execute(
                    "INSERT INTO hw_model (model,norm,kind,page,table_index,row_index,cells,headers) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (model, norm(model), kind, page, int(rec.get("table_index") or 0),
                     ridx, json.dumps(cells, ensure_ascii=False),
                     json.dumps(headers, ensure_ascii=False)))
                for alias in {norm(model), norm(model.replace("-", " ")),
                              norm(re.sub(r"^(?:core|geforce|radeon)\s+", "", model, flags=re.I))}:
                    if len(alias) >= 4:
                        con.execute(
                            "INSERT INTO hw_alias (alias,kind,row_id,model,vendor) VALUES (?,?,?,?,?)",
                            (alias, kind, n, model, page.split(" ")[0]))
                by_page[page] = by_page.get(page, 0) + 1

    con.commit()
    con.close()

    print(f"抽出 {n} 個型號（去重後），來自 {len(by_page)} 個頁面")
    for page, c in sorted(by_page.items(), key=lambda x: -x[1])[:15]:
        print(f"  {c:5d}  {page}")
    if len(by_page) > 15:
        print(f"  … 另有 {len(by_page) - 15} 個頁面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())