#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Wikipedia 的晶片組清單頁（含被模板包住的 AM4／AM5 表格）匯入型號庫。

為什麼要另外一支：
  `List of AMD chipsets` 的 AM4／AM5 章節只有一行 `{{AM5 chipsets}}`——表格本體在模板裡，
  舊的匯入流程（把整頁交給表格解析）看不到，於是 A620／B650／X670／B850／X870E
  這批**現在買得到的晶片組全部缺席**；Intel 頁雖然是普通表格，但型號寫成 `'''Z790'''`，
  當年被當成「沒有型號欄」整張跳過。缺了這一塊，「B650 支援哪些世代／能不能超頻」
  就只能靠模型猜。

  Wikipedia 的維基表格有兩種寫法會讓簡單解析器失真：
    * 屬性沒有 `|` 分隔：`| colspan=9 {{yes}}`（整列合併成一個 Yes）
    * 標籤用模板包起來：`! rowspan=8 {{Vertical header|va=middle|Processor features support}}`
  所以這裡自己建格線（colspan 重複填、rowspan 帶到後續列），並把
  `{{yes}}／{{no}}／{{partial|X}}／{{abbr|GPP|…}}` 轉成人看得懂的字。

兩種表格形狀都吃：
  1) 一般表：一列一個晶片組（Intel 頁、AMD 舊世代章節）
  2) 轉置表：一欄一個晶片組、列是特性（AM4／AM5 模板）→ 轉成「一列一個晶片組」入庫

寫入 kind='wiki-chipset'（不與 lib-chipset 混）：
  兩支匯入腳本各自管自己的 kind 前綴，lib 那邊重跑時 `DELETE ... kind LIKE 'lib-%'`
  不會把這裡的資料一起清掉。查詢端用 kind IN ('lib-chipset','wiki-chipset') 讀。

用法：
  python corpus/import_chipsets.py --docs corpus/raw-fetched --db corpus/entities.db
"""

import argparse
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from import_wiki_wikitext import norm, split_tables  # noqa: E402

LICENSE = "CC-BY-SA-4.0"

# 晶片組名：A620／B650E／X870E／H610／Z790／W680／C741／TRX50／WRX90
CHIPSET = re.compile(r"^(?:amd|intel)?\s*([abxzqhwr]\d{3}[a-z]{0,2}|trx\d{2}|wrx\d{2})$", re.I)

PAGES = [
    ("List_of_AMD_chipsets", "lib:List of AMD chipsets", "AMD"),
    ("List_of_Intel_chipsets", "lib:List of Intel chipsets", "Intel"),
    ("Template_AM5_chipsets", "lib:Template AM5 chipsets", "AMD"),
    ("Template_AM4_chipsets", "lib:Template AM4 chipsets", "AMD"),
]

ATTR = re.compile(
    r"\b(rowspan|colspan|style|class|align|scope|valign|abbr|data-[a-z-]+)\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s|]*)",
    re.I)
BOOL_TPL = {"yes": "Yes", "no": "No", "unk": "?", "n/a": "N/A", "n": "No", "y": "Yes"}


def cell_text(raw):
    """一格 → 人看得懂的字。保留 Yes／No／Optional 這種語義，丟掉引用與模板雜訊。"""
    s = str(raw or "")
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", " ", s, flags=re.S)

    def tpl(m):
        body = m.group(1)
        name, _, rest = body.partition("|")
        name = name.strip().lower()
        if name in BOOL_TPL:
            return BOOL_TPL[name]
        if name.startswith(("efn", "ref", "citation", "sfn", "note")):
            return ""
        if name == "abbr":
            return rest.split("|")[0].strip()
        if rest:
            return rest.split("|")[-1].strip()
        return ""

    s = re.sub(r"\{\{([^{}]*)\}\}", tpl, s)
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)
    s = s.replace("{{", " ").replace("}}", " ")
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("{{", " ").replace("}}", " ")
    s = re.sub(r"''+", "", s)
    s = re.sub(r"^\s*va\s*=\s*\w+\s*\|", "", s)          # 模板殘骸
    s = re.sub(r"\s+", " ", s).strip(" |")
    return s


def parse_cell(raw):
    """(值, rowspan, colspan)。屬性可能沒有 `|` 分隔（`colspan=9 {{yes}}`）。"""
    s = str(raw or "").strip()
    rs = cs = 1
    while True:
        m = ATTR.match(s)
        if not m:
            break
        key = m.group(1).lower()
        val = m.group(2).strip("\"'")
        if key == "rowspan":
            rs = int(val) if val.isdigit() else 1
        elif key == "colspan":
            cs = int(val) if val.isdigit() else 1
        s = s[m.end():].lstrip()
        if s.startswith("|"):
            s = s[1:]
            break
    if s.startswith("|"):
        s = s[1:]
    return cell_text(s), rs, cs


def build_grid(block):
    """維基表格 → 二維格線（colspan 的值重複填滿，rowspan 的值帶到後續列）。

    注意：一格一行是常態（`! B650` 換行再接 `! B650E`），只有 `|-` 才開新列。
    把每個 `!`／`|` 行當成一列，AM5 模板就會被拆成 171 個「單格列」。
    """
    grid, pending, raw_cells = [], {}, []

    def flush():
        if not raw_cells:
            return
        row, col = {}, 0
        # 先把上一列 rowspan 留下來的值放進來，順便遞減
        for c, (v, left) in sorted(pending.items()):
            row[c] = v
            if left <= 1:
                del pending[c]
            else:
                pending[c] = (v, left - 1)
        for raw in list(raw_cells):
            while col in row:
                col += 1
            v, rs, cs = parse_cell(raw)
            for k in range(max(1, cs)):
                row[col + k] = v
                if rs > 1:
                    pending[col + k] = (v, rs - 1)
            col += max(1, cs)
        raw_cells.clear()
        if row:
            grid.append([row.get(i, "") for i in range(max(row) + 1)])

    for line in block:
        s = line.strip()
        if not s:
            continue
        if s.startswith("|-") or s.startswith("|}"):
            flush()
            continue
        if s.startswith("!"):
            raw_cells.extend(s[1:].split("!!"))
        elif s.startswith("|"):
            raw_cells.extend(s[1:].split("||"))
    flush()
    return grid


def name_of(cell):
    """一格 → 晶片組名（`A620 / A620A` 會回兩個）。"""
    c = re.sub(r"^\s*(?:amd|intel)\s+", "", str(cell or "").strip(), flags=re.I)
    out = []
    for part in re.split(r"\s*/\s*|\s*,\s*", c):
        m = CHIPSET.match(part.strip())
        if m:
            out.append(m.group(1).upper())
    return out


def emit(con, model, headers, cells, page, url, vendor, state):
    key = norm(model)
    if not key or key in state["seen"]:
        return 0
    have = con.execute(
        "SELECT 1 FROM hw_model WHERE norm=? AND kind IN ('lib-chipset','wiki-chipset')",
        (key,)).fetchone()
    if have:
        state["seen"].add(key)
        state["dup"] += 1
        return 0
    state["row_id"] += 1
    con.execute(
        "INSERT INTO hw_model (row_id,model,norm,kind,page,table_index,row_index,"
        "cells,headers,license,url,vendor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (state["row_id"], model, key, "wiki-chipset", page, state["ti"], 0,
         json.dumps(cells, ensure_ascii=False), json.dumps(headers, ensure_ascii=False),
         LICENSE, url, vendor))
    state["seen"].add(key)
    state["written"] += 1
    return 1


def import_transposed(grid, page, url, vendor, con, state):
    """一欄一個晶片組：轉成每列一個晶片組，欄名＝左側特性名。"""
    head_i = head_cols = None
    for i, row in enumerate(grid[:4]):
        cols = {}
        for j, c in enumerate(row):
            for nm in name_of(c):
                cols.setdefault(j, []).append(nm)
        if len(cols) >= 3:
            head_i, head_cols = i, cols
            break
    if head_i is None:
        return 0
    first = min(head_cols)
    featured = []
    for row in grid[head_i + 1:]:
        labels = [row[j] for j in range(min(first, len(row))) if row[j]]
        label = " / ".join(dict.fromkeys(labels))          # 去重、保留順序
        if not label:
            continue
        featured.append((label, row))
    added = 0
    for j, names in head_cols.items():
        heads, cells = [], []
        for label, row in featured:
            v = row[j] if j < len(row) else ""
            if v:
                heads.append(label[:70])
                cells.append(v[:120])
        if len(cells) >= 2:
            for name in names:
                added += emit(con, name, heads, cells, page, url, vendor, state)
    return added


def import_normal(grid, page, url, vendor, con, state):
    """一列一個晶片組：型號後面接的就是這顆晶片組的規格。"""
    if not grid:
        return 0
    header = grid[0] if sum(1 for c in grid[0] if not name_of(c)) >= 2 else [""] * len(grid[0])
    added = 0
    for row in grid[1:]:
        idx = None
        for i, c in enumerate(row[:3]):
            if name_of(c):
                idx = i
                break
        if idx is None:
            continue
        values = [v for v in row[idx + 1:] if v]
        heads = [header[i + 1] if header and i + 1 < len(header) else f"欄位 {i + 1}"
                 for i in range(len(values))]
        if heads and len(set(heads)) != len(heads):
            heads = [f"欄位 {i + 1}" for i in range(len(values))]
        for name in name_of(row[idx]):
            if len(values) >= 2:
                added += emit(con, name, heads, values, page, url, vendor, state)
        else:
            continue
    return added


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=os.path.join(HERE, "raw-fetched"))
    ap.add_argument("--db", default=os.path.join(HERE, "entities.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("DELETE FROM hw_model WHERE kind='wiki-chipset'")
    state = {"row_id": con.execute("SELECT coalesce(max(row_id),0) FROM hw_model").fetchone()[0],
             "written": 0, "dup": 0, "seen": set(), "ti": 0}

    for fname, page, vendor in PAGES:
        path = os.path.join(args.docs, fname + ".wiki")
        if not os.path.exists(path):
            print(f"  （略過，沒有檔案）{fname}")
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        title = fname.replace("Template_", "Template:").replace("_", " ")
        url = "https://en.wikipedia.org/wiki/" + title
        n_t = n_n = 0
        for ti, (block, _ln) in enumerate(split_tables(text)):
            state["ti"] = ti
            grid = build_grid(block)
            if len(grid) < 3:
                continue
            before = state["written"]
            import_transposed(grid, page, url, vendor, con, state)
            if state["written"] > before:
                n_t += 1
                continue
            before = state["written"]
            import_normal(grid, page, url, vendor, con, state)
            if state["written"] > before:
                n_n += 1
        print(f"  {fname}: 轉置表 {n_t} 張、一般表 {n_n} 張")

    con.commit()
    print(f"寫入 {state['written']} 個晶片組（略過已在庫的 {state['dup']} 個）")
    rows = con.execute("SELECT model, json_array_length(headers) FROM hw_model"
                       " WHERE kind='wiki-chipset' ORDER BY model").fetchall()
    sys.stdout.write("  " + " ".join(f"{m}({n})" for m, n in rows) + "\n")
    con.close()


if __name__ == "__main__":
    raise SystemExit(main())
