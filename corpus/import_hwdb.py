#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「PC CPU／GPU 歷史型號資料庫」xlsx 吃進 entities.db 與檢索語料。

這一支解決的是兩種完全不同的問題：

  1. **查表**：`GTX 1080 Ti 幾 CUDA 核心`、`i7-6700K 的插槽`——型號是主鍵，
     答案是一個確定的事實。放進 `hw_cpu` / `hw_gpu` 表，加一張 alias 表處理
     「GeForce GTX 1080 Ti」與「GTX 1080 Ti」是同一張卡。
  2. **檢索**：`2006 到 2026 顯示卡換了幾代`、`哪些 OEM 型號容易和零售版搞混`
     ——這種要讀敘述的，切成 chunk 進 BM25。

兩邊都做，因為同一列資料對兩類問題都有用；只做其中一邊，另一半的問題就會掉回去猜。

xlsx 的來源可信度分級（表格裡本來就有「證據等級」欄）照抄進 metadata，
檢索回來的片段不會被誤當成官方規格。

用法：
  python corpus/import_hwdb.py --xlsx "C:/…/CPU_GPU_2006-2026_歷史型號資料庫_v0_3.xlsx"
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "entities.db")
DEFAULT_CHUNKS = os.path.join(HERE, "corpus-hwdb.jsonl")

TITLE = "CPU／GPU 歷史型號資料庫（2006–2026）"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS hw_cpu (
        row_id INTEGER PRIMARY KEY, year TEXT, vendor TEXT, model TEXT, family TEXT,
        segment TEXT, uarch TEXT, process TEXT, socket TEXT, cores TEXT, threads TEXT,
        tdp TEXT, status TEXT, tags TEXT, evidence TEXT, note TEXT, url TEXT, src TEXT,
        sheet TEXT, rowno INTEGER)""",
    """CREATE TABLE IF NOT EXISTS hw_gpu (
        row_id INTEGER PRIMARY KEY, year TEXT, vendor TEXT, model TEXT, family TEXT,
        segment TEXT, uarch TEXT, chip TEXT, vram TEXT, interface TEXT, status TEXT,
        tags TEXT, evidence TEXT, note TEXT, url TEXT, src TEXT, sheet TEXT, rowno INTEGER)""",
    """CREATE TABLE IF NOT EXISTS hw_oem (
        row_id INTEGER PRIMARY KEY, category TEXT, year TEXT, vendor TEXT, model TEXT,
        family TEXT, segment TEXT, arch TEXT, chip TEXT, special TEXT,
        evidence TEXT, note TEXT, url TEXT, src TEXT, sheet TEXT, rowno INTEGER)""",
    """CREATE TABLE IF NOT EXISTS hw_es (
        row_id INTEGER PRIMARY KEY, category TEXT, year TEXT, vendor TEXT, model TEXT,
        arch TEXT, part TEXT, state TEXT, evidence_type TEXT, evidence TEXT, note TEXT,
        url TEXT, src TEXT, sheet TEXT, rowno INTEGER)""",
    """CREATE TABLE IF NOT EXISTS hw_year (
        year TEXT PRIMARY KEY, intel TEXT, amd TEXT, gpu TEXT, note TEXT,
        intel_src TEXT, amd_src TEXT, gpu_src TEXT)""",
    """CREATE TABLE IF NOT EXISTS hw_source (
        id TEXT PRIMARY KEY, name TEXT, covers TEXT, trust TEXT, url TEXT, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS hw_alias (
        alias TEXT NOT NULL, kind TEXT NOT NULL, row_id INTEGER NOT NULL,
        model TEXT, vendor TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_hw_alias ON hw_alias(alias)",
    "CREATE INDEX IF NOT EXISTS idx_hw_cpu_model ON hw_cpu(model)",
    "CREATE INDEX IF NOT EXISTS idx_hw_gpu_model ON hw_gpu(model)",
]


def norm(s):
    """型號比對用的鍵：去掉空白、標點、商標符號，全小寫。"""
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = re.sub(r"[®™©]", "", s)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def aliases(model):
    """「GeForce GTX 1080 Ti」也要能用「GTX 1080 Ti」「1080 Ti」查到。

    只取**尾端連續詞**組成的別名：前綴（廠商、系列）是最常被省略的部分，
    而「1080」這種單一數字詞不放進去——顯示器解析度 1920×1080 會誤中。
    """
    words = [w for w in re.split(r"\s+", str(model or "").strip()) if w]
    out = set()
    for k in range(len(words)):
        tail = words[k:]
        if len(tail) < 2 and k > 0:
            continue
        a = norm(" ".join(tail))
        if len(a) >= 4:
            out.add(a)
    full = norm(model)
    if len(full) >= 4:
        out.add(full)
    return sorted(out, key=len, reverse=True)


def cell(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def clean_row(row):
    return [cell(v) for v in row]


def build_text(fields):
    """把「欄位=值」串成一句可讀的話，空欄位不佔位置。"""
    parts = []
    for label, value in fields:
        if value:
            parts.append(f"{label} {value}")
    return "；".join(parts)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--chunks", default=DEFAULT_CHUNKS)
    ap.add_argument("--keep", action="store_true", help="保留既有 hw_* 資料（預設先清空，重覆跑不會疊）")
    args = ap.parse_args()

    import openpyxl

    wb = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    need = ["CPU_型號索引", "GPU_型號索引", "OEM_特殊型號", "ES_QS_未公開", "年度_世代索引", "資料來源"]
    missing = [n for n in need if n not in wb.sheetnames]
    if missing:
        print("缺少工作表：" + "、".join(missing))
        return 1

    con = sqlite3.connect(args.db)
    for ddl in SCHEMA:
        con.execute(ddl)
    if not args.keep:
        for table in ("hw_cpu", "hw_gpu", "hw_oem", "hw_es", "hw_year", "hw_source", "hw_alias"):
            con.execute(f"DELETE FROM {table}")

    chunks = []
    counters = {}

    def add_chunk(source, heading, text, extra=None):
        cid = f"hwdb.{len(chunks):05d}"
        rec = {
            "id": cid,
            "source": source,
            "lang": "zh-Hant",
            "doc_title": TITLE,
            "heading_path": heading,
            "anchor": "",
            "line_start": "0",
            "line_end": "0",
            "paragraph": "",
            "text": text,
            "url": (extra or {}).get("url", ""),
            "license": "user-provided-compilation",
            "origin": "hwdb",
        }
        chunks.append(rec)

    # ── CPU ─────────────────────────────────────────────────────────
    ws = wb["CPU_型號索引"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    n = 0
    for i, raw in enumerate(rows, start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        year, vendor, model, family, segment, uarch, process, socket, cores, threads, tdp, \
            status, tags, evidence, note, url, src = (v + [""] * 17)[:17]
        if not model:
            continue
        n += 1
        con.execute(
            "INSERT INTO hw_cpu (row_id,year,vendor,model,family,segment,uarch,process,socket,"
            "cores,threads,tdp,status,tags,evidence,note,url,src,sheet,rowno) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, year, vendor, model, family, segment, uarch, process, socket, cores, threads,
             tdp, status, tags, evidence, note, url, src, ws.title, i))
        for a in aliases(model):
            con.execute("INSERT INTO hw_alias (alias,kind,row_id,model,vendor) VALUES (?,?,?,?,?)",
                        (a, "cpu", n, model, vendor))
        body = build_text([
            ("年份", year), ("廠商", vendor), ("型號", model), ("家族", family),
            ("市場區隔", segment), ("微架構／代號", uarch), ("製程", process),
            ("插槽／封裝", socket), ("核心", cores), ("執行緒", threads), ("TDP", tdp),
            ("狀態", status), ("標籤", tags), ("證據等級", evidence), ("備註", note),
            ("來源", src), ("URL", url),
        ])
        add_chunk(f"hwdb.cpu.{n}", f"CPU 型號索引 第 {i} 列（{vendor} {model}）",
                  f"【CPU 型號】{vendor} {model}。{body}", {"url": url})
    counters["cpu"] = n

    # ── GPU ─────────────────────────────────────────────────────────
    ws = wb["GPU_型號索引"]
    n = 0
    for i, raw in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        year, vendor, model, family, segment, uarch, chip, vram, interface, status, \
            tags, evidence, note, url, src = (v + [""] * 15)[:15]
        if not model:
            continue
        n += 1
        con.execute(
            "INSERT INTO hw_gpu (row_id,year,vendor,model,family,segment,uarch,chip,vram,"
            "interface,status,tags,evidence,note,url,src,sheet,rowno) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, year, vendor, model, family, segment, uarch, chip, vram, interface,
             status, tags, evidence, note, url, src, ws.title, i))
        for a in aliases(model):
            con.execute("INSERT INTO hw_alias (alias,kind,row_id,model,vendor) VALUES (?,?,?,?,?)",
                        (a, "gpu", n, model, vendor))
        body = build_text([
            ("年份", year), ("廠商", vendor), ("型號", model), ("家族", family),
            ("市場區隔", segment), ("微架構／代號", uarch), ("GPU 晶片", chip),
            ("顯存", vram), ("介面／封裝", interface), ("狀態", status), ("標籤", tags),
            ("證據等級", evidence), ("備註", note), ("來源", src), ("URL", url),
        ])
        add_chunk(f"hwdb.gpu.{n}", f"GPU 型號索引 第 {i} 列（{vendor} {model}）",
                  f"【GPU 型號】{vendor} {model}。{body}", {"url": url})
    counters["gpu"] = n

    # ── OEM ─────────────────────────────────────────────────────────
    ws = wb["OEM_特殊型號"]
    n = 0
    for i, raw in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        category, year, vendor, model, family, segment, arch, chip, special, \
            evidence, note, url, src = (v + [""] * 13)[:13]
        if not model:
            continue
        n += 1
        con.execute(
            "INSERT INTO hw_oem (row_id,category,year,vendor,model,family,segment,arch,chip,"
            "special,evidence,note,url,src,sheet,rowno) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, category, year, vendor, model, family, segment, arch, chip, special,
             evidence, note, url, src, ws.title, i))
        # OEM 型號加 "OEM" 後綴才是全名，單獨的型號容易和零售版混淆，所以不進 alias。
        body = build_text([
            ("類別", category), ("年份", year), ("廠商", vendor), ("型號", model),
            ("家族", family), ("市場／封裝", segment), ("架構", arch), ("晶片", chip),
            ("特殊分類", special), ("證據等級", evidence), ("說明", note),
            ("來源", src), ("URL", url),
        ])
        add_chunk(f"hwdb.oem.{n}", f"OEM 特殊型號 第 {i} 列（{vendor} {model}）",
                  f"【OEM 型號】{vendor} {model}。{body}", {"url": url})
    counters["oem"] = n

    # ── ES / QS ─────────────────────────────────────────────────────
    ws = wb["ES_QS_未公開"]
    n = 0
    for i, raw in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        category, year, vendor, model, arch, part, state, etype, evidence, note, url, src = \
            (v + [""] * 12)[:12]
        if not model:
            continue
        n += 1
        con.execute(
            "INSERT INTO hw_es (row_id,category,year,vendor,model,arch,part,state,evidence_type,"
            "evidence,note,url,src,sheet,rowno) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, category, year, vendor, model, arch, part, state, etype, evidence, note,
             url, src, ws.title, i))
        body = build_text([
            ("類別", category), ("年份／世代", year), ("廠商", vendor), ("型號", model),
            ("架構／平台", arch), ("ES碼／Part Number", part), ("已知狀態", state),
            ("證據類型", etype), ("證據等級", evidence), ("鑑別注意事項", note),
            ("來源", src), ("URL", url),
        ])
        add_chunk(f"hwdb.es.{n}", f"ES／QS 未公開 第 {i} 列（{vendor} {model}）",
                  f"【工程樣品／未公開型號】{vendor} {model}。{body}", {"url": url})
    counters["es"] = n

    # ── 年度世代索引 ────────────────────────────────────────────────
    ws = wb["年度_世代索引"]
    n = 0
    for i, raw in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        year, intel, amd, gpu, note, isrc, asrc, gsrc = (v + [""] * 8)[:8]
        if not year:
            continue
        n += 1
        con.execute(
            "INSERT OR REPLACE INTO hw_year (year,intel,amd,gpu,note,intel_src,amd_src,gpu_src) "
            "VALUES (?,?,?,?,?,?,?,?)", (year, intel, amd, gpu, note, isrc, asrc, gsrc))
        body = build_text([
            ("Intel", intel), ("AMD", amd), ("GPU（NVIDIA／AMD／Intel）", gpu),
            ("口徑", note), ("Intel 來源", isrc), ("AMD 來源", asrc), ("GPU 來源", gsrc),
        ])
        add_chunk(f"hwdb.year.{n}", f"年度世代索引 {year}",
                  f"【{year} 年 CPU／GPU 世代】{body}")
    counters["year"] = n

    # ── 資料來源 ────────────────────────────────────────────────────
    ws = wb["資料來源"]
    n = 0
    for i, raw in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        v = clean_row(raw)
        if not any(v):
            continue
        sid, name, covers, trust, url, note = (v + [""] * 6)[:6]
        if not sid:
            continue
        n += 1
        con.execute(
            "INSERT OR REPLACE INTO hw_source (id,name,covers,trust,url,note) VALUES (?,?,?,?,?,?)",
            (sid, name, covers, trust, url, note))
        body = build_text([
            ("名稱", name), ("涵蓋", covers), ("可信度", trust), ("URL", url), ("限制", note),
        ])
        add_chunk(f"hwdb.src.{n}", f"資料來源 {sid}",
                  f"【{sid} 資料來源】{body}", {"url": url})
    counters["source"] = n

    con.commit()
    con.close()
    wb.close()

    with open(args.chunks, "w", encoding="utf-8", newline="\n") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("匯入完成：", json.dumps(counters, ensure_ascii=False))
    print(f"  chunks : {len(chunks)} 筆 → {args.chunks}")
    print(f"  entities: {args.db}（hw_cpu / hw_gpu / hw_oem / hw_es / hw_year / hw_source / hw_alias）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())