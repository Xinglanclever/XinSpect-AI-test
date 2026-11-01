#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把外部硬體語料庫（PCI/USB/OUI/PNP/CVE/Wiki）接進 XinSpect-Ai。

這份語料和曦覽文件是兩種東西，不能混在同一種索引裡：

  · **實體（entities）** 是「ID → 名稱」的確定映射。`10de` 就是 NVIDIA，沒有模糊空間。
    這種東西不該交給語言模型，該走 SQLite 精確查找——快、免費、不可能編錯。
  · **Wiki 文件與規格表** 是散文與表格，才需要 BM25 檢索。

授權也分成兩級，回答時必須標出來：
  · `ok-train` 類（pci.ids / usb.ids / Mesa / CISA KEV / endoflife.date）可訓練可散布。
  · `attr-required` 類（Wikipedia，CC-BY-SA-4.0）可訓練，但**產出要署名**。
  所以每個 chunk 都帶著自己的 license 欄位，服務那一側會把它帶到出處裡。

用法：
  python import_external.py --src "<corpus 目錄>"
"""

import argparse
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ENTITIES = os.path.join(HERE, "entities.db")
OUT_CHUNKS = os.path.join(HERE, "corpus-external.jsonl")

# ── wikitext 清洗 ────────────────────────────────────────────────────────
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_TEMPLATE_DEEP = re.compile(r"\{\{[^{}]*(\{\{[^{}]*\}\}[^{}]*)*\}\}")
RE_REF = re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S)
RE_TAG = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
RE_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
RE_EXT_LINK = re.compile(r"\[https?://[^\s\]]+\s*([^\]]*)\]")
RE_BOLD = re.compile(r"'{2,5}")
RE_HEADING = re.compile(r"^={2,6}\s*(.*?)\s*={2,6}\s*$", re.M)
RE_TABLE = re.compile(r"^\{\|.*?^\|\}", re.S | re.M)


def clean_wikitext(text):
    """把 wikitext 清成可讀文字。不清乾淨的檢索會被模板參數污染——
    `{{Infobox ... | title = ...}}` 這種片段會讓「什麼是 X」命中整段模板。"""
    t = text
    for _ in range(3):                      # 模板可能多層巢狀
        t = RE_TEMPLATE_DEEP.sub("", t)
        t = RE_TEMPLATE.sub("", t)
    t = RE_REF.sub("", t)
    t = RE_TABLE.sub("", t)
    t = RE_LINK.sub(r"\1", t)
    t = RE_EXT_LINK.sub(r"\1", t)
    t = RE_HEADING.sub(r"\n## \1\n", t)
    t = RE_TAG.sub("", t)
    t = RE_BOLD.sub("", t)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = t.replace("&mdash;", "—").replace("&ndash;", "–").replace("&quot;", '"')
    t = re.sub(r"\[\[Category:[^\]]*\]\]", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def table_to_text(rec, max_rows=40):
    """規格表轉成可檢索的文字。

    rowspan/colspan 的殘留（`style="..." rowspan="2"|`）在清洗時直接剝掉，
    但要保留「這個值屬於哪一欄」的線索，所以用「欄名：值」的成對寫法。
    """
    headers = [re.sub(r"^.*\|", "", h).strip() for h in (rec.get("headers") or [])]
    rows = []
    for row in (rec.get("rows") or [])[:max_rows]:
        cells = []
        for c in row:
            s = str(c)
            s = re.sub(r'style="[^"]*"', "", s)
            s = re.sub(r'(row|col)span="\d+"', "", s)
            s = s.strip().lstrip("|").strip()
            s = RE_LINK.sub(r"\1", s)
            s = RE_BOLD.sub("", s)
            cells.append(s)
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    page = rec.get("page", "")
    title = page.split("/")[-1].replace(".", " ").replace("_", " ")
    lines = [f"{title} 規格表（第 {rec.get('table_index', 0) + 1} 張）", ""]
    if any(headers):
        lines.append("欄位：" + " ｜ ".join(h for h in headers if h))
    for cells in rows:
        pairs = []
        for i, c in enumerate(cells):
            if not c:
                continue
            if i < len(headers) and headers[i]:
                pairs.append(f"{headers[i]}={c}")
            else:
                pairs.append(c)
        if pairs:
            lines.append("- " + "；".join(pairs))
    return "\n".join(lines)


# ── 實體：SQLite 精確查找 ────────────────────────────────────────────────
ENTITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS pci_vendor (id TEXT PRIMARY KEY, vendor TEXT, license TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS pci_device (
  vendor_id TEXT, device_id TEXT, vendor TEXT, name TEXT, license TEXT, source TEXT,
  PRIMARY KEY (vendor_id, device_id));
CREATE TABLE IF NOT EXISTS pci_subsystem (
  vendor_id TEXT, device_id TEXT, sub_id TEXT, name TEXT, license TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS pci_class (class_id TEXT PRIMARY KEY, name TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS usb_vendor (id TEXT PRIMARY KEY, vendor TEXT, license TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS usb_device (
  vendor_id TEXT, device_id TEXT, vendor TEXT, name TEXT, license TEXT, source TEXT,
  PRIMARY KEY (vendor_id, device_id));
CREATE TABLE IF NOT EXISTS usb_class (class_id TEXT PRIMARY KEY, name TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS pnp_id (id TEXT PRIMARY KEY, vendor TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS oui (prefix TEXT PRIMARY KEY, vendor TEXT, registry TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS drive_model (model TEXT, family TEXT, license TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS gpu_device (device_id TEXT, vendor_hint TEXT, codename TEXT, marketing TEXT, family TEXT, source TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS cve (cve TEXT PRIMARY KEY, vendor TEXT, product TEXT, name TEXT, date_added TEXT, ransomware TEXT, license TEXT);
CREATE TABLE IF NOT EXISTS software_cycle (
  product TEXT, cycle TEXT, release_date TEXT, eol TEXT, latest TEXT, lts TEXT, source TEXT, license TEXT,
  PRIMARY KEY (product, cycle));
"""

ENTITY_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_device_vendor ON pci_device(vendor_id);
CREATE INDEX IF NOT EXISTS ix_usb_device_vendor ON usb_device(vendor_id);
CREATE INDEX IF NOT EXISTS ix_oui_vendor ON oui(vendor);
CREATE INDEX IF NOT EXISTS ix_drive_model ON drive_model(model);
CREATE INDEX IF NOT EXISTS ix_gpu_device ON gpu_device(device_id);
CREATE INDEX IF NOT EXISTS ix_sw_product ON software_cycle(product);
"""


def norm_id(v):
    """ID 一律折成小寫、去掉 0x 前綴與空白。`0x10DE` 與 `10de` 是同一個東西。"""
    s = str(v or "").strip().lower()
    s = s.removeprefix("0x")
    return s


def import_entities(src, db_path):
    ent = os.path.join(src, "jsonl", "entities.jsonl")
    if not os.path.exists(ent):
        print(f"  ! 找不到 {ent}")
        return {}

    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.executescript(ENTITY_SCHEMA)

    counts = {}
    batch = {}

    def add(kind, row):
        batch.setdefault(kind, []).append(row)
        counts[kind] = counts.get(kind, 0) + 1

    with open(ent, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get("type")
            lic = d.get("license", "")
            s = d.get("source", "")
            if t == "pci_vendor":
                add("pci_vendor", (norm_id(d.get("vendor_id")), d.get("vendor"), lic, s))
            elif t == "pci_device":
                add("pci_device", (norm_id(d.get("vendor_id")), norm_id(d.get("device_id")),
                                   d.get("vendor"), d.get("name"), lic, s))
            elif t == "pci_subsystem":
                add("pci_subsystem", (norm_id(d.get("vendor_id")), norm_id(d.get("device_id")),
                                      norm_id(d.get("sub_id")), d.get("name"), lic, s))
            elif t == "pci_class":
                add("pci_class", (norm_id(d.get("class_id")), d.get("name"), lic))
            elif t == "usb_vendor":
                add("usb_vendor", (norm_id(d.get("vendor_id")), d.get("vendor"), lic, s))
            elif t == "usb_device":
                add("usb_device", (norm_id(d.get("vendor_id")), norm_id(d.get("device_id")),
                                   d.get("vendor"), d.get("name"), lic, s))
            elif t == "usb_class":
                add("usb_class", (norm_id(d.get("class_id")), d.get("name"), lic))
            elif t == "pnp_id":
                add("pnp_id", (str(d.get("id") or "").strip().upper(), d.get("vendor"), lic))
            elif t == "oui":
                add("oui", (str(d.get("mac_prefix") or "").strip().upper(), d.get("vendor"),
                            d.get("registry"), lic))
            elif t == "drive_model":
                add("drive_model", (d.get("model"), d.get("family"), lic, s))
            elif t == "gpu_device":
                add("gpu_device", (norm_id(d.get("device_id")), None, d.get("codename"),
                                   d.get("marketing_name"), d.get("family"), s, lic))
            elif t == "cve":
                add("cve", (d.get("cve"), d.get("vendor"), d.get("product"), d.get("name"),
                            d.get("date_added"), d.get("ransomware"), lic))
            elif t == "software_cycle":
                add("software_cycle", (d.get("product"), str(d.get("cycle")), d.get("release_date"),
                                       str(d.get("eol")), str(d.get("latest")), str(d.get("lts")), s, lic))

    # 每張表的欄位數都要與 INSERT 的 ? 個數一致；對不上時 SQLite 只會回一句
    # 「table X has N columns but M values were supplied」，不會說是哪一筆，
    # 所以這裡把 schema 與 SQL 放在一起看，不要憑記憶。
    sql = {
        "pci_vendor": "INSERT OR REPLACE INTO pci_vendor VALUES (?,?,?,?)",
        "pci_device": "INSERT OR REPLACE INTO pci_device VALUES (?,?,?,?,?,?)",
        "pci_subsystem": "INSERT OR REPLACE INTO pci_subsystem VALUES (?,?,?,?,?,?)",
        "pci_class": "INSERT OR REPLACE INTO pci_class VALUES (?,?,?)",
        "usb_vendor": "INSERT OR REPLACE INTO usb_vendor VALUES (?,?,?,?)",
        "usb_device": "INSERT OR REPLACE INTO usb_device VALUES (?,?,?,?,?,?)",
        "usb_class": "INSERT OR REPLACE INTO usb_class VALUES (?,?,?)",
        "pnp_id": "INSERT OR REPLACE INTO pnp_id VALUES (?,?,?)",
        "oui": "INSERT OR REPLACE INTO oui VALUES (?,?,?,?)",
        "drive_model": "INSERT OR REPLACE INTO drive_model VALUES (?,?,?,?)",
        "gpu_device": "INSERT OR REPLACE INTO gpu_device VALUES (?,?,?,?,?,?,?)",
        "cve": "INSERT OR REPLACE INTO cve VALUES (?,?,?,?,?,?,?)",
        "software_cycle": "INSERT OR REPLACE INTO software_cycle VALUES (?,?,?,?,?,?,?,?)",
    }
    for kind, rows in batch.items():
        con.executemany(sql[kind], rows)
    con.executescript(ENTITY_INDEXES)
    con.commit()
    con.close()
    return counts


# ── Wiki 文件與表格：切塊成可檢索語料 ────────────────────────────────────
def import_documents(src, out_path, sources_map):
    docs = os.path.join(src, "jsonl", "documents.jsonl")
    tables = os.path.join(src, "jsonl", "tables.jsonl")
    written = 0
    kept = 0
    dropped_short = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        if os.path.exists(docs):
            # 先把每個來源的段落收在一起，才能標出「這是第幾段」——
            # 引用時「第 24 段」比「行號 0」有用得多，而維基的段落本來就沒有行號。
            page_seen = {}
            with open(docs, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    text = clean_wikitext(d["text"])
                    if len(text) < 260:
                        dropped_short += 1
                        continue
                    src_id = d["source"]
                    label, page, lic = resolve_source(src_id, sources_map)
                    page_seen[src_id] = page_seen.get(src_id, 0) + 1
                    para = page_seen[src_id]
                    chunk = {
                        "id": d["id"],
                        "source": src_id,
                        "lang": "en",          # Wikipedia 抓的是 en.wikipedia
                        "doc_title": label,
                        "heading_path": f"{label}（Wikipedia）第 {para} 段",
                        "anchor": "",
                        "line_start": 0,
                        "line_end": 0,
                        "paragraph": para,
                        "text": text,
                        "url": wiki_url(page),
                        "license": d.get("license") or lic,
                        "origin": "external",
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    written += 1

        if os.path.exists(tables):
            with open(tables, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    text = table_to_text(rec)
                    if len(text) < 200:
                        continue
                    src_id = rec.get("page", "")
                    label, page, lic = resolve_source(src_id, sources_map)
                    idx = rec.get("table_index", 0) + 1
                    chunk = {
                        "id": rec["id"],
                        "source": src_id,
                        "lang": "en",
                        "doc_title": f"{label} 規格表",
                        "heading_path": f"{label}（Wikipedia）規格表 {idx}",
                        "anchor": "",
                        "line_start": 0,
                        "line_end": 0,
                        "table_index": idx,
                        "text": text,
                        "url": wiki_url(page),
                        "license": rec.get("license") or lic,
                        "origin": "external",
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    written += 1
                    kept += 1

    return written, kept, dropped_short


def load_source_titles(src):
    """sources.jsonl 裡有人類可讀的標題與真實的維基頁名。

    為什麼要有這一份：`documents.jsonl` 的 `source` 是內部 id（`wiki.pcie`），
    直接拿它組 URL 會得到 `https://en.wikipedia.org/wiki/wiki.pcie`——那個頁面不存在。
    真實頁名在 `title_wiki`，只有在 sources.jsonl 裡。
    """
    m = {}
    p = os.path.join(src, "sources.jsonl")
    if not os.path.exists(p):
        return m
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            m[d["id"]] = {
                "label": d.get("title_wiki") or d.get("title") or d["id"],
                "page": d.get("title_wiki") or "",
                "url": d.get("url") or "",
                "lic": d.get("lic") or "",
            }
    return m


def wiki_url(page):
    """維基頁名 → 可點的網址。頁名裡的空白要換成底線並做百分比編碼。"""
    if not page:
        return ""
    from urllib.parse import quote
    return "https://en.wikipedia.org/wiki/" + quote(page.replace(" ", "_"))


def resolve_source(src_id, sources_map):
    """把一個來源 id 解成 (顯示名稱, 維基頁名, 授權)。

    為什麼要這一支：分類型的來源（`wiki.cat.amd.cpu`）抓回來的是**底下每一頁**的內容，
    文件記錄長成 `wiki.cat.amd.cpu/Alchemy (processor)`——精確比對 sources.jsonl
    會找不到，於是 URL 變空、標題變成一串內部 id。這種 id 的真實頁名在斜線後面。
    """
    meta = sources_map.get(src_id)
    if meta:
        page = meta.get("page") or meta.get("label") or ""
        # 分類來源的 title_wiki 是「Category:XXX」，那不是條目頁；單一文檔才用它。
        if page.startswith("Category:"):
            page = ""
        return meta.get("label") or src_id, page, meta.get("lic") or ""

    if "/" in src_id:
        parent, child = src_id.split("/", 1)
        pmeta = sources_map.get(parent, {})
        return child, child, pmeta.get("lic") or ""

    # 有些來源的 `page` 直接就是條目名（不是內部 id），例如規格表裡的 "Intel Core"。
    # 這種沒有斜線、也不在 sources.jsonl 裡；它本身就是頁名，直接用。
    # 但 `wiki.xxx` 這種內部 id 不能當頁名——那會組出 wiki.pcie 這種不存在的網址，
    # 那種情況寧可留空（前端就不顯示連結），也不要給一個假連結。
    if not src_id.startswith("wiki."):
        return src_id, src_id, ""

    return src_id, "", ""


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="外部語料庫的 corpus 目錄")
    ap.add_argument("--entities", default=OUT_ENTITIES)
    ap.add_argument("--chunks", default=OUT_CHUNKS)
    args = ap.parse_args()

    print(f"讀取 {args.src}")
    counts = import_entities(args.src, args.entities)
    print(f"  實體 -> {args.entities}")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"    {k:16s} {v:>7,}")

    titles = load_source_titles(args.src)
    n, tables, short = import_documents(args.src, args.chunks, titles)
    print(f"  Wiki 切塊 -> {args.chunks}")
    print(f"    總計 {n:,}（其中規格表 {tables:,}）；過短略過 {short:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
