#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""吃「硬體規格庫」：Wikipedia 抽取庫（CC BY-SA）與 BuildCores v4 組件庫（CC BY-NC-SA）。

為什麼要分兩個 DB：
  授權不同。Wikipedia 那份 CC BY-SA 4.0，可以跟現有語料一起出貨；
  BuildCores 那份是 **CC BY-NC-SA 4.0**——禁商業使用、且要求相同方式分享。
  把它併進主 DB，等於主張我們整份資料庫要改採 NC-SA，這對要上線的產品是致命的。
  所以：主 DB（entities.db）只收 Wiki；組件庫寫到 entities-nc.db，
  由 `--nc-db` 指定，伺服器只有在明確開啟 `XINSPECT_ALLOW_NC=1` 時才載入。

資料長什麼樣：
  wiki jsonl 每筆：category / manufacturer / family / model / specs{} / extra{} / numeric{} /
                  source（頁名）/ source_url
  v4 jsonl   每筆：category / specs{…}（欄位已是整理過的，如 socket、chipset、wattage_w）
  兩個來源的 `specs` 都是「欄位→值」，所以直接變成 hw_model 的 headers/cells——
  回覆時可以逐欄標名（「cores 8；threads 16；tdp_w 105 W」），比照抄整列原文更清楚。

用法：
  python corpus/import_hw_library.py \
      --wiki-jsonl "<…>/out/hardware.jsonl" \
      --v4-dir      "<…>/out/v4" \
      --db          corpus/entities.db \
      --nc-db       corpus/entities-nc.db \
      --chunks      corpus/corpus-library.jsonl \
      --nc-chunks   corpus/corpus-v4.jsonl
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

WIKI_LICENSE = "CC-BY-SA-4.0"
NC_LICENSE = "CC-BY-NC-SA-4.0"

# 類別 → 型號庫的 kind
# 一律用 lib- 前綴：這樣才分得開「我們自己抓的 wiki 表格」（wiki-cpu／wiki-gpu）
# 與「使用者给的規格庫」（lib-cpu／lib-gpu），重跑本脚本只清 lib-* 不會踩到前者。
KIND = {
    "cpu": "lib-cpu", "gpu": "lib-gpu", "motherboard": "lib-chipset",
    "memory": "lib-memory", "ssd": "lib-ssd", "hdd": "lib-hdd",
    "npu": "lib-npu", "socket": "lib-socket", "fan": "lib-fan",
}
LABEL = {
    "lib-cpu": "CPU", "lib-gpu": "GPU", "lib-chipset": "晶片組",
    "lib-memory": "記憶體", "lib-ssd": "SSD", "lib-hdd": "HDD",
    "lib-npu": "NPU", "lib-socket": "插槽", "lib-fan": "風扇",
}

CAT_LABEL = {
    "motherboards": "主機板", "psu": "電源供應器", "cpu_coolers": "CPU 散熱器",
    "case_fans": "機殼風扇",
}

# 品牌字（不帶數字的家族名）才能安全地跟裸型號併成完整名稱：
#   Opteron ＋ 1210 → Opteron 1210 ✓
#   Ryzen 5000 ＋ 5800X → Ryzen 5000 5800X ✗（真名是 Ryzen 7 5800X，不能猜）
BARE_FAMILY = re.compile(
    r"^(?:opteron|phenom(?:\s+ii)?|athlon(?:\s+(?:64|x2|xp|ii))?|sempron|turion|duron"
    r"|celeron|pentium(?:\s+(?:d|m|4|iii|ii|pro|gold|silver))?|atom|xeon|epyc"
    r"|itanium|fx)$", re.I)

BARE_MODEL = re.compile(r"^\d{3,5}[a-z]{0,3}$", re.I)


def norm(s):
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = re.sub(r"[®™©]", "", s)
    return re.sub(r"[^0-9a-z]+", "", s)


def strip_prefix(s):
    return re.sub(r"^(?:intel|amd|nvidia|asus|msi|gigabyte|asrock|biostar|evga|"
                  r"colorful|maxsun|nzxt|thermaltake|corsair|cooler master|noctua"
                  r"|arctic|be quiet!?|deepcool|antec|seasonic|super flower|"
                  r"silverstone|fsp|lian li|fractal design|phanteks|sapphire"
                  r"|powercolor|xpg|zotac|palit|gainward|pny|inno3d|galax"
                  r"|geforce|radeon|core|ryzen)\s+", "", str(s or ""), flags=re.I)


def aliases(model, extra_bits=()):
    """全名、去品牌、數字字母尾端，以及呼叫端指定的補充寫法（variant／MPN）。"""
    out = set()
    cands = [model] + [b for b in extra_bits if b]
    for cand in cands:
        for form in (cand, strip_prefix(cand)):
            k = norm(form)
            if len(k) >= 4:
                out.add(k)
        words = [w for w in re.split(r"\s+", str(cand).strip()) if w]
        for i in range(1, len(words)):
            tail = norm(" ".join(words[i:]))
            if len(tail) >= 5 and re.search(r"\d", tail) and re.search(r"[a-z]", tail):
                out.add(tail)
    return sorted(out, key=len, reverse=True)


def full_model(rec):
    """組出可查的型號：裸數字配上不帶數字的品牌家族。"""
    m = (rec.get("model") or "").strip()
    fam = (rec.get("family") or "").strip()
    if m and BARE_MODEL.match(m) and fam and BARE_FAMILY.match(fam):
        composed = f"{fam} {m}"
        if norm(composed) != norm(m):
            return composed
    return m


def flat_specs(specs, extra=None):
    """specs 的鍵直接當表頭；extra 的原始欄位用 `原文:` 前綴保留（不丟資料）。"""
    headers, cells = [], []
    for k, v in (specs or {}).items():
        if v in (None, "", []):
            continue
        headers.append(str(k))
        cells.append(str(v) if not isinstance(v, (list, tuple)) else "; ".join(map(str, v)))
    for k, v in (extra or {}).items():
        if v in (None, "", []):
            continue
        headers.append(f"原文:{k}")
        cells.append(str(v) if not isinstance(v, (list, tuple)) else "; ".join(map(str, v)))
    return headers, cells


def ensure_columns(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(hw_model)")}
    for name in ("license", "url", "vendor"):
        if name not in cols:
            con.execute(f"ALTER TABLE hw_model ADD COLUMN {name} TEXT")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-jsonl", default="")
    ap.add_argument("--v4-dir", default="")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--nc-db", default=os.path.join(HERE, "entities-nc.db"))
    ap.add_argument("--chunks", default="")
    ap.add_argument("--nc-chunks", default="")
    args = ap.parse_args()

    if args.wiki_jsonl and not os.path.exists(args.wiki_jsonl):
        print("找不到 wiki jsonl：", args.wiki_jsonl)
        return 1
    if not (args.wiki_jsonl or args.v4_dir):
        print("至少要給 --wiki-jsonl 或 --v4-dir")
        return 1

    stats = {"wiki": 0, "wiki_rows": 0, "v4": 0, "v4_rows": 0}

    # ── Wikipedia 抽取庫（CC BY-SA）→ 主 DB ─────────────────────────
    if args.wiki_jsonl:
        con = sqlite3.connect(args.db)
        ensure_columns(con)
        con.execute("""CREATE TABLE IF NOT EXISTS hw_alias (
            alias TEXT NOT NULL, kind TEXT NOT NULL, row_id INTEGER NOT NULL,
            model TEXT, vendor TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_hw_alias ON hw_alias(alias)")
        # 只清掉「上一次由本脚本寫入」的列（kind 以 lib- 開頭；wiki-cpu/wiki-gpu 由
        # import_wiki_wikitext.py 管，不動它，避免兩支互踩）
        con.execute("DELETE FROM hw_model WHERE kind LIKE 'lib-%'")
        con.execute("DELETE FROM hw_alias WHERE kind LIKE 'lib-%'")

        row_id = con.execute("SELECT coalesce(max(row_id),0) FROM hw_model").fetchone()[0]
        chunks = []
        with open(args.wiki_jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                stats["wiki"] += 1
                cat = (rec.get("category") or "").lower()
                kind = KIND.get(cat)
                model = full_model(rec)
                if not kind or not model:
                    continue
                headers, cells = flat_specs(rec.get("specs"), rec.get("extra"))
                if len(cells) < 2:
                    continue
                row_id += 1
                stats["wiki_rows"] += 1
                source = rec.get("source") or ""
                url = rec.get("source_url") or ""
                con.execute(
                    "INSERT INTO hw_model (row_id,model,norm,kind,page,table_index,row_index,"
                    "cells,headers,license,url,vendor) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row_id, model, norm(strip_prefix(model)) or norm(model), kind,
                     f"lib:{source}" if source else "lib", 0, 0,
                     json.dumps(cells, ensure_ascii=False),
                     json.dumps(headers, ensure_ascii=False),
                     WIKI_LICENSE, url, rec.get("manufacturer") or ""))
                for a in aliases(model, (rec.get("family") or "",
                                         rec.get("manufacturer") or "")):
                    con.execute("INSERT INTO hw_alias (alias,kind,row_id,model,vendor)"
                                " VALUES (?,?,?,?,?)",
                                (a, kind, row_id, model, rec.get("manufacturer") or source))

                # 檢索 chunk：一筆一個，欄位逐項列出，方便回答時引用（不是給模型讀整頁）
                label = LABEL.get(kind, cat)
                head = " ".join(x for x in (rec.get("manufacturer"), model) if x)
                body = "；".join(f"{h} {c}" for h, c in zip(headers, cells) if c)
                chunks.append({
                    "id": f"lib.{row_id}",
                    "source": f"wiki.{source}" if source else f"lib.{cat}",
                    "lang": "zh-Hant",
                    "doc_title": f"{label}規格：{head}",
                    "heading_path": f"{label} · {head}",
                    "anchor": "", "line_start": "0", "line_end": "0", "paragraph": "",
                    "text": f"【{label}】{head}。{body}",
                    "url": url, "license": WIKI_LICENSE, "origin": "wiki-lib",
                })
        con.commit()
        con.close()
        if args.chunks:
            with open(args.chunks, "w", encoding="utf-8", newline="\n") as fh:
                for rec in chunks:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wikipedia 庫：讀 {stats['wiki']} 筆 → 寫入 {stats['wiki_rows']} 列"
              f"（{WIKI_LICENSE}）" + (f"、chunk {len(chunks)} 筆" if args.chunks else ""))

    # ── BuildCores v4（CC BY-NC-SA）→ 獨立 NC DB ────────────────────
    if args.v4_dir:
        if not os.path.isdir(args.v4_dir):
            print("找不到 v4 目錄：", args.v4_dir)
            return 1
        ncon = sqlite3.connect(args.nc_db)
        ncon.execute("""CREATE TABLE IF NOT EXISTS hw_component (
            comp_id INTEGER PRIMARY KEY, category TEXT, model TEXT, norm TEXT,
            manufacturer TEXT, specs TEXT, source_url TEXT, license TEXT)""")
        ncon.execute("""CREATE TABLE IF NOT EXISTS hw_comp_alias (
            alias TEXT, category TEXT, comp_id INTEGER, model TEXT)""")
        ncon.execute("CREATE INDEX IF NOT EXISTS idx_comp_alias ON hw_comp_alias(alias)")
        ncon.execute("CREATE INDEX IF NOT EXISTS idx_comp_norm ON hw_component(norm)")
        ncon.execute("DELETE FROM hw_component")
        ncon.execute("DELETE FROM hw_comp_alias")

        nc_chunks = []
        comp_id = 0
        for fn in sorted(os.listdir(args.v4_dir)):
            if not fn.endswith(".jsonl"):
                continue
            cat = fn[:-6]
            label = CAT_LABEL.get(cat, cat)
            with open(os.path.join(args.v4_dir, fn), encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    specs = rec.get("specs") or {}
                    model = (specs.get("model") or "").strip()
                    if not model:
                        continue
                    stats["v4"] += 1
                    comp_id += 1
                    stats["v4_rows"] += 1
                    lic = rec.get("license") or NC_LICENSE
                    ncon.execute(
                        "INSERT INTO hw_component (comp_id,category,model,norm,manufacturer,"
                        "specs,source_url,license) VALUES (?,?,?,?,?,?,?,?)",
                        (comp_id, cat, model, norm(model), specs.get("manufacturer") or "",
                         json.dumps(specs, ensure_ascii=False),
                         rec.get("source_url") or "", lic))
                    for a in aliases(model, (specs.get("variant") or "",
                                             specs.get("mpn") or "")):
                        ncon.execute(
                            "INSERT INTO hw_comp_alias (alias,category,comp_id,model)"
                            " VALUES (?,?,?,?)", (a, cat, comp_id, model))

                    body = "；".join(
                        f"{k} {v}" for k, v in specs.items()
                        if v not in (None, "", []) and k not in ("model",))
                    nc_chunks.append({
                        "id": f"v4.{comp_id}",
                        "source": f"buildcores.{cat}",
                        "lang": "zh-Hant",
                        "doc_title": f"{label}規格：{model}",
                        "heading_path": f"{label} · {model}",
                        "anchor": "", "line_start": "0", "line_end": "0", "paragraph": "",
                        "text": f"【{label}】{model}。{body}",
                        "url": rec.get("source_url") or "", "license": lic,
                        "origin": "v4-nc",
                    })
        ncon.commit()
        ncon.close()
        if args.nc_chunks:
            with open(args.nc_chunks, "w", encoding="utf-8", newline="\n") as fh:
                for rec in nc_chunks:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"v4 組件庫：讀 {stats['v4']} 筆 → {args.nc_db}"
              f"（{NC_LICENSE}；非商業，獨立 DB）"
              + (f"、chunk {len(nc_chunks)} 筆" if args.nc_chunks else ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
