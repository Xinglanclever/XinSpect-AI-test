#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把型號庫裡「查得到的事實」寫成三語 SFT 問答，讓模型學會這批硬體題的答法。

為什麼不直接把 Wikipedia 原文丟進訓練：
  原文是 CC BY-SA，貼進權重等於把別人的文字做成衍生品。這裡只取**事實欄位**
  （核心數、時脈、插槽、TDP…——事實本身不受著作權保護），句子一律用我們自己的模板寫，
  出處以〔來源：頁名〕的形式附在答案後面，既不抄原文，也保留可回查性。

  另一個理由是形狀：伺服器回答硬體題時是「照抄那一列、附出處、查不到就說查不到」，
  而 1.7B 預設會「大概 8 核」。SFT 要教的就是這個形狀，不是要它背事實。

產出三類樣本（每類都三語）：
  1. 欄位問答：型號＋欄位 → 值＋出處（只收欄位齊、型號具體的列）
  2. 拒答與邊界：查不到／行動版 BGA／來源沒有這一欄／不能讀使用者電腦
  3. 出處紀律：從曦覽文件抽一段，問一個那段**沒寫**的細節，答「文件沒有寫到 X」

用法：
  python tools/make_hw_sft.py --db corpus/entities.db --out train/sft-hw.jsonl
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

SYSTEM = {
    "zh-Hant": "你是 XinSpect-Ai，曦覽官網的硬體與文件助手。硬體規格要照型號庫的原文回答並附來源；"
               "文件問題只依〔資料片段〕回答並附〔檔名:行號〕。查不到就說查不到，不要用常識補。"
               "你不能讀使用者的電腦。",
    "zh-Hans": "你是 XinSpect-Ai，曦览官网的硬件与文档助手。硬件规格要照型号库的原文回答并附来源；"
               "文档问题只依〔资料片段〕回答并附〔文件名:行号〕。查不到就说查不到，不要用常识补。"
               "你不能读用户的电脑。",
    "en": "You are XinSpect-Ai, the XinSpect hardware and documentation assistant. For hardware "
          "specs, quote the model database and cite the source; for documentation questions, answer "
          "only from the [passages] and cite [file:line]. If the data is not there, say so instead "
          "of filling it in from memory. You cannot read the user's computer.",
}

# ── 欄位正規化 ─────────────────────────────────────────────────────────
# 表格的欄名每頁都不一樣（`| TDP`、`原文:TDP (watts)`、`tdp_w`），先洗成同一組鍵。
FIELD_PATTERNS = [
    ("cores", r"^cores?$|^cores \(threads\)$|core count|核心數|处理器内核"),
    ("threads", r"^threads?$|thread count"),
    ("base_clock", r"base[_ ]?clock|clock rate|^clock( \(ghz\))?$|^clock speed$"),
    ("boost_clock", r"boost"),
    ("l3", r"^l3"),
    ("l2", r"^l2"),
    ("tdp", r"^tdp"),
    ("socket", r"^socket"),
    ("process", r"process|製程|制程"),
    ("release", r"release date|launch date|^launch$|^released?$"),
    ("price", r"price|建議售價"),
    ("shaders", r"shaders|stream processors|cuda cores?|統一著色器"),
    ("vram", r"(memory.{0,24}size|frame ?buffer|^vram$|^memory$)"),
    ("mem_bus", r"(memory.{0,24}bus ?width|^memory bus|bus width)"),
    ("mem_type", r"(memory.{0,24}(type|standard)|^memory type|dram type)"),
    ("igpu", r"igpu|integrated graphics|graphics"),
]

FIELD_ALIAS = {}
for _key, _pat in FIELD_PATTERNS:
    FIELD_ALIAS[_key] = re.compile(_pat, re.I)


def canon_field(name):
    """欄名 → 正規鍵。認不出來回 None（這種欄位不進訓練：欄義不明的值是用猜的）。"""
    n = re.sub(r"^原文\s*:\s*", "", str(name or "")).strip()
    n = re.sub(r"^[|\s]+", "", n)
    n = re.sub(r"\s+", " ", n).lower()
    for key, pat in FIELD_ALIAS.items():
        if pat.search(n):
            return key, unit_of(n)
    return None, ""


UNIT_IN_NAME = [
    (r"(_gb\b|\(gb\)|gb$)", " GB"),
    (r"(_mb\b|\(mb\)|mb$)", " MB"),
    (r"(_ghz\b|\(ghz\))", " GHz"),
    (r"(_mhz\b|\(mhz\))", " MHz"),
    (r"(_w\b|\(watts?\)|\bw$)", " W"),
    (r"(_nm\b|\(nm\))", " nm"),
    (r"(_bits?|\(bit\)|bit$)", " bit"),
    (r"(_gbs\b|tb/s)", " GB/s"),
    (r"(_mm2|\(mm 2\))", " mm²"),
]


def unit_of(header):
    """`memory_size_gb` 這種欄名把單位寫在名字裡，值就只剩純數字——要補回單位才能讀。"""
    for pat, unit in UNIT_IN_NAME:
        if re.search(pat, header, re.I):
            return unit
    return ""


def tidy(v):
    """值 → 一句話。太長、帶模板殘骸、或只是 Yes／No 的值不收（那類要靠欄名語義）。"""
    s = re.sub(r"\s+", " ", str(v or "")).strip(" ;|")
    s = re.sub(r"^原文\s*:\s*", "", s)
    s = s.replace("—N/a", "").replace("N/a", "").strip()
    if not s or len(s) > 60:
        return ""
    if re.fullmatch(r"(yes|no|y|n|n/a|-|—|\?|none)", s, re.I):
        return ""
    if re.search(r"[{}\[\]]", s):
        return ""
    return s


# ── 值必須過得了這個欄位「物理上該長什麼樣」──────────────────────────
# 上游的抽取庫偶爾會整欄錯位（H200 那一列的 `shaders` 值是 `9.0`——那是 PCIe 世代，
# `process_nm` 是 `835`）。欄名與長度都對得上，所以「對齊」檢查救不了；
# 但規格值有物理範圍，超出範圍就不是這一欄的東西。訓練資料裡留這種值，
# 等於教模型把 PCIe 世代當核心數講——比不訓練更糟。
SOCKET_VALUE = re.compile(
    r"^\s*(?:(?:socket|lga|f|sp|tr)\s?)?"
    r"(lga\s?\d{3,4}(?:-\d)?(?:\s?\(v\d\))?|am\d\+?|fm\d\+?|s(?:wr)?x?\s?\d{1,4}"
    r"|sp\d|tr\d|kg\d{3}|bga\d{3,4}|sxm\d)\s*$", re.I)


def _num(s):
    m = re.search(r"(\d+(?:\.\d+)?)", str(s or "").replace(",", ""))
    return float(m.group(1)) if m else None


def _int(s):
    v = _num(s)
    return int(v) if v is not None else None


VALID = {
    "cores": lambda v: 1 <= (_int(re.sub(r"[（(].*", "", str(v))) or 0) <= 1024,
    "threads": lambda v: 1 <= (_int(v) or 0) <= 4096,
    "base_clock": lambda v: v is not None and 0.3 <= (_num(v) or 0) <= 7.0
                            and not re.search(r"\bmhz\b", v, re.I),
    "boost_clock": lambda v: v is not None and 0.3 <= (_num(v) or 0) <= 7.0
                             and not re.search(r"\bmhz\b", v, re.I),
    "l2": lambda v: re.search(r"\d", v) and bool(re.search(r"(kb|mb|k b|m b|k$|mb$)", v, re.I)),
    "l3": lambda v: re.search(r"\d", v) and bool(re.search(r"(kb|mb|k b|m b|mb$)", v, re.I)),
    "tdp": lambda v: 3 <= (_num(re.sub(r"[（(].*", "", str(v))) or 0) <= 1500,
    "socket": lambda v: bool(SOCKET_VALUE.match(v)),
    "process": lambda v: 2 <= (_num(re.sub(r"[（(].*", "", str(v))) or 0) <= 20,
    "release": lambda v: bool(re.search(r"(19|20)\d{2}", v)),
    "price": lambda v: bool(re.search(r"(\$\s?\d|\d+\s?(usd|美元))", v, re.I)),
    "shaders": lambda v: 128 <= (_int(v) or 0) <= 524288,
    "vram": lambda v: re.search(r"\d", v) and bool(re.search(r"(gb|mb|gddr|hbm|ddr)", v, re.I)),
    "mem_bus": lambda v: 32 <= (_int(v) or 0) <= 8192,
}


def good(key, raw, unit=""):
    v = tidy(raw)
    if not v:
        return ""
    if unit and re.fullmatch(r"\d+(?:\.\d+)?", v):
        v = v + unit
    try:
        return v if VALID.get(key, lambda _: True)(v) else ""
    except Exception:
        return ""


# 型號名裡不該出現的東西：iGPU 描述、卡型、產品線後綴。
# 「Celeron G1830 HD Graphics (10 EUs)」這種名字是上游把內显型號拼在後面，
# 留著問「Celeron G1830 HD Graphics (10 EUs) 有幾核心」會教模型用錯名字。
MODEL_NOISE = re.compile(
    r"(hd graphics|uhd graphics|iris|radeon graphics|vega \(|geforce graphics"
    r"|\(PCIe card\)|\(SXM card\)|accelerator|processor with|\d+ EUs?\)|\(6th gen)", re.I)


def clean_model(m):
    s = re.sub(r"\s+", " ", str(m or "")).strip()
    if MODEL_NOISE.search(s):
        return ""
    if not re.search(r"\d", s) or len(s) > 40:
        return ""
    if re.search(r"(series|family|list of|chipset|socket|microarchitecture)", s, re.I):
        return ""
    return s


# ── 問題與答案模板（三語）────────────────────────────────────────────
Q = {
    "cores": {"zh-Hant": "{m} 有幾核心、幾執行緒？", "zh-Hans": "{m} 有几个核心、几个线程？",
              "en": "How many cores and threads does the {m} have?"},
    "base_clock": {"zh-Hant": "{m} 的基礎時脈與加速時脈是多少？",
                   "zh-Hans": "{m} 的基础频率和加速频率是多少？",
                   "en": "What are the base and boost clocks of the {m}?"},
    "l3": {"zh-Hant": "{m} 的 L3 快取有多大？", "zh-Hans": "{m} 的 L3 缓存有多大？",
           "en": "How large is the L3 cache on the {m}?"},
    "tdp": {"zh-Hant": "{m} 的 TDP 是多少？", "zh-Hans": "{m} 的 TDP 是多少？",
            "en": "What is the TDP of the {m}?"},
    "socket": {"zh-Hant": "{m} 用哪個插槽？", "zh-Hans": "{m} 用哪个插槽？",
               "en": "Which socket does the {m} use?"},
    "release": {"zh-Hant": "{m} 什麼時候推出？", "zh-Hans": "{m} 什么时候推出？",
                "en": "When was the {m} released?"},
    "price": {"zh-Hant": "{m} 發表時的建議售價是多少？", "zh-Hans": "{m} 发布时的建议售价是多少？",
              "en": "What was the launch price of the {m}?"},
    "vram": {"zh-Hant": "{m} 的記憶體容量與匯流排寬度是多少？",
             "zh-Hans": "{m} 的显存容量和位宽是多少？",
             "en": "What are the memory size and bus width of the {m}?"},
    "shaders": {"zh-Hant": "{m} 有幾個串流處理器？", "zh-Hans": "{m} 有几个流处理器？",
                "en": "How many stream processors does the {m} have?"},
    "process": {"zh-Hant": "{m} 用什麼製程？", "zh-Hans": "{m} 用什么制程？",
                "en": "What process node is the {m} built on?"},
}

LBL = {
    "zh-Hant": {"cores": "核心", "threads": "執行緒", "base": "基礎時脈", "boost": "加速時脈",
                "l3": "L3 快取", "l2": "L2 快取", "tdp": "TDP", "socket": "插槽",
                "process": "製程", "release": "推出時間", "price": "建議售價",
                "vram": "記憶體", "bus": "匯流排寬度", "shaders": "串流處理器"},
    "zh-Hans": {"cores": "核心", "threads": "线程", "base": "基础频率", "boost": "加速频率",
                "l3": "L3 缓存", "l2": "L2 缓存", "tdp": "TDP", "socket": "插槽",
                "process": "制程", "release": "发布时间", "price": "建议售价",
                "vram": "显存", "bus": "位宽", "shaders": "流处理器"},
    "en": {"cores": "cores", "threads": "threads", "base": "base clock", "boost": "boost clock",
           "l3": "L3 cache", "l2": "L2 cache", "tdp": "TDP", "socket": "socket",
           "process": "process node", "release": "release date", "price": "launch price",
           "vram": "memory", "bus": "bus width", "shaders": "stream processors"},
}

CITE = {"zh-Hant": "〔來源：{p}〕", "zh-Hans": "〔来源：{p}〕", "en": "[source: {p}]"}
NOT_FOUND = {
    "zh-Hant": "型號庫裡查不到 {m} 的這一項，我不猜。",
    "zh-Hans": "型号库里查不到 {m} 的这一项，我不猜。",
    "en": "The model database does not have this field for the {m}; I will not guess.",
}


def row_fields(headers, cells):
    """一列 → {正規鍵: 值}，只留過得了驗證的欄位。

    欄數不對的列整列跳過：wiki 表格的表頭跟值常常錯位（表頭 10 格、值 12 格），
    zip 下去會把「Cores (threads)」對到型號、「TDP」對到 4.5（其實是時脈）。
    這種列拿去訓練或回答，錯得很有自信，比不回答還糟。
    """
    if len(headers) != len(cells):
        return {}
    d = {}
    for h, c in zip(headers, cells):
        key, unit = canon_field(h)
        if not key:
            continue
        v = good(key, c, unit)
        if not v:
            continue
        if key not in d or len(d[key]) > len(v):
            d[key] = v
    return d


def cpu_rows(con, kinds, limit):
    """挑「型號具體、欄位過驗證」的 CPU 列：先取欄位多的，再依型號去重。"""
    rows, seen = [], set()
    for kind in kinds:
        for model, headers, cells, page in con.execute(
                "SELECT model, headers, cells, page FROM hw_model WHERE kind=?", (kind,)):
            m = clean_model(model)
            if not m:
                continue
            key = re.sub(r"[^0-9a-z]+", "", m.lower())
            if key in seen:
                continue
            f = row_fields(json.loads(headers or "[]"), json.loads(cells or "[]"))
            if len([k for k in ("cores", "base_clock", "l3", "socket", "tdp") if k in f]) < 2:
                continue
            seen.add(key)
            rows.append((len(f), m, f, page))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return rows[:limit]


def gpu_rows(con, kinds, limit):
    rows, seen = [], set()
    for kind in kinds:
        for model, headers, cells, page in con.execute(
                "SELECT model, headers, cells, page FROM hw_model WHERE kind=?", (kind,)):
            m = clean_model(model)
            if not m:
                continue
            key = re.sub(r"[^0-9a-z]+", "", m.lower())
            if key in seen:
                continue
            f = row_fields(json.loads(headers or "[]"), json.loads(cells or "[]"))
            if len([k for k in ("shaders", "vram", "mem_bus", "tdp") if k in f]) < 2:
                continue
            seen.add(key)
            rows.append((len(f), m, f, page))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return rows[:limit]


def tidy_support(v):
    """晶片組支援欄的值：Yes／No／Varies 本身就是要留的答案（tidy 會把它們當空值丟掉）。"""
    s = re.sub(r"\s+", " ", str(v or "")).strip(" ;|.,")
    if not s or len(s) > 40:
        return ""
    if re.fullmatch(r"(yes|no|varies|partial|optional|n/a|—|-|\?)", s, re.I):
        return s
    return s if re.search(r"(yes|no|varies|partial|optional)", s, re.I) else ""


def chipset_rows(con, limit):
    """晶片組：把 CPU 支援／PCIe 通道／SATA 這種「一整列都在講同一件事」的資料寫成問答。"""
    out = []
    for model, headers, cells, page in con.execute(
            "SELECT model, headers, cells, page FROM hw_model"
            " WHERE kind IN ('lib-chipset','wiki-chipset')"):
        hd = [re.sub(r"^原文\s*:\s*", "", str(h or "")).strip() for h in json.loads(headers or "[]")]
        cc = json.loads(cells or "[]")
        sup = {}
        for h, c in zip(hd, cc):
            if "CPU support" in h:
                gen = h.split("CPU support")[-1].strip(" /:")
                gen = re.sub(r"^.*/\s*", "", gen) if "/" in gen else gen
                v = tidy_support(c)
                if gen and v:
                    sup[gen] = v
        pcie = {re.sub(r"^.*/\s*", "", h): tidy(c) for h, c in zip(hd, cc)
                if re.search(r"PCIe (lane|5\.0|4\.0)|PCIe support", h, re.I) and tidy(c)}
        sata = next((tidy(c) for h, c in zip(hd, cc) if re.search(r"^SATA", h, re.I) and tidy(c)), "")
        if len(sup) >= 2 or len(pcie) >= 2:
            out.append((model, sup, pcie, sata, page))
    return out[:limit]


def make_example(lang, q, a):
    return {"messages": [{"role": "system", "content": SYSTEM[lang]},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def consistent(model, f, agree):
    """只留「多來源都同意」的欄位（單來源的型號就原樣保留，沒有東西可比）。"""
    key = re.sub(r"[^0-9a-z]+", "", str(model or "").lower())
    ok = agree.get(key) or {}
    out = {}
    for k, v in f.items():
        nv = re.sub(r"\s+", "", v)
        if k in ok and ok[k] != nv:
            continue
        out[k] = v
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "corpus", "entities.db"))
    ap.add_argument("--out", default=os.path.join(HERE, os.pardir, "train", "sft-hw.jsonl"))
    ap.add_argument("--cpus", type=int, default=260)
    ap.add_argument("--gpus", type=int, default=160)
    ap.add_argument("--chipsets", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    langs = ("zh-Hant", "zh-Hans", "en")
    out, stats = [], Counter()

    # ── 1) CPU 欄位問答 ────────────────────────────────────────────────
    # 每顆型號只取 2 題（輪流換欄位），避免同一欄位灌滿訓練集——
    # 訓練集的比例會變成模型的偏好：全都在問 TDP，模型就會把所有問題當成 TDP。
    PRIORITY = ["cores", "socket", "tdp", "l3", "base_clock", "release", "price"]
    for idx, (_n, model, raw_f, page) in enumerate(cpu_rows(con, ("lib-cpu", "wiki-cpu"), args.cpus)):
        f = raw_f
        avail = [k for k in PRIORITY if k in f]
        start = (idx % max(1, len(avail) - 1)) if len(avail) > 2 else 0
        picks = (avail[start:] + avail[:start])[:2]
        for key in picks:
            v1 = f[key]
            v2 = f.get("boost_clock", "") if key == "base_clock" else f.get("threads", "")
            for lang in langs:
                L = LBL[lang]
                if key == "cores":
                    m = re.match(r"(\d+)\s*[（(]\s*(\d+)\s*[)）]", v1)
                    cores = m.group(1) if m else v1
                    v2 = m.group(2) if m else (v2 if (v2 or "").isdigit() else "")
                    a = (f"{cores} {L['cores']}" if lang == "en" else f"{cores} 個{L['cores']}")
                    if v2:
                        a += (f" / {v2} {L['threads']}" if lang == "en"
                              else f"、{v2} {L['threads']}")
                    a += f"\n{CITE[lang].format(p=page)}"
                elif key == "base_clock":
                    parts = [f"{L['base']} {v1}"]
                    if v2:
                        parts.append(f"{L['boost']} {v2}")
                    a = "、".join(parts) + f"\n{CITE[lang].format(p=page)}"
                else:
                    a = f"{L.get(key, key)} {v1}\n{CITE[lang].format(p=page)}"
                out.append(make_example(lang, Q[key][lang].format(m=model), a))
                stats[f"cpu-{key}"] += 1

    # ── 2) GPU 欄位問答 ────────────────────────────────────────────────
    for idx, (_n, model, raw_f, page) in enumerate(gpu_rows(con, ("lib-gpu", "wiki-gpu"), args.gpus)):
        f = raw_f
        avail = [k for k in ("shaders", "vram", "tdp", "price", "release") if k in f]
        picks = (avail[idx % len(avail):] + avail[:idx % len(avail)])[:2] if avail else []
        for key in picks:
            for lang in langs:
                L = LBL[lang]
                if key == "vram":
                    v = f[key] + (f" / {L['bus']} {f['mem_bus']}" if f.get("mem_bus") else "")
                    a = f"{L['vram']} {v}\n{CITE[lang].format(p=page)}"
                else:
                    a = f"{L.get(key, key)} {f[key]}\n{CITE[lang].format(p=page)}"
                out.append(make_example(lang, Q[key][lang].format(m=model), a))
                stats[f"gpu-{key}"] += 1

    # ── 3) 晶片組：支援世代／PCIe 通道 ─────────────────────────────────
    for model, sup, pcie, sata, page in chipset_rows(con, args.chipsets):
        for lang in langs:
            if sup:
                if lang == "en":
                    q = f"Which CPU generations does the {model} chipset support?"
                    a = "; ".join(f"{g}: {v}" for g, v in list(sup.items())[:6])
                else:
                    q = {"zh-Hant": f"{model} 晶片組支援哪些 CPU 世代？",
                         "zh-Hans": f"{model} 芯片组支持哪些 CPU 世代？"}[lang]
                    a = "；".join(f"{g}：{v}" for g, v in list(sup.items())[:6])
                out.append(make_example(lang, q, a + f"\n{CITE[lang].format(p=page)}"))
                stats["chipset-support"] += 1
            if pcie:
                if lang == "en":
                    q = f"How are the PCIe lanes allocated on the {model} chipset?"
                    a = "; ".join(f"{k}: {v}" for k, v in list(pcie.items())[:5])
                else:
                    q = {"zh-Hant": f"{model} 晶片組的 PCIe 通道怎麼分配？",
                         "zh-Hans": f"{model} 芯片组的 PCIe 通道怎么分配？"}[lang]
                    a = "；".join(f"{k}：{v}" for k, v in list(pcie.items())[:5])
                if sata:
                    a += ("；SATA " if lang != "en" else "; SATA ") + sata
                out.append(make_example(lang, q, a + f"\n{CITE[lang].format(p=page)}"))
                stats["chipset-pcie"] += 1

    # ── 3b) 使用者自備的 xlsx 型號索引（證據等級 A/B：這批最可信）────────
    # 這份是自己維護的表格，欄位乾淨、每一列都標了證據等級，所以答案要附等級——
    # 「A/B」代表有原廠或實測來源可查，跟 Wikipedia 的二手整理不是同一級。
    XLSX_Q = {
        "cores": {"zh-Hant": "{m} 有幾核心、幾執行緒？", "zh-Hans": "{m} 有几个核心、几个线程？",
                  "en": "How many cores and threads does the {m} have?"},
        "socket": {"zh-Hant": "{m} 用哪個插槽、什麼製程？",
                   "zh-Hans": "{m} 用哪个插槽、什么制程？",
                   "en": "Which socket and process node does the {m} use?"},
        "tdp": {"zh-Hant": "{m} 的 TDP 與產品定位是什麼？", "zh-Hans": "{m} 的 TDP 与产品定位是什么？",
                "en": "What are the TDP and segment of the {m}?"},
        "uarch": {"zh-Hant": "{m} 的微架構與推出年份？", "zh-Hans": "{m} 的微架构与发布年份？",
                  "en": "What microarchitecture and year is the {m}?"},
        "vram": {"zh-Hant": "{m} 的顯示記憶體與匯流排介面？",
                 "zh-Hans": "{m} 的显存与总线接口？",
                 "en": "What memory and interface does the {m} use?"},
    }

    def cite_xlsx(sheet, rowno, evidence, lang):
        ev = "證據等級 " + evidence if lang == "zh-Hant" else (
            "证据等级 " + evidence if lang == "zh-Hans" else "evidence grade " + evidence)
        if lang == "zh-Hant":
            return "〔來源：" + sheet + " 第 " + str(rowno) + " 列；" + ev + "〕"
        if lang == "zh-Hans":
            return "〔来源：" + sheet + " 第 " + str(rowno) + " 行；" + ev + "〕"
        return "[source: " + sheet + " row " + str(rowno) + "; " + ev + "]"

    def pair2(a, b, sep):
        return sep.join(x for x in (a, b) if x)

    try:
        cpu_x = con.execute(
            "SELECT model, cores, threads, socket, process, tdp, segment, uarch, year,"
            " evidence, sheet, rowno FROM hw_cpu").fetchall()
    except sqlite3.Error:
        cpu_x = []
    for idx, cx in enumerate(cpu_x):
        model, cores, threads, socket, process, tdp, segment, uarch, year, evidence, sheet, rowno = cx
        fields = {}
        if cores:
            fields["cores"] = (str(cores), str(threads or ""))
        if socket or process:
            fields["socket"] = (pair2(socket, process, " / "), "")
        if tdp or segment:
            fields["tdp"] = (pair2(tdp, segment, " · "), "")
        if uarch or year:
            fields["uarch"] = (pair2(uarch, year, " · "), "")
        avail = [k for k in ("cores", "socket", "tdp", "uarch") if k in fields]
        if not avail:
            continue
        start = idx % len(avail)
        for key in (avail[start:] + avail[:start])[:2]:
            v1, v2 = fields[key]
            for lang in langs:
                if key == "cores" and v2:
                    a = (v1 + " " + LBL[lang]["cores"] + " / " + v2 + " " + LBL[lang]["threads"]
                         if lang == "en"
                         else v1 + " 個" + LBL[lang]["cores"] + "、" + v2 + " " + LBL[lang]["threads"])
                else:
                    a = v1
                out.append(make_example(lang, XLSX_Q[key][lang].format(m=model),
                                        a + "\n" + cite_xlsx(sheet, rowno, evidence, lang)))
                stats["xlsx-cpu-" + key] += 1

    try:
        gpu_x = con.execute(
            "SELECT model, vram, interface, chip, uarch, year, evidence, sheet, rowno"
            " FROM hw_gpu").fetchall()
    except sqlite3.Error:
        gpu_x = []
    for idx, gx in enumerate(gpu_x):
        model, vram, interface, chip, uarch, year, evidence, sheet, rowno = gx
        fields = {}
        if vram or interface:
            fields["vram"] = (pair2(vram, interface, " / "), "")
        if chip or uarch:
            fields["uarch"] = (pair2(chip, uarch, " · "), "")
        avail = [k for k in ("vram", "uarch") if k in fields]
        if not avail:
            continue
        start = idx % len(avail)
        for key in (avail[start:] + avail[:start])[:1]:
            v1 = fields[key][0]
            for lang in langs:
                out.append(make_example(lang, XLSX_Q[key][lang].format(m=model),
                                        v1 + "\n" + cite_xlsx(sheet, rowno, evidence, lang)))
                stats["xlsx-gpu-" + key] += 1

    # ── 4) 拒答／邊界：這些是「加固」的重點，不是附錄 ──────────────────
    # 這幾種問法在真實使用中最容易被小模型硬答：不存在的型號、焊死的行動版、
    # 來源沒有的欄位、以及「幫我看我的電腦」。每一種都要有一句明確、不繞的拒答。
    refusals = [
        ({"zh-Hant": "i9-15900K 有幾核心？", "zh-Hans": "i9-15900K 有几个核心？",
          "en": "How many cores does the i9-15900K have?"},
         {"zh-Hant": "型號庫裡沒有 i9-15900K 這一顆（Intel 沒有這個型號），我不會用推測的規格回答。",
          "zh-Hans": "型号库里没有 i9-15900K 这一颗（Intel 没有这个型号），我不会用推测的规格回答。",
          "en": "The model database has no i9-15900K (Intel never shipped that SKU), so I will not answer with guessed specs."}),
        ({"zh-Hant": "我的 CPU 現在幾度？", "zh-Hans": "我的 CPU 现在几度？",
          "en": "What temperature is my CPU running at right now?"},
         {"zh-Hant": "我讀不到你的電腦。要量測溫度請用曦覽的感測器讀取流程；我只能回答文件與型號規格。",
          "zh-Hans": "我读不到你的电脑。要测量温度请用曦览的传感器读取流程；我只能回答文档与型号规格。",
          "en": "I cannot read your machine. Use XinSpect's sensor read path to measure it; I only answer from documents and model specs."}),
        ({"zh-Hant": "Ryzen 9 7945HX 可以裝哪張主機板？", "zh-Hans": "Ryzen 9 7945HX 可以装哪张主板？",
          "en": "Which motherboard can I put a Ryzen 9 7945HX in?"},
         {"zh-Hant": "7945HX 是行動版 BGA 封裝，直接焊在主機板上，沒有可搭配的零售主機板——這一題不成立，不是資料缺。",
          "zh-Hans": "7945HX 是移动版 BGA 封装，直接焊在主板上，没有可搭配的零售主板——这个问题本身不成立，不是资料缺。",
          "en": "The 7945HX is a mobile BGA part soldered to the board; there is no retail motherboard to pair it with. The question does not apply."}),
        ({"zh-Hant": "B650 晶片組支援幾條 PCIe 6.0 通道？", "zh-Hans": "B650 芯片组支持几条 PCIe 6.0 通道？",
          "en": "How many PCIe 6.0 lanes does the B650 chipset support?"},
         {"zh-Hant": "來源的 B650 那一列沒有 PCIe 6.0 這一欄，只有 PCIe 4.0／5.0。沒有寫到的我不補。",
          "zh-Hans": "来源的 B650 那一列没有 PCIe 6.0 这一栏，只有 PCIe 4.0／5.0。没写到的我不补。",
          "en": "The B650 row in the source has no PCIe 6.0 column, only PCIe 4.0/5.0. I will not fill in what is not stated."}),
        ({"zh-Hant": "幫我看一下我這台電腦的 BIOS 版本。", "zh-Hans": "帮我看一下我这台电脑的 BIOS 版本。",
          "en": "Check the BIOS version on my machine."},
         {"zh-Hant": "我沒有辦法讀取你的電腦，也看不到 BIOS。請在開機自檢畫面或系統資訊裡查；我可以教你怎麼找。",
          "zh-Hans": "我没办法读取你的电脑，也看不到 BIOS。请在开机自检画面或系统信息里查；我可以教你哪里找。",
          "en": "I cannot read your machine or see its BIOS. Check the POST screen or system information; I can tell you where to look."}),
        ({"zh-Hant": "EPYC 9654 的保固是幾年？", "zh-Hans": "EPYC 9654 的保修是几年？",
          "en": "How many years of warranty does the EPYC 9654 have?"},
         {"zh-Hant": "型號庫沒有保固年限這一欄（它記的是規格與推出時間），我不猜數字。保固要看經銷商與地區條款。",
          "zh-Hans": "型号库没有保修年限这一栏（它记的是规格与发布时间），我不猜数字。保修要看经销商与地区条款。",
          "en": "The model database records specs and release dates, not warranty terms, so I will not invent a number. Warranty depends on the reseller and region."}),
    ]
    for qs, ans in refusals:
        for lang in langs:
            out.append(make_example(lang, qs[lang], ans[lang]))
            stats["refusal"] += 1

    random.shuffle(out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"寫出 {len(out)} 筆 → {args.out}")
    for k, v in stats.most_common(20):
        print(f"  {k:18s} {v}")


if __name__ == "__main__":
    raise SystemExit(main())
