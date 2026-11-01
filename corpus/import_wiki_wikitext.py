#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""從語料裡的 wikitext，把「型號清單」表格拆成可查的資料列。

為什麼用 wikitext 而不是已經切好的 tables.jsonl：
  tables.jsonl 是「已渲染」的表格，`rowspan` 的合併儲存格在抽取階段就掉了，
  所以 Intel Core 那幾張表只剩「Core i7」這種群組名，型號（i7-12700K）整批不見。
  wikitext 裡 `rowspan="2" | i7-12700K` 是原樣的，自己解才拿得到型號。

範圍：只收「有 Cores／Threads／TDP／Socket／Shaders 這類欄位」的表格，
      再從每一列**前面幾格**找出型號字串（型號幾乎都寫在列首；整列一起比對
      會把「Celeron … 1066 …」這種規格數字誤認成型號）。寧可少收，不要把標題列當型號。

用法（可給多個目錄／檔案，會一起看去重）：
  python corpus/import_wiki_wikitext.py --docs corpus/raw-fetched corpus/raw
  python corpus/import_wiki_wikitext.py --docs <dir> --sample 40   # 抽樣人工看精確度
"""

import argparse
import json
import os
import re
import sqlite3
import sys

# `python -I`（隔離模式）不會把腳本目錄放進 sys.path，所以自己補上，
# 否則匯入同目錄的 wiki_html 會失敗。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wiki_html

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "entities.db")

SPEC_HINT = re.compile(
    r"(cores?|threads?|tdp|socket|shaders?|cuda|clock|memory|製程|核心|執行緒)", re.I)
GPU_HINT = re.compile(r"(shaders?|cuda|stream processors|graphics|vram|memory bus)", re.I)

# 型號樣式：各家族的名稱格式。寧可漏收，不要亂收。
MODEL_RE = re.compile(
    r"(?:\bcore\s*2\s+(?:duo|quad|extreme|solo)\s+[eqx]\w{3,5}\b"
    r"|\b(?:core\s+)?(?:i[3579]|m[357]|ultra\s*[3579])[-\s]?\d{3,5}[a-z]{0,3}\b"
    r"|\bryzen\s+(?:ai\s+)?\d\s*\d{3,4}[a-z0-9]{0,4}\b"
    r"|\bthreadripper(?:\s+pro)?\s+\d{4}[a-z]{0,3}\b"
    r"|\bepyc\s+\d{4}[a-z]{0,4}\b"
    r"|\bxeon\s+(?:platinum|gold|silver|bronze)?\s*(?:[a-z]{1,2}\d{0,2}-)?\d{3,5}[a-z]{0,3}\b"
    r"|\b(?:atom|celeron|pentium|opteron|turion|sempron|duron)\s+"
    r"(?:[a-z]+\s+)?[a-z]?\d{3,4}[a-z]{0,2}\b"
    r"|\bathlon\s+(?:64\s+)?(?:x2\s+)?[a-z]?\d{3,4}\+?\b"
    r"|\bphenom\s+(?:ii\s+)?x[2346]\s+\d{3,4}\b"
    r"|\ba(?:1|4|6|8|10)-\d{4}[a-z]{0,2}\b"
    r"|\bfx-\d{4}[a-z]{0,2}\b"
    r"|\bsnapdragon\s+(?:\d{3}\w?|\d\s+gen\s*\d|x\s+(?:elite|plus))\b"
    r"|\bm[1-4]\s*(?:pro|max|ultra)?\b"
    r"|\ba\d{1,2}\s+(?:bionic|fusion)\b"
    r"|\b(?:geforce\s+)?(?:gtx|rtx)\s*\d{3,4}\s*(?:ti|super|max-q|d)?\b"
    r"|\b(?:geforce\s+)?(?:gts|gt)\s+\d{3,4}\s*(?:ti|super|max-q)?\b(?!\s*(?:mb|gb|mhz))"
    r"|\bgeforce\s+\d{1,4}(?:\s*(?:mx|gts|gt|ti|ultra|le|gs|gtx))?\b"
    r"|\bradeon\s+(?:hd\s*)?\d{3,5}[a-z]{0,2}(?:\s*(?:pro|xt|xtx|le|se))?\b"
    r"|\bradeon\s+(?:rx\s*)?vega\s*\d{2}\b"
    r"|\bradeon\s+(?:rx\s*)?\d{3,4}[a-z]{0,2}(?:\s+(?:xtx|xt|gre|m|s|pro|gold|silver|ultra))?\b"
    r"|\brx\s+\d{3,4}[a-z]{0,2}(?:\s+(?:xtx|xt|gre|m|s))?\b"
    r"|\br[5-9]\s?\d{3}[a-z]?\b"
    r"|\barc\s+[abg]\d{3}[a-z]{0,2}\b"
    r"|\bfirepro\s+[a-z]?\d{3,4}\w?\b"
    r"|\bquadro\s+(?:rtx\s+)?[a-z]?\d{3,4}[a-z]{0,3}\b"
    r"|\btesla\s+[a-z]\d{2,4}\b)", re.I)


def norm(s):
    s = re.sub(r"\[[^\]]*\]", " ", str(s or ""))
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[^0-9a-z]+", "", s.lower())


def clean(s):
    s = str(s or "")
    # 儲存格開頭殘留的屬性（`colspan="2"`、`style="…"`）不是資料。
    s = re.sub(r'^\s*(?:(?:rowspan|colspan|style|class|align|scope|data-[a-z-]+)'
               r'\s*=\s*"[^"]*"\s*)+', "", s)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\{\{[^{}]*\}\}", " ", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    s = re.sub(r"''+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def cell_parts(raw):
    """(值, rowspan, colspan)。`rowspan="3" | X` 這種屬性要留著展開用。"""
    if "|" in raw:
        head, _, tail = raw.partition("|")
        if re.search(r"\b(rowspan|colspan|style|class|align|scope)\s*=", head, re.I):
            rs = re.search(r"rowspan\s*=\s*\"?(\d+)", head, re.I)
            cs = re.search(r"colspan\s*=\s*\"?(\d+)", head, re.I)
            return (tail.strip(),
                    int(rs.group(1)) if rs else 1,
                    int(cs.group(1)) if cs else 1)
    return (raw.strip(), 1, 1)


def split_cells(line):
    s = line.strip()
    if s.startswith("||"):
        return s[2:].split("||")
    if s.startswith("|"):
        return s[1:].split("||")
    return []


def parse_table(block):
    """回傳 rows：list[list[str]]，rowspan／colspan 依 MediaWiki 語義展開到後續列。

    不展開的話，i7-12700K 那一列會拿不到「核心數」——值掛在它上面幾列的 rowspan 裡。
    代價：跨行寫成的儲存格（內容接在下一行）會讓欄位偶爾漂移一格；我們的回答照列
    原文引用、不逐欄解讀，所以可接受。
    """
    rows, cur = [], []
    col = 0
    pending = {}                       # col -> [value, 還要用幾列]

    def close_row():
        nonlocal cur, col
        if cur:
            rows.append(cur)
        cur, col = [], 0

    for line in block:
        s = line.strip()
        if s.startswith("|-"):
            close_row()
            continue
        if s.startswith("!"):
            raws = [r for r in s[1:].split("!!")]
        elif s.startswith("|"):
            raws = split_cells(s)
        else:
            continue
        for raw in raws:
            while col in pending:
                v, left = pending[col]
                cur.append(v)
                if left <= 1:
                    del pending[col]
                else:
                    pending[col][1] = left - 1
                col += 1
            val, rs, cs = cell_parts(raw)
            cur.append(val)
            if rs > 1:
                pending[col] = [val, rs - 1]
            # colspan 跨過的欄位：把殘留的 rowspan 清掉。不清的話，那格的值
            # 會一直被帶進後面的列，變成「這張表的型號欄其實是上一列的」。
            for _ in range(1, max(1, cs)):
                col += 1
                pending.pop(col, None)
            col += 1
    close_row()
    return rows


def table_header(block):
    """表頭：wikitext 裡 `!` 開頭那幾行，去掉屬性與連結語法。"""
    out = []
    for line in block:
        t = line.strip()
        if t.startswith("!"):
            out.extend(clean(x) for x in t[1:].split("!!") if clean(x))
    return out[:16]


# 這些列的「型號」其實是群組名（「Core i5/i7/i9-13000H Series」「Celeron 800/900/1000」），
# 查表時拿它去比對會把整代 CPU 對到一個不存在的型號。
# Intel／AMD 的規格表常把「品牌」寫在合併儲存格（Core i7、Ryzen 5、Xeon Gold），
# 型號欄只留裸數字（880、5600X、E5-2680 v3）。只認整串型號的話，這些頁面會一格都抽不到。
BRAND_CELL = re.compile(
    r"^(?:intel\s+)?(?:core\s*(?:2\s*)?(?:i[3579]|ultra\s*[3579]|m[357]|[3579])"
    r"|xeon(?:\s+(?:platinum|gold|silver|bronze))?"
    r"|celeron|pentium(?:\s+(?:gold|silver|dual-core))?|atom|itanium"
    r"|ryzen(?:\s+(?:ai\s+)?\d)?|threadripper(?:\s+pro)?|epyc"
    r"|athlon(?:\s+(?:ii|x2|64|xp))?|phenom(?:\s+(?:ii))?"
    r"|sempron|turion|opteron|duron)\b", re.I)
SKU_CELL = re.compile(
    r"^(?:[a-z]{1,2}\d{0,2}-?)?\d{3,5}[a-z0-9]{0,4}(?:\s*v\d)?"
    r"(?:\s+(?:be|black edition|extreme edition|mobile|pro|ulv))?$", re.I)


def compose_model(lead):
    """品牌格＋型號格 → 完整型號（「Core i7」＋「880」→「Core i7 880」）。"""
    for i, cell in enumerate(lead[:4]):
        if not BRAND_CELL.match(cell):
            continue
        for j in range(i + 1, min(len(lead), i + 3)):
            if SKU_CELL.match(lead[j]):
                return f"{cell} {lead[j]}"
    return None


GROUP_ROW = re.compile(
    r"(?:\bseries\s*$|\bfamily\s*$|generation\s*$|微架構|系列|\d\s*/\s*\d)", re.I)

# 圖片說明列（infobox 的 caption 也會被解析成表格列），不是型號資料。
CAPTION_ROW = re.compile(
    r"(\btop\s*:|\bbottom\s*:|\blogo\b|\b\d+px\b|\.(?:png|jpe?g|svg)\b"
    r"|founders edition released)", re.I)

# 「Ryzen 5 (5600X), Ryzen 9 (5900X, 5950X)」：品牌後面括號裡的才是型號。
GROUP_SKUS = re.compile(
    r"((?:ryzen(?:\s+(?:ai\s+)?\d)?|threadripper(?:\s+pro)?|epyc|core\s+i[3579]"
    r"|xeon(?:\s+(?:platinum|gold|silver|bronze))?))\s*\(([^)]*\d[^)]*)\)", re.I)

PAGE_BRAND = re.compile(
    r"^(?:list_of_|table_of_)?(epyc|xeon|ryzen|threadripper|celeron|pentium|atom"
    r"|athlon|sempron|opteron|duron|phenom)", re.I)


def page_brand(page):
    """由頁名推品牌：`Epyc` → `EPYC`、`List_of_Intel_Xeon_processors_…` → `Xeon`。"""
    base = re.sub(r"^(?:list_of_|table_of_)", "", page or "", flags=re.I)
    m = PAGE_BRAND.match(base)
    if not m:
        return None
    return "EPYC" if m.group(1).lower() == "epyc" else m.group(1).title()


GROUP_SKUS = re.compile(
    r"((?:ryzen(?:\s+(?:ai\s+)?\d)?|threadripper(?:\s+pro)?|epyc|core\s+i[3579]"
    r"|xeon(?:\s+(?:platinum|gold|silver|bronze))?))\s*\(([^)]*\d[^)]*)\)", re.I)

# 顯示卡型號的辨識字（用來把「這張表是 GPU 還是 CPU」判對，不再靠欄位猜）。
GPU_MODEL = re.compile(
    r"(geforce|radeon|quadro|tesla|firepro|rtx|gtx|\brx\s|\bgt\s|\bgts\s|\barc\s"
    r"|snapdragon)", re.I)


def strip_prefix(model):
    """「Core i7-12700K」→「i7-12700K」：同一個型號在別頁常不帶前綴。"""
    return re.sub(r"^(?:intel|amd|nvidia|core\s*[23]?|geforce|radeon|athlon|"
                  r"phenom|opteron|turion|sempron)\s+", "", model, flags=re.I)


def aliases(model):
    """全名、去前綴、以及「有數字也有字母」的尾端組合（純數字尾端會誤中解析度）。"""
    out = set()
    full, bare = norm(model), norm(strip_prefix(model))
    if len(full) >= 4:
        out.add(full)
    if len(bare) >= 4:
        out.add(bare)
    if re.match(r"snapdragon\s", model, re.I):
        sd = norm("sd " + re.sub(r"^snapdragon\s+", "", model, flags=re.I))
        if len(sd) >= 4:
            out.add(sd)
    words = [w for w in re.split(r"\s+", model.strip()) if w]
    for k in range(1, len(words)):
        tail = norm(" ".join(words[k:]))
        # 5 個字元起（5600X、12700K 這種長度才收得到），且必須數字＋字母混用，
        # 以免「1080」這種純數字把解析度當成顯示卡。
        if len(tail) >= 5 and re.search(r"\d", tail) and re.search(r"[a-z]", tail):
            out.add(tail)
    return sorted(out, key=len, reverse=True)


def iter_docs(path, limit=0):
    """path 可以是裝滿 *.wiki 的目錄，也可以是 documents.jsonl。"""
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if f.endswith(".wiki") or f.endswith(".html"))
        for i, name in enumerate(names):
            if limit and i >= limit:
                break
            with open(os.path.join(path, name), encoding="utf-8", errors="replace") as fh:
                yield os.path.splitext(name)[0], fh.read(), os.path.splitext(name)[1]
        return

    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            try:
                doc = json.loads(line)
            except Exception:
                continue
            yield (doc.get("source") or doc.get("id") or ""), (doc.get("text") or ""), ""


SOCKET_DECL = re.compile(
    r"[Ss]ocket\s*:?\s*(?:\[\[\s*)?(?:Socket\s+)?"
    r"(LGA\s?\d{3,4}(?:-\d)?|AM\d\+?|FM\d\+?|SP\d|s?WRX\d{1,2}|TR[SD]\d|BGA\d{3,4})",
    re.I)
HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
# 單純一個「值」是不是插槽（表格已經自己寫了插槽時，不要再補段落的）
SOCK_VAL = re.compile(
    r"^(?:socket\s*)?(LGA\s?\d{3,4}(?:-\d)?|AM\d\+?|FM\d\+?|SP\d|s?WRX\d{1,2}"
    r"|TR[SD]\d|BGA\d{3,4})$", re.I)


def scan_sections(text):
    """每一行屬於哪個標題、那個標題宣告了什麼插槽。

    Intel／AMD 的清單頁把插槽寫在**段落開頭的子彈**（`* Socket: [[LGA 1156]]`），
    不是表格的每一列。列本身看不到它，所以切表時要把段落帶上。
    """
    out = []           # (line_no, heading, socket)
    heading, sock = "", None
    for i, ln in enumerate(text.splitlines()):
        m = HEADING.match(ln.strip())
        if m:
            heading, sock = m.group(2).strip(), None
            out.append((i, heading, None))
            continue
        d = SOCKET_DECL.search(ln)
        if d and sock is None:
            sock = re.sub(r"\s+", " ", d.group(1)).strip()
        out.append((i, heading, sock))
    return out


def split_tables(text):
    """回傳 [(block, 起始行號)]；有行號才對得上 section 的插槽宣告。"""
    blocks, cur, start = [], None, 0
    for i, ln in enumerate(text.splitlines()):
        if ln.lstrip().startswith("{|"):
            cur, start = [ln], i
        elif cur is not None:
            cur.append(ln)
            if ln.lstrip().startswith("|}"):
                blocks.append((cur, start))
                cur = None
    if cur:
        blocks.append((cur, start))
    return blocks


CPULIST_RE = re.compile(r"\{\{\s*cpulist\b", re.I)


def unwrap_templates(s):
    """{{nowrap|65 W}} → 65 W；{{dts|2018|2}} → 2018 2。有 = 的是參數模板，整段丟。"""
    for _ in range(4):
        new = re.sub(r"\{\{[^{}]*\}\}",
                     lambda m: ("" if "=" in m.group(0) else
                                (m.group(0)[2:-2].split("|")[-1].strip())),
                     s)
        if new == s:
            break
        s = new
    return s


def split_params(body):
    """在 `{{cpulist|…}}` 的內文裡按 `|` 切，但不要切到位於巢狀模板／連結裡的 `|`。"""
    parts, buf, depth_t, depth_l = [], [], 0, 0
    i = 0
    while i < len(body):
        two = body[i:i + 2]
        if two == "{{":
            depth_t += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}":
            depth_t = max(0, depth_t - 1)
            buf.append(two)
            i += 2
            continue
        if two == "[[":
            depth_l += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]":
            depth_l = max(0, depth_l - 1)
            buf.append(two)
            i += 2
            continue
        ch = body[i]
        if ch == "|" and depth_t == 0 and depth_l == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf))
    return parts


def iter_cpulist(text):
    """抽 `{{cpulist|model=…|cores=…|tdp=…}}`。

    現代 Xeon 清單頁（Skylake、Ice Lake、Cascade Lake…）不是表格，而是每顆 CPU
    一個模板呼叫。模板參數是結構化的（model／cores／threads／tdp／socket），
    比表格還好用：不會有合併儲存格掉欄的問題。
    """
    for m in CPULIST_RE.finditer(text):
        start = m.end()
        depth, i = 1, start
        while i < len(text) and depth:
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
            else:
                i += 1
        if depth:
            continue
        body = text[start:i - 2]
        kv = {}
        for p in split_params(body):
            if "=" not in p:
                continue
            k, _, v = p.partition("=")
            k = k.strip().lower()
            v = clean(unwrap_templates(v))
            if k and v:
                kv[k] = v
        if kv.get("model"):
            yield kv


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True, nargs="+",
                    help="一個或多個目錄（裝 *.wiki／*.html）或 documents.jsonl")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="印出 N 個抽樣，人工看精確度")
    ap.add_argument("--keep", action="store_true", help="保留既有 wiki-* 資料（預設先清空）")
    ap.add_argument("--chunks", default="",
                    help="順便寫一份檢索用 chunk（例：corpus/corpus-wiki.jsonl）")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.execute("""CREATE TABLE IF NOT EXISTS hw_model (
        row_id INTEGER PRIMARY KEY, model TEXT, norm TEXT, kind TEXT, page TEXT,
        table_index INTEGER, row_index INTEGER, cells TEXT, headers TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_hw_model_norm ON hw_model(norm)")
    con.execute("""CREATE TABLE IF NOT EXISTS hw_alias (
        alias TEXT NOT NULL, kind TEXT NOT NULL, row_id INTEGER NOT NULL, model TEXT, vendor TEXT)""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_hw_alias ON hw_alias(alias)")
    if not args.keep:
        # 只清「我們自己抓的 wiki 表格」；lib-* 是 import_hw_library.py 寫的規則庫，
        # 兩支腳本各管各的，避免重跑一支就把另一支的資料清空。
        con.execute("DELETE FROM hw_model WHERE kind LIKE 'wiki-%'")
        con.execute("DELETE FROM hw_alias WHERE kind LIKE 'wiki-%'")

    seen, total, docs_hit = {}, 0, 0
    # seen 的鍵是 (型號, 來源頁)：同一顆 CPU 在 Alder Lake 頁與 Core 清單頁各有一列，
    # 兩列的欄位不同、可能互相補足。只留一列會把資訊丟掉，所以兩邊都存，
    # 查詢時再挑「那顆自己的、欄位最多的」那一列（rag/entities.py 的 hw_models）。
    by_page = {}
    collected = {}

    def _standalone(model, cells):
        """這一列是不是「這顆自己的」規格列（而不是一格寫著整組 SKU 的家族列）。

        型號格常帶括號註記（「Radeon RX 7900 XTX (Navi 31)」），所以除了完全相同，
        「以型號開頭」也算——但不能太短，否則「RX 7900」會把整個系列都算進來。
        """
        want = [w for w in (norm(model), norm(strip_prefix(model))) if w]
        for c in cells:
            cn = norm(c) if c else ""
            if not cn:
                continue
            for w in want:
                if cn == w or (len(w) >= 6 and cn.startswith(w)
                               and len(cn) <= len(w) + 24):
                    return True
        return False

    def add_model(model, kind, page, cells, headers, ti=0, ri=0):
        """型號＋同一列的原文 → hw_model／hw_alias。回傳是否真的新增／升級。"""
        nonlocal total
        key = norm(strip_prefix(model))
        if len(norm(model)) < 4:
            return False
        own = _standalone(model, cells)
        rank = (1 if own else 0, len(cells))

        prev = seen.get((key, page))
        if prev is not None:
            # 同一型號可能先被家族列／包裝列佔走，後來才遇到逐顆規格列：
            # 只有在「更像這一顆自己的規格、而且欄位更多」時才換掉。
            if rank <= prev[1]:
                return False        # 同頁重覆（不同 step 列）只留欄位最多的那一条
            con.execute("DELETE FROM hw_model WHERE row_id=?", (prev[0],))
            con.execute("DELETE FROM hw_alias WHERE row_id=? AND kind=?", (prev[0], kind))

        total += 1
        seen[(key, page)] = (total, rank)
        # row_id 要自己指定：SQLite 的自動 rowid 在刪過列之後會和計數器岔開，
        # 而 hw_alias.row_id 指的就是它——岔開的結果是「型號對到別人的規格列」。
        con.execute(
            "INSERT INTO hw_model (row_id,model,norm,kind,page,table_index,row_index,cells,headers)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (total, model, key, kind, page, ti, ri,
             json.dumps(cells, ensure_ascii=False),
             json.dumps(headers, ensure_ascii=False)))
        collected[key] = (model, page, cells)
        for alias in aliases(model):
            con.execute(
                "INSERT INTO hw_alias (alias,kind,row_id,model,vendor) VALUES (?,?,?,?,?)",
                (alias, kind, total, model, page))
        return True

    def process_rows(rows, page, kind, header, ti, sec_sock=None):
        added = 0
        for ri, cells in enumerate(rows):
            lead = [clean(c) for c in cells if clean(c)][:3]
            head_text = " | ".join(lead)
            if GROUP_ROW.search(lead[0] if lead else ""):
                continue
            # 圖片說明列（「Top: Logo…」「… 275px|…」）不是型號資料，別讓它搶到型號。
            if CAPTION_ROW.search(head_text):
                continue

            # 一列可能寫著一整組 SKU：「Ryzen 5 (5600X), Ryzen 9 (5900X, 5950X)」。
            # 這種列本身是家族，但括號裡是實打實的型號，逐個展開才查得到。
            models = []
            wide = " | ".join([clean(c) for c in cells if clean(c)][:6])
            for gm in GROUP_SKUS.finditer(wide):
                brand = re.sub(r"\s+", " ", gm.group(1)).strip()
                for sku in gm.group(2).split(","):
                    mm = MODEL_RE.search(f"{brand} {sku.strip()}")
                    if mm:
                        models.append(mm.group(0).strip())

            if not models:
                m = MODEL_RE.search(head_text)
                if m:
                    models.append(m.group(0).strip())
                else:
                    cm = MODEL_RE.search(compose_model(lead) or "")
                    if cm:
                        models.append(cm.group(0).strip())
                    else:
                        # 有些頁面（Epyc）的表格只寫裸型號（9754S），品牌在頁名或群組表頭。
                        brand = page_brand(page)
                        if brand and lead and SKU_CELL.match(lead[0]):
                            cm = MODEL_RE.search(f"{brand} {lead[0]}")
                            if cm:
                                models.append(cm.group(0).strip())

            clean_cells = [clean(c) for c in cells if clean(c)]
            row_header = list(header)
            if sec_sock and not any(SOCK_VAL.match(str(c)) for c in clean_cells):
                # 段落插槽：加在最後，值和標籤一起加，維持 headers/cells 對齊
                clean_cells = clean_cells + [sec_sock]
                row_header = row_header + ["Socket(段落)"]
            for model in models:
                # 用型號本身判 CPU／GPU：Intel Core 的表也有 Graphics 欄，
                # 靠欄位猜會把 i7 歸成顯示卡，查表回覆就標錯類別。
                model_kind = "wiki-gpu" if GPU_MODEL.search(model) else "wiki-cpu"
                if add_model(model, model_kind, page, clean_cells, row_header, ti, ri):
                    added += 1
        return added

    for docs_path in args.docs:
        for page, text, ext in iter_docs(docs_path, args.limit_docs):
            doc_added = 0

            if ext == ".html":
                # 模板轉嵌出來的表格（{{AMD Radeon RX 6000 series}} 之類）只在渲染後看得到。
                for ti, table in enumerate(wiki_html.parse_tables(text)):
                    body = " ".join(" ".join(r) for r in table)
                    if not SPEC_HINT.search(body):
                        continue
                    kind = "wiki-gpu" if GPU_HINT.search(body[:2000]) else "wiki-cpu"
                    header = table[0][:16] if table else []
                    doc_added += process_rows(table, page, kind, header, ti)
            else:
                if not ("wikitable" in text or CPULIST_RE.search(text)):
                    continue
                if not SPEC_HINT.search(text[:40000]):
                    continue
                sections = scan_sections(text)
                for ti, (block, start_ln) in enumerate(split_tables(text)):
                    body = "\n".join(block)
                    if not SPEC_HINT.search(body):
                        continue
                    kind = "wiki-gpu" if GPU_HINT.search(body[:2000]) else "wiki-cpu"
                    # 段落宣告的插槽（`* Socket: [[LGA 1156]]`）補進這一張表的每一列。
                    # 這是 CPU↔主機板相容性唯一可靠的來源；表格本身常常沒有這一欄。
                    sec_sock = None
                    if start_ln < len(sections):
                        sec_sock = sections[start_ln][2]
                    doc_added += process_rows(parse_table(block), page, kind,
                                              table_header(block), ti, sec_sock)

                # 模板式清單：現代 Xeon 頁沒有表格，每顆 CPU 一個 {{cpulist|model=…|cores=…}}
                for kv in iter_cpulist(text):
                    model = kv["model"]
                    keys = [k for k in kv if k != "model"]
                    if add_model(model, "wiki-cpu", page, [kv[k] for k in keys], keys):
                        doc_added += 1

            if doc_added:
                by_page[page] = by_page.get(page, 0) + doc_added
                docs_hit += 1

    con.commit()
    con.close()

    rows_out = list(collected.values())
    if args.chunks:
        # 檢索用：一列一個 chunk，讓「某型號什麼時候出」這種敘述題也能引用到原始清單。
        with open(args.chunks, "w", encoding="utf-8", newline="\n") as fh:
            for i, (model, page, cells) in enumerate(rows_out):
                title = page.replace("_", " ")
                body = "；".join(cells[:12])
                rec = {
                    "id": f"wiki.hw.{i:05d}",
                    "source": f"wiki.{page}",
                    "lang": "en",
                    "doc_title": title,
                    "heading_path": f"{title} · {model}",
                    "anchor": "",
                    "line_start": "0",
                    "line_end": "0",
                    "paragraph": "",
                    "text": f"【型號】{model}（{title}）。{body}",
                    "url": f"https://en.wikipedia.org/wiki/{page.replace('_', ' ')}",
                    "license": "CC-BY-SA-4.0",
                    "origin": "wiki-hw",
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"chunks : {len(rows_out)} 筆 → {args.chunks}")

    print(f"抽出 {len(collected)} 個型號（去重後），來自 {docs_hit} 份文件 / {len(by_page)} 個來源")
    for page, c in sorted(by_page.items(), key=lambda x: -x[1])[:20]:
        print(f"  {c:5d}  {page}")
    if args.sample:
        import random
        random.seed(7)
        print(f"\n--- 抽樣 {args.sample} 個（型號 | 來源 | 前幾格）---")
        for model, page, cells in random.sample(rows_out, min(args.sample, len(rows_out))):
            print(f"  {model:22s} | {page[:36]:36s} | {' / '.join(cells[:4])[:78]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
