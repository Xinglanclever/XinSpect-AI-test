#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""硬體 ID 的精確查找：PCI / USB / MAC OUI / PNP / GPU device / CVE / 軟體生命週期。

為什麼這一層不經過語言模型：
  `10de:1b80` 對應什麼，是一個**查表**問題，不是語言問題。丟給模型只會得到
  兩種結果——答對（它剛好記得）或答錯（它編一個）。而這份語料裡有 21,835 個
  PCI 裝置、21,698 個 USB 裝置、80,603 個 OUI 前綴；模型不可能記得全部，
  也不該被要求記得。查表是零成本、零幻覺的。

  所以服務的分工是：ID 類問題先查表，查到就**直接用查到的名稱回答**，
  模型只負責把結果講成一句話（甚至不需要）。查不到就明說查不到——
  「這張表裡沒有這個 ID」與「沒有這個廠商」是兩件事，不能混。
"""

import json
import os
import re
import sqlite3
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "..", "corpus", "entities.db")

# ── ID 樣式 ─────────────────────────────────────────────────────────────
# 依「最有辨識度」排序：先認有冒號的 PCI 對，再認單一 4 碼 ID，最後才是 OUI。
RE_PCI_PAIR = re.compile(r"\b([0-9a-f]{4})\s*[:：]\s*([0-9a-f]{4})\b", re.I)
RE_USB_PAIR = re.compile(r"\b([0-9a-f]{4})\s*[:：]\s*([0-9a-f]{4})\b", re.I)
RE_HEX4 = re.compile(r"\b(?:0x)?([0-9a-f]{4})\b", re.I)
RE_OUI = re.compile(r"\b([0-9a-f]{2}(?:[:：-]?[0-9a-f]{2}){2})\b", re.I)
RE_HWID = re.compile(r"\b(PCI|USB|ACPI|HID|PNP|VEN|DEV)[\\_]([A-Z0-9]{3,8})\b", re.I)
RE_CVE = re.compile(r"\bCVE-(\d{4})-(\d{4,7})\b", re.I)
RE_VENDOR_WORD = re.compile(r"(廠商|厂商|vendor|製造商|制造商|哪家|哪一家|誰的|谁的)", re.I)
RE_DEVICE_WORD = re.compile(r"(裝置|装置|device|零件|型號|型号|是什麼|是什么|what is)", re.I)
# 型號庫的 kind 是依表格判的，偶爾會判錯；回報類別時直接看型號字（顯示卡家族字）。
GPU_WORD = re.compile(r"(geforce|radeon|quadro|tesla|firepro|rtx|gtx|\brx\s|snapdragon)", re.I)

RE_MAC_WORD = re.compile(r"(mac|oui|網卡|网卡|前綴|前缀|prefix)", re.I)

# 晶片組名（問句點名時才查型號庫）：B650／B650E／X870E／Z790／W680／TRX50／WRX90。
# 排除 Intel Arc 顯示卡的 A310／A380／A580／A750／A770——它們長得跟晶片組一樣。
CHIPSET_TOKEN = re.compile(r"\b(?:[ABXZQHWC]\d{3}[A-Z]{0,2}|TRX\d{2}|WRX\d{2})\b")
ARC_GPU_NAMES = {"A310", "A380", "A580", "A750", "A770"}

# ── CPU／GPU 型號 ───────────────────────────────────────────────────────
# 為什麼要另外一套：`i7-12700K 幾核` 和 `10de:1b80` 一樣是查表問題，不是語言問題。
# 型號庫（corpus/import_hwdb.py＋corpus/import_wiki_wikitext.py）有幾千列，
# 模型記不全、也記不準；查到就照抄，查不到就明說。
RE_HW_MODEL = re.compile(
    r"(?:\bcore\s*2\s+(?:duo|quad|extreme|solo)\s+[eqx]\w{3,5}\b"
    r"|\b(?:core\s+)?(?:i[3579]|m[357]|ultra\s*[3579])[-\s]?\d{3,5}[a-z]{0,3}\b"
    r"|\bryzen\s+(?:ai\s+)?\d\s*\d{3,4}[a-z0-9]{0,4}\b"
    r"|\bthreadripper(?:\s+pro)?\s*\d{4}[a-z]{0,3}\b"
    r"|\bepyc\s+\d{4}[a-z]{0,4}\b"
    r"|\bxeon\s+(?:platinum|gold|silver|bronze)?\s*(?:[a-z]{1,2}\d{0,2}-)?\d{3,5}[a-z]{0,3}\b"
    r"|\b(?:atom|celeron|pentium|opteron|turion|sempron|duron)\s+(?:[a-z]+\s+)?[a-z]?\d{3,4}[a-z]{0,2}\b"
    r"|\bathlon\s+(?:64\s+)?(?:x2\s+)?[a-z]?\d{3,4}\+?\b"
    r"|\bphenom\s+(?:ii\s+)?x[2346]\s+\d{3,4}\b"
    r"|\ba(?:1|4|6|8|10)-\d{4}[a-z]{0,2}\b"
    r"|\bfx-\d{4}[a-z]{0,2}\b"
    r"|\bsnapdragon\s+(?:\d{3}\w?|\d\s+gen\s*\d|x\s+(?:elite|plus))\b"
    r"|\bm[1-4]\s*(?:pro|max|ultra)\b"
    r"|\ba\d{1,2}\s+(?:bionic|fusion)\b"
    r"|\b(?:geforce\s+)?(?:gtx|rtx)\s*\d{3,4}\s*(?:ti|super|max-q|d)?\b"
    r"|\b(?:geforce\s+)?(?:gts|gt)\s+\d{3,4}\s*(?:ti|super)?\b(?!\s*(?:mb|gb|mhz))"
    r"|\bgeforce\s+\d{1,4}(?:\s*(?:mx|gts|gt|ti|ultra|le|gs|gtx))?\b"
    r"|\bradeon\s+(?:hd\s*)?\d{3,5}[a-z]{0,2}(?:\s*(?:pro|xt|xtx|le|se))?\b"
    r"|\bradeon\s+(?:rx\s*)?\d{3,4}[a-z]{0,2}(?:\s+(?:xtx|xt|gre|m|s|pro|gold|silver|ultra))?\b"
    r"|\brx\s+\d{3,4}[a-z]{0,2}(?:\s+(?:xtx|xt|gre|m|s))?\b"
    r"|\barc\s+[abg]\d{3}[a-z]{0,2}\b"
    r"|\bquadro\s+(?:rtx\s+)?[a-z]?\d{3,4}[a-z]{0,3}\b"
    r"|\btesla\s+[a-z]\d{2,4}\b"
    # 使用者常常只打後面的數字字母組合：「12700K 幾核」「5600X 多少瓦」「3080Ti 多少記憶體」。
    # 純數字（1080、2024）不在此列——型號別名一律含字母，查不到就是查不到，不亂猜。
    r"|\b(?:[a-z]{1,3}-?)?\d{3,5}[a-z]{1,3}\b)", re.I)


def _norm(s):
    return re.sub(r"[^0-9a-f]", "", str(s or "").lower())


class EntityStore:
    """唯讀的 SQLite 包裝。連線照執行緒各自開——標準函式的 ThreadingHTTPServer 每個
    請求一個執行緒，共用一條 sqlite 連線會撞 `SQLite objects created in a thread...`。
    """

    def __init__(self, path=DEFAULT_DB):
        self.path = os.path.abspath(path)
        self.available = os.path.exists(self.path)
        self._local = None

    def _conn(self):
        import threading
        if self._local is None:
            self._local = threading.local()
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    # ── 單筆查詢 ────────────────────────────────────────────────────
    def pci_vendor(self, vid):
        cur = self._conn().execute("SELECT * FROM pci_vendor WHERE id=?", (_norm(vid),))
        return cur.fetchone()

    def pci_device(self, vid, did):
        con = self._conn()
        row = con.execute(
            "SELECT * FROM pci_device WHERE vendor_id=? AND device_id=?",
            (_norm(vid), _norm(did))).fetchone()
        if row:
            return row
        # 有些人的寫法是 device:vendor，順序相反
        return con.execute(
            "SELECT * FROM pci_device WHERE vendor_id=? AND device_id=?",
            (_norm(did), _norm(vid))).fetchone()

    def usb_vendor(self, vid):
        cur = self._conn().execute("SELECT * FROM usb_vendor WHERE id=?", (_norm(vid),))
        return cur.fetchone()

    def usb_device(self, vid, pid):
        con = self._conn()
        row = con.execute(
            "SELECT * FROM usb_device WHERE vendor_id=? AND device_id=?",
            (_norm(vid), _norm(pid))).fetchone()
        if row:
            return row
        return con.execute(
            "SELECT * FROM usb_device WHERE vendor_id=? AND device_id=?",
            (_norm(pid), _norm(vid))).fetchone()

    def oui(self, prefix):
        p = _norm(prefix)[:6].upper()
        if len(p) != 6:
            return None
        return self._conn().execute("SELECT * FROM oui WHERE prefix=?", (p,)).fetchone()

    def pnp(self, hid):
        h = str(hid or "").strip().upper()
        return self._conn().execute("SELECT * FROM pnp_id WHERE id=?", (h,)).fetchone()

    def gpu_device(self, did):
        cur = self._conn().execute(
            "SELECT * FROM gpu_device WHERE device_id=? LIMIT 5", (_norm(did),))
        return cur.fetchall()

    def cve(self, cid):
        return self._conn().execute(
            "SELECT * FROM cve WHERE cve=?", (str(cid).upper(),)).fetchone()

    def software(self, product):
        cur = self._conn().execute(
            "SELECT * FROM software_cycle WHERE lower(product)=lower(?) "
            "ORDER BY release_date DESC LIMIT 8", (str(product).strip(),))
        return cur.fetchall()

    def search_vendor(self, name, limit=8):
        """反查：名稱 → 有哪些 ID。查不到就空手回去，不要猜。"""
        like = f"%{name}%"
        con = self._conn()
        out = []
        for table, cols in (
            ("pci_vendor", "id, vendor"),
            ("usb_vendor", "id, vendor"),
            ("oui", "prefix, vendor"),
        ):
            for row in con.execute(
                    f"SELECT {cols} FROM {table} WHERE vendor LIKE ? LIMIT ?", (like, limit)):
                out.append((table, tuple(row)))
        return out[:limit]

    # ── CPU／GPU 型號查表 ──────────────────────────────────────────
    @staticmethod
    def _model_norm(s):
        s = re.sub(r"[®™©]", "", str(s or "")).lower()
        return re.sub(r"[^0-9a-z]+", "", s)

    @classmethod
    def _model_aliases(cls, token):
        """同一個型號的不同寫法：全名、去掉品牌前綴、以及尾巴的數字（12700K）。"""
        t = re.sub(r"\s+", " ", token.strip())
        out = []
        for cand in (t, re.sub(r"^(?:intel|amd|nvidia|core\s*[23]?|geforce|radeon)\s+",
                               "", t, flags=re.I)):
            k = cls._model_norm(cand)
            if len(k) >= 4 and k not in out:
                out.append(k)
        return out

    # ── 主機板／CPU 關聯查詢（組件庫，CC BY-NC-SA，預設不載入）──────
    def _components(self):
        """惰性載入。授權是 NC-SA：只有 XINSPECT_ALLOW_NC=1 才碰這份資料。"""
        if getattr(self, "_comp", None) is not None:
            return self._comp
        self._comp = False
        if os.environ.get("XINSPECT_ALLOW_NC") != "1":
            return False
        try:
            sys.path.insert(0, HERE)
            import component_query
            self._comp = component_query.load()
        except Exception:
            self._comp = False
        return self._comp

    def component_facts(self, question):
        """把「哪些板搭哪些 CPU／超頻／擴充／世代」翻成查詢結果（不是給模型猜）。

        ⚠ 兩段授權要分開：晶片組那一列在 entities.db（Wikipedia 事實欄，CC BY-SA），
        不需要 NC 授權；「這顆晶片組有幾張板」要組件庫（CC BY-NC-SA），才受閘門限制。
        從前把晶片組查詢一起關在 NC 閘門後面，結果是**沒有 NC 資料的正式站**問
        「B650 支援哪些 CPU 世代」時整題落回文件檢索，答出一段無關的 CPU 章節。
        """
        q = question
        low = q.lower()
        facts = []

        # 點名晶片組（B650／Z790…）→ 直接抄支援表。這一段只讀 entities.db，
        # 所以在閘門之前就要跑；chipset_facts 內的組件庫補充會自己判斷 comp 是否可用。
        if CHIPSET_TOKEN.search(q):
            facts.extend(self.chipset_facts(q))

        comp = self._components()
        if not comp:
            return facts

        # 問工作站世代 → 數出每代板況（要組件庫才有「幾張板」）。
        if re.search(r"工作站|伺服器|server|hedt", low) or (
                re.search(r"演進|進化", low) and re.search(r"主機板|主板|board", low)):
            facts.extend(self.workstation_facts())

        board_word = re.search(r"主機板|主板|motherboard|\bboard\b|板子|搭配|相容|兼容|插槽|socket", low)
        exp_word = re.search(r"u\.?2|m\.?2|多.?gpu|雙.?gpu|\bsli\b|crossfire|pcie", low)
        oc_word = re.search(r"超頻|overclock|\boc\b", low)
        total_word = re.search(r"哪些|多少|幾張|幾款|同時|同時滿足|全部|列出", q)

        if exp_word and total_word:
            m2m = re.search(r"m\.?2[^0-9]{0,8}(\d+)", low)
            m2 = int(m2m.group(1)) if m2m else 2
            u2 = 1 if re.search(r"u\.?2", low) else 0
            x16 = 2 if re.search(r"多.?gpu|雙.?gpu|sli|crossfire", low) else 1
            gen5 = bool(re.search(r"pcie\s*5|gen\s*5", low))
            ecc = bool(re.search(r"\becc\b", low))
            rows, funnel, top = comp.expansion(x16_min=x16, m2_min=m2, u2_min=u2,
                                               gen5=gen5, ecc=ecc, exclude_flagged=True)
            detail = "；".join(f"{b['model']}（{comp.describe(b)}）" for b in top[:5])
            facts.append({
                "kind": "board_filter",
                "id": f"x16>={x16},m2>={m2},u2>={u2}",
                "name": "主機板擴充條件篩選",
                "text": (f"依組件庫 {len(comp)} 張主機板篩選：條件 x16≥{x16}、M.2≥{m2}"
                         + (f"、U.2≥{u2}" if u2 else "") + ("、PCIe 5.0" if gen5 else "")
                         + ("、ECC" if ecc else "")
                         + f" → 符合 {len(rows)} 張。"
                         + (f"例：{detail}" if detail else "沒有符合的板子。")
                         + f"（全庫：x16≥2 有 {funnel['x16>=2']} 張、U.2>0 有 {funnel['u2>0']} 張、"
                           f"M.2≥3 有 {funnel['m2>=3']} 張；已排除 PCIe 世代被來源模板預填的 "
                           f"{funnel['PCIe 世代被模板預填（不可信）']} 張）"),
                "license": "CC-BY-NC-SA-4.0", "url": "",
            })

        if oc_word:
            import component_query as cq
            # 和查詢層共用同一套規則，避免兩處口徑不一致（先前 852 vs 932）
            intel = sum(1 for b in comp.boards
                        if cq.INTEL_OC.search(b["chipset"] or "") and not b["platform_is_amd"])
            bx = sum(1 for b in comp.boards if b["platform_is_amd"]
                     and re.search(r"^AMD\s+[BX]\d{3}", b["chipset"] or "", re.I))
            a_series = sum(1 for b in comp.boards if b["platform_is_amd"]
                           and re.search(r"^AMD\s+A\d{3}", b["chipset"] or "", re.I))
            facts.append({
                "kind": "board_oc", "id": "oc", "name": "超頻條件",
                "text": (f"組件庫裡可判定「允許改 CPU 倍頻」的主機板：Intel {intel} 張"
                         f"（Z／X299／W 系列）；AMD {bx} 張（B／X 系列；另有 A320／A520／A620 共 "
                         f"{a_series} 張官方不支援 CPU 倍頻，只能超記憶體）。"
                         "資料**沒有 VRM 相數／散熱模組**欄位，所以「完整超頻條件」無法只靠這份資料判定——"
                         "能確定的是晶片組是否開放倍頻、有沒有 8-pin 以上 CPU 供電接頭。"),
                "license": "CC-BY-NC-SA-4.0", "url": "",
            })

        if board_word:
            # 問句裡有具體 CPU 型號 → 先解插槽再列板子
            for tok in RE_HW_MODEL.finditer(q):
                r = comp.boards_for_cpu(tok.group(0))
                if r.get("found"):
                    inferred = "推定" in str(r.get("from") or "")
                    facts.append({
                        "kind": "compatibility",
                        "id": r["socket"],
                        "name": f"{tok.group(0)} 的主機板",
                        "text": (f"{tok.group(0)} 的插槽是 {r['socket']}"
                                 + ("（依世代規則推定：型號庫那一列沒有 socket 欄，"
                                    "桌機世代與插槽一對一，所以可以推定；行動版不推）"
                                    if inferred else "")
                                 + f"，組件庫裡同插槽主機板 "
                                 f"{r['board_count']} 張（晶片組 {len(r['chipsets'])} 種："
                                 f"{'、'.join(r['chipsets'][:8])}）。例：{'；'.join(r['examples'][:5])}"
                                 + (f"。保留：{r['note']}" if r.get("note") else "")),
                        "license": "CC-BY-NC-SA-4.0",
                        "url": r.get("url") or "",
                    })
                    break
            else:
                if re.search(r"哪些.*世代|支援哪些|世代.*哪些", q):
                    line = comp.socket_lineage()
                    facts.append({
                        "kind": "board_sockets", "id": "sockets", "name": "插槽分佈",
                        "text": ("組件庫主機板的插槽分佈（前 10）："
                                 + "、".join(f"{x['socket']} {x['boards']} 張" for x in line[:10])
                                 + "｜同插槽不等於同世代相容，例：LGA1151 有 6/7 代與 8/9 代兩種電氣規格、"
                                   "LGA2011 有 2011/2011-3/2011-v4，AM4 跨 Ryzen 1000–5000 需要 BIOS。"),
                        "license": "CC-BY-NC-SA-4.0", "url": "",
                    })
                    gen = self.chipset_generations()
                    if gen:
                        facts.append(gen)
        return facts[:4]

    def chipset_generations(self, limit=8):
        """晶片組 → 支援的 CPU 世代（Wikipedia 晶片組表的『CPU support …』欄，CC BY-SA）。

        這是最誠實的答案來源：來源自己逐世代標 Yes／Varies／No，
        我們不用猜「同插槽所以相容」。
        """
        if not self.available:
            return None
        con = self._conn()
        try:
            rows = con.execute(
                """SELECT model, headers, cells, url FROM hw_model
                   WHERE kind IN ('lib-chipset','wiki-chipset') ORDER BY model LIMIT 700""").fetchall()
        except sqlite3.Error:
            return None
        out = []
        for model, headers, cells, url in rows:
            d = dict(zip(json.loads(headers or "[]"), json.loads(cells or "[]")))
            sup = {k.split("CPU support ")[-1].replace("原文:", "").strip(): v
                   for k, v in d.items() if "CPU support" in k}
            # 只要「逐世代 Yes／Varies／No」那種；舊晶片組的「CPU support Athlon 64, Sempron」
            # 是一串型號清單，格式不同，放後面就好
            graded = {g: v for g, v in sup.items() if re.fullmatch(r"[YN]\w*", str(v) or "")}
            if graded:
                out.append(f"{model}：" + "、".join(
                    f"{g} {v}" for g, v in list(graded.items())[:6]))
        if not out:
            return None
        return {
            "kind": "chipset_generation", "id": "chipset-gen",
            "name": "晶片組支援的 CPU 世代",
            "text": "來源逐世代標記（Yes／Varies／No），節選：" + "；".join(out[:limit]),
            "license": "CC-BY-SA-4.0", "url": "",
        }

    def chipset_facts(self, question):
        """問句裡點名了晶片組（B650／Z790／X870E）→ 直接抄那一列的支援表。

        「B650 支援哪些 CPU 世代」不能靠檢索：整份語料都在講 B650，模型會挑一段順口的。
        型號庫裡這一列逐項寫著 Zen 4／Zen 5 是 Yes，還寫著能不能超頻、PCIe 5.0 有沒有，
        照抄比生成可靠。組件庫那一側再補上「這個晶片組在市面上有幾張板、都是什麼插槽」。
        """
        out = []
        comp = self._components()
        for m in CHIPSET_TOKEN.finditer(question or ""):
            name = m.group(0).upper()
            if name in ARC_GPU_NAMES:
                continue
            row = None
            if self.available:
                con = self._conn()
                try:
                    row = con.execute(
                        "SELECT model, headers, cells, url FROM hw_model"
                        " WHERE norm=? AND kind IN ('lib-chipset','wiki-chipset')"
                        " ORDER BY CASE kind WHEN 'lib-chipset' THEN 0 ELSE 1 END LIMIT 1",
                        (name.lower(),)).fetchone()
                except sqlite3.Error:
                    row = None
            if not row:
                continue
            d = dict(zip(json.loads(row[1] or "[]"), json.loads(row[2] or "[]")))
            sup = {k.split("CPU support")[-1].strip(" /:").replace("原文:", "").strip(): v
                   for k, v in d.items() if "CPU support" in k}
            oc = [v for k, v in d.items() if re.search(r"overclock|超頻", k, re.I)]
            pcie5 = [f"{k.split('/')[-1].strip()}: {v}" for k, v in d.items()
                     if re.search(r"PCIe 5\.0", k, re.I)]
            ecc = [v for k, v in d.items() if re.search(r"\bECC\b", k, re.I)]
            boards, socks = 0, {}
            if comp:
                hit = [b for b in comp.boards
                       if re.search(rf"\b{re.escape(name)}\b", b["chipset"] or "", re.I)]
                boards = len(hit)
                socks = dict(Counter(b["socket"] for b in hit if b["socket"]))
            parts = [f"CPU 支援（來源逐項）：{'、'.join(f'{g} {v}' for g, v in sup.items())}"
                     if sup else None]
            if not sup:
                # Intel 頁沒有「逐世代」欄位，但有 architecture／platform——那是這顆晶片組
                # 出道時搭配的世代，講「支援」會太滿，照抄成「來源標示的架構」。
                arch = next((v for k, v in d.items()
                             if re.fullmatch(r"(architecture|platform|microarchitecture)", k, re.I)), "")
                if arch:
                    parts = [f"來源標示架構：{arch}（這是晶片組出道時搭配的世代，"
                             "向下相容要看主機板 BIOS，不等於完整支援清單）"]
                else:
                    parts = ["來源沒有 CPU 支援或架構欄位"]
            if oc:
                parts.append(f"可超頻：{'、'.join(oc[:2])}")
            if pcie5:
                parts.append("PCIe 5.0：" + "、".join(pcie5[:3]))
            if ecc:
                parts.append(f"ECC：{ecc[0]}")
            if comp:
                parts.append(f"組件庫中 {boards} 張板"
                             + (f"（插槽 {'、'.join(f'{k} {v} 張' for k, v in socks.items())}）"
                                if socks else "——市面上沒有這個晶片組的板"))
            out.append({
                "kind": "chipset_support", "id": f"chipset-{name}",
                "name": f"{name} 的支援與規格", "text": "；".join(parts) + "。",
                "license": "CC-BY-SA-4.0（型號庫）／CC-BY-NC-SA-4.0（板數統計）",
                "url": row[3] or "",
            })
        return out[:2]

    def workstation_facts(self):
        """工作站／HEDT 主機板的世代演進：每代幾張板、什麼記憶體、擴充與 PCIe 世代。"""
        comp = self._components()
        if not comp:
            return []
        gens = [g for g in comp.workstation_evolution() if g["boards"]]
        if not gens:
            return []
        lines = []
        for g in gens:
            y = f"{g['years'][0]}–{g['years'][1]}" if g.get("years") else "年份不明"
            lines.append(
                f"{g['socket']}（{g['chipsets']}；{y}）{g['boards']} 張，{g['ram']}，"
                f"x16 最多 {g['x16_max']}、M.2 最多 {g['m2_max']}、U.2 {g['u2']} 張、"
                f"PCIe 5.0 {g['gen5']} 張")
        missing = [g["socket"] for g in comp.workstation_evolution() if not g["boards"]]
        return [{
            "kind": "workstation_generation", "id": "ws-gen",
            "name": "工作站主機板的世代演進",
            "text": ("按世代（插槽→配套晶片組）數出來的組件庫實況：" + "；".join(lines)
                     + (f"。組件庫沒有收錄的世代：{'、'.join(missing)}——"
                        "這幾代在零售組件庫本來就少，不是不存在，查詢時要說明資料缺口" if missing else "")
                     + "。演進主軸：記憶體 DDR3→DDR4→DDR5、PCIe 3.0→4.0→5.0、"
                       "高階平台從『需两颗 CPU 才吃滿通道』走向單顆 64 條 PCIe 5.0。"),
            "license": "CC-BY-NC-SA-4.0", "url": "",
        }]

    @staticmethod
    def _pair_label(h, c):
        """(標籤, 要不要留)。欄名與值明顯不符時，留標籤就是在教模型講錯的話。

        這一層是「把整列照抄給模型看」，所以標籤的正確性比完整性重要：
        lib 的 GPU 列有整欄錯位（`process_nm` 那格其實是 TFLOPS、`release_date` 那格是價格），
        講成「製程 73.236」比不講更糟；但 `boost_clock_ghz` 的 2235 是 MHz，
        換個標籤就還是好資料。
        """
        s = str(c or "").strip()
        hl = str(h or "").lower()
        num = re.search(r"(\d+(?:\.\d+)?)", s.replace(",", ""))
        v = float(num.group(1)) if num else None
        if "ghz" in hl and v is not None and v > 20:
            return re.sub(r"_?ghz", "_mhz", str(h).strip(), flags=re.I), True
        if "process" in hl or "制程" in hl or "製程" in str(h):
            if v is None or not (1 <= v <= 20) and not re.search(r"nm|奈米|纳米", s, re.I):
                return "", False
        if "tdp" in hl and re.match(r"^\$|usd", s, re.I):
            return "", False
        if ("release" in hl or "launch_date" in hl) and re.match(r"^\$|^usd|^\d{3,4}$", s, re.I) \
                and not re.search(r"(19|20)\d{2}", s):
            return "", False
        return str(h).strip(), True

    def hw_models(self, question, limit=2):
        """回傳 list[dict]：查到就照抄那一列的原文，不做解讀。"""
        if not self.available:
            return []
        con = self._conn()
        out, seen = [], set()

        for m in RE_HW_MODEL.finditer(question):
            for alias in self._model_aliases(m.group(0)):
                if alias in seen:
                    continue
                seen.add(alias)

                row = None
                try:
                    rows = con.execute(
                        """SELECT a.kind, a.model, m.page, m.headers, m.cells
                           FROM hw_alias a
                           LEFT JOIN hw_model m ON m.row_id=a.row_id AND m.kind=a.kind
                           WHERE a.alias=? AND (a.kind LIKE 'wiki-%' OR a.kind LIKE 'lib-%')
                           LIMIT 4""", (alias,)).fetchall()
                except sqlite3.Error:
                    rows = []
                rows = [r for r in rows if r["page"]]
                if rows:
                    # 同一型號可能出現在「家族總表」（一格寫著整組 SKU）、「分段包裝列」
                    # 與「逐顆规格表」。後者才有這顆自己的核心數／TDP，優先選它。
                    def _cells_of(r):
                        return json.loads(r["cells"] or "[]")

                    def _is_own(r):
                        """這一列是「這顆自己的」嗎：型號字出現在某格（含尾端縮寫）。"""
                        full = self._model_norm(r["model"] or "")
                        for c in _cells_of(r):
                            cn = self._model_norm(c)
                            if len(cn) < 4:
                                continue
                            # 「以型號開頭」只認短註記（「(Navi 31)」）；
                            # 一整組 SKU 的家族格（「Ryzen 5 (5600X), Ryzen 7 …」）不算。
                            if cn == full or full.endswith(cn) or (
                                    len(cn) >= 6 and cn.startswith(full)
                                    and len(cn) <= len(full) + 24):
                                return True
                        return False

                    def _aligned(r):
                        """欄名與值數量對得上，才代表「這格是什麼」是可信的。"""
                        return 1 if len(json.loads(r["headers"] or "[]")) == len(_cells_of(r)) else 0

                    def _groupish(cells):
                        """家族列：某一格把整組 SKU 寫在一起（`Ryzen 5 (5600X), Ryzen 7 (5800X)`）。

                        不能再用 `_is_own` 當「是不是家族列」的判準——lib 的列根本沒有型號欄
                        （型號在 hw_model.model 裡），任何 lib 列都會被判成家族列，
                        於是欄名被丟掉、回答變成一串沒有標籤的數字。
                        """
                        pat = re.compile(
                            r"(?:Ryzen|Core|Xeon|Threadripper|EPYC|GeForce|Radeon|Celeron|Pentium)"
                            r"\s*[A-Za-z]*\s*[（(]?\s*[A-Z]{0,2}\d{3,5}[A-Za-z]{0,4}\s*[)）]?"
                            r"\s*[,，、]\s*(?:[A-Za-z]+\s*){0,3}\d{3,5}", re.I)
                        return any(pat.search(str(c or "")) for c in cells)

                    # 排序：欄位對齊 > 是這顆自己的列 > 內容多。
                    # 先前只看「內容多」，結果同一型號會挑到欄位錯位的 wiki 列
                    # （表頭 10 格、值 12 格），回答就把 4.5 GHz 說成 TDP 4.5 W。
                    row = max(rows, key=lambda r: (_aligned(r), 1 if _is_own(r) else 0,
                                                   len(_cells_of(r))))
                    headers = json.loads(row["headers"] or "[]")
                    raw_cells = _cells_of(row)
                    # 家族列（一格寫著整組 SKU）不是這顆的規格，回答時要講清楚，
                    # 否則「5600X 幾核心」會拿到整個 Vermeer 世代的 6/8/12/16 核。
                    is_group = _groupish(raw_cells)
                    if len(headers) == len(raw_cells) and raw_cells:
                        pairs, dropped = [], 0
                        for h, c in zip(headers, raw_cells):
                            if not str(c or "").strip():
                                continue
                            label, keep = self._pair_label(h, c)
                            if not keep:
                                dropped += 1
                                continue
                            h, keep2 = label, True
                            c = str(c)
                            # lib 的 CPU 列把「8 (16)」放在名叫 `threads` 的欄裡，
                            # 直接寫「threads 8 (16)」會讓人（和模型）讀成 8 執行緒。
                            if h.lower() == "threads" and re.fullmatch(
                                    r"\d{1,4}\s*\(\s*\d{1,4}\s*\)", str(c).strip()):
                                h = "cores (threads)"
                            # 同一批列的第一欄常被標成 `process_nm`，值卻是品牌名
                            # （`Core i7`、`Ryzen 7`）——照寫會變成「製程 Core i7」。
                            elif re.search(r"(process|architecture|arch)", h, re.I) \
                                    and re.match(r"^(Core|Ryzen|Xeon|Celeron|Pentium|Athlon|"
                                                 r"FX|EPYC|Threadripper|Snapdragon)\b",
                                                 str(c).strip()):
                                h = "brand"
                            pairs.append((h, c))
                        body = "；".join(f"{h} {c}" for h, c in pairs)
                        if dropped:
                            body += f"（另有 {dropped} 個欄位欄名與值不符，已略過）"
                        cells = [c for _h, c in pairs]
                    else:
                        cells = [c for c in raw_cells if str(c or "").strip()]
                        body = "；".join(cells)
                    page = row["page"]
                    note = "（此列為系列／家族資料，非單顆型號的規格）" if is_group else ""
                    if len(body) > 600:          # 表格列有時很長；留前段就夠回答
                        body = body[:600].rstrip("；") + "…"
                    out.append({
                        "kind": "cpu_model" if not GPU_WORD.search(row["model"] or "")
                        else "gpu_model",
                        "id": row["model"],
                        "name": row["model"],
                        "text": (f"{row['model']}（Wikipedia：{page.replace('_', ' ')}）"
                                 f"{note}：{body}"),
                        "license": "CC-BY-SA-4.0",
                        "url": "https://en.wikipedia.org/wiki/" + page.replace("_", " "),
                    })
                    break

                row = None
                try:
                    row = con.execute(
                        """SELECT a.kind, a.model, a.row_id, a.vendor
                           FROM hw_alias a WHERE a.alias=? AND a.kind IN ('cpu','gpu')
                           LIMIT 1""", (alias,)).fetchone()
                except sqlite3.Error:
                    row = None
                if row:
                    table = "hw_cpu" if row["kind"] == "cpu" else "hw_gpu"
                    try:
                        d = con.execute(f"SELECT * FROM {table} WHERE row_id=?",
                                        (row["row_id"],)).fetchone()
                    except sqlite3.Error:
                        d = None
                    if d:
                        d = dict(d)
                        if table == "hw_cpu":
                            body = "；".join(f"{k} {d.get(k)}" for k in (
                                "year", "vendor", "family", "segment", "uarch", "process",
                                "socket", "cores", "threads", "tdp", "status", "note") if d.get(k))
                        else:
                            body = "；".join(f"{k} {d.get(k)}" for k in (
                                "year", "vendor", "family", "segment", "uarch", "chip", "vram",
                                "interface", "status", "note") if d.get(k))
                        out.append({
                            "kind": "cpu_model" if table == "hw_cpu" else "gpu_model",
                            "id": d.get("model"),
                            "name": d.get("model"),
                            "text": (f"{d.get('vendor', '')} {d.get('model', '')}"
                                     f"（型號資料庫，證據等級：{d.get('evidence') or '未標'}）：{body}"),
                            "license": "user-provided-compilation",
                            "url": d.get("url") or "",
                        })
                        break
            if len(out) >= limit:
                break
        return out[:limit]

    # ── 一句話解析：把問題裡的 ID 找出來並查 ────────────────────────
    def resolve(self, question):
        """回傳 list[dict]：每一筆都是「查到的確定事實」。

        沒有查到就回空 list——呼叫端必須能區分「查了、沒這筆」與「沒問題可查」。
        """
        if not self.available:
            return []
        q = question
        found = []
        con = self._conn()

        # 型號（i7-12700K、RTX 3080…）先查，因為它比 4 碼十六進位 ID 更長更具體。
        found.extend(self.hw_models(q))

        # 主機板／CPU 的關聯問題（哪些板搭哪些 CPU、超頻、擴充、工作站世代）。
        # 這批資料在 entities-nc.db（CC BY-NC-SA），只有 XINSPECT_ALLOW_NC=1 時才載入。
        found.extend(self.component_facts(q))

        m = RE_CVE.search(q)
        if m:
            row = self.cve(m.group(0))
            if row:
                found.append({
                    "kind": "cve",
                    "id": row["cve"],
                    "name": row["name"],
                    "text": (f"{row['cve']}：{row['name']}。廠商：{row['vendor']}；"
                             f"加入 KEV 日期：{row['date_added']}；勒索軟體關聯：{row['ransomware']}。"),
                    "license": row["license"],
                    "url": f"https://nvd.nist.gov/vuln/detail/{row['cve']}",
                })

        # PCI / USB 的 vendor:device 對
        for m in RE_PCI_PAIR.finditer(q):
            a, b = m.group(1), m.group(2)
            row = self.pci_device(a, b)
            if row:
                found.append({
                    "kind": "pci_device",
                    "id": f'{row["vendor_id"]}:{row["device_id"]}',
                    "name": f'{row["vendor"]} {row["name"]}',
                    "text": (f'PCI {row["vendor_id"]}:{row["device_id"]} = '
                             f'{row["vendor"]} {row["name"]}'),
                    "license": row["license"],
                    "url": None,
                })
                continue
            urow = self.usb_device(a, b)
            if urow:
                found.append({
                    "kind": "usb_device",
                    "id": f'{urow["vendor_id"]}:{urow["device_id"]}',
                    "name": f'{urow["vendor"]} {urow["name"]}',
                    "text": (f'USB {urow["vendor_id"]}:{urow["device_id"]} = '
                             f'{urow["vendor"]} {urow["name"]}'),
                    "license": urow["license"],
                    "url": None,
                })

        # MAC OUI（先於單一 4 碼 ID，因為 OUI 更長更具體）
        for m in RE_OUI.finditer(q):
            row = self.oui(m.group(1))
            if row:
                found.append({
                    "kind": "oui",
                    "id": row["prefix"],
                    "name": row["vendor"],
                    "text": f'MAC 前綴 {row["prefix"]} = {row["vendor"]}（{row["registry"]}）',
                    "license": row["license"],
                    "url": None,
                })

        # Windows 硬體 ID 形式：PCI\VEN_10DE&DEV_1B80、ACPI\PNP0A08
        for m in RE_HWID.finditer(q):
            kind, code = m.group(1).upper(), m.group(2).upper()
            if kind == "PCI" and code.startswith("VEN_"):
                row = self.pci_vendor(code[4:])
                if row:
                    found.append({
                        "kind": "pci_vendor", "id": row["id"], "name": row["vendor"],
                        "text": f'PCI 廠商 {row["id"]} = {row["vendor"]}',
                        "license": row["license"], "url": None,
                    })
            elif kind in ("ACPI", "PNP", "HID"):
                # 兩種寫法都要認：
                #   ACPI\PNP0A08   → 前三碼是廠商（PNP = Microsoft），後面是裝置
                #   ACPI\AUI1000   → 前四碼是廠商代碼（AUI）
                # pnp.ids 這份表只有「廠商前綴」，沒有逐裝置名稱；所以查得到廠商、
                # 查不到「PNP0A08 是 PCIe 根複合體」——那是 ACPI 規格的事，不是這張表。
                row = self.pnp(code) or self.pnp(code[:3]) or self.pnp(code[:4])
                if row:
                    found.append({
                        "kind": "pnp_id", "id": code, "name": row["vendor"],
                        "text": (f'PNP/ACPI 硬體 ID {code} 的廠商前綴 = {row["id"]}，'
                                 f'廠商 = {row["vendor"]}（逐裝置名稱不在這張表裡）'),
                        "license": row["license"], "url": None,
                    })

        # 單一 4 碼 ID：只有在問題明白問「廠商」時才當 PCI vendor 查，
        # 否則 `10de` 這種字串在任何句子裡都可能出現，亂猜會製造假答案
        if not found and RE_VENDOR_WORD.search(q):
            for m in RE_HEX4.finditer(q):
                code = m.group(1).lower()
                row = self.pci_vendor(code)
                if row:
                    found.append({
                        "kind": "pci_vendor", "id": row["id"], "name": row["vendor"],
                        "text": f'PCI 廠商 {row["id"]} = {row["vendor"]}',
                        "license": row["license"], "url": None,
                    })
                    break
                row = self.usb_vendor(code)
                if row:
                    found.append({
                        "kind": "usb_vendor", "id": row["id"], "name": row["vendor"],
                        "text": f'USB 廠商 {row["id"]} = {row["vendor"]}',
                        "license": row["license"], "url": None,
                    })
                    break

        # 去重（同一筆可能被兩個樣式命中）
        seen = set()
        out = []
        for f in found:
            key = (f["kind"], f["id"])
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def stats(self):
        if not self.available:
            return {}
        con = self._conn()
        out = {}
        for t in ("pci_vendor", "pci_device", "usb_vendor", "usb_device", "oui",
                  "pnp_id", "gpu_device", "cve", "software_cycle", "drive_model"):
            try:
                out[t] = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            except Exception:
                pass
        return out


if __name__ == "__main__":
    import json
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    store = EntityStore()
    print("db:", store.path, "available:", store.available)
    print(json.dumps(store.stats(), indent=2, ensure_ascii=False))

    for q in ["10de:1b80 是什麼裝置？", "PCI vendor ID 8086 是哪家？",
              "MAC 前綴 286FB9 是哪家網卡？", "CVE-2026-97229 是什麼？",
              "ACPI\\PNP0A08 是什麼？", "0x10DE 是誰的？",
              "曦覽會不會上傳資料？"]:
        hits = store.resolve(q)
        print(f"\n{q}")
        for h in hits:
            print(f"   [{h['kind']}] {h['text']}")
        if not hits:
            print("   （查無）")
