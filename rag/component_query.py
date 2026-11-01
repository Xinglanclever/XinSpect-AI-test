#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""主機板／CPU 的結構化查詢：相容性、超頻、擴充、工作站世代。

為什麼不能交給檢索或模型：
  「哪些主機板能同時滿足雙 x16＋U.2＋3 個 M.2」是一個 **filter 查詢**，
  不是「哪一段文字談到 U.2」。BM25 只會找回談過 U.2 的文章，
  模型會把「談過」當成「符合」。所以這裡一律走 SQL／記憶體過濾，
  數字是數出來的，不是生成的。

資料在兩個 DB：
  entities.db     —— 型號庫（CPU／GPU／晶片組；CC BY-SA）
  entities-nc.db  —— v4 組件庫（主機板／電源／散熱／風扇；CC BY-NC-SA，非商業）

老實話優先：
  * 主機板的 socket 分佈很完整，但 **CPU 那側只有部分記錄帶 socket 欄**，
    找不到就說找不到，不用「同名插槽」猜相容性。
  * LGA1151、LGA2011 這類「同一名字、不同代」的插槽，資料本身分不出 v1／v2，
    所以這種相容性問題一定要留保留條件，不能一口氣答「全部支援」。
  * `pcie_version` 在晶片組表有大量的值是 `Yes`（來源欄位被錯映到布林欄），
    所以 PCIe 世代一律以主機板的 `pcie_summary` 為準，不用晶片組那欄。
"""

import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_DB = os.path.abspath(os.path.join(HERE, "..", "corpus", "entities.db"))
NC_DB = os.path.abspath(os.path.join(HERE, "..", "corpus", "entities-nc.db"))

# 超頻：Intel 只有 Z／W 系列能改倍頻（B／H 僅記憶體），AMD AM4／AM5 全晶片組開放倍頻。
INTEL_OC = re.compile(r"\b(?:Z[3-9]70|Z[3-9]90|Z68|Z77|Z87|Z97|Z170|Z270|X299|W[468]80|W680)\b", re.I)
AMD_OC = re.compile(r"\b(?:X\d{3}|B\d{3}[EX]?|A\d{3}[BDX]?)\b", re.I)

SOCKET_NORM = re.compile(r"[^0-9a-z]+")


def snorm(s):
    """插槽名正規化：`LGA 1700`＝`LGA1700`、`Socket AM5`＝`AM5`。"""
    t = SOCKET_NORM.sub("", str(s or "").lower())
    return re.sub(r"^(?:socket|lga)", "", t) if t.startswith("socket") else t


def to_int(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return 0


def count_slots(summary, width):
    """`4× PCIe 4.0 x16; 1× PCIe 4.0 x1` → x16 有 4 個。

    直接數字串 `x16` 會把「4× PCIe 4.0 x16」算成 1 個——工作站板動輒 3–4 條 x16，
    這一算錯，『雙 x16』的篩選就會把整批板子排除掉。乘數寫在前面的才是數量。
    """
    total = 0
    for seg in re.split(r"[;,]", str(summary or "")):
        if not re.search(r"\bx" + str(width) + r"\b", seg, re.I):
            continue
        m = re.match(r"\s*(\d+)\s*[×x]", seg)
        total += int(m.group(1)) if m else 1
    return total


class Components:
    """記憶體裡的組件索引：載入一次（3,684 板＝2.4 MB），查詢走純 Python。"""

    def __init__(self, nc_db=NC_DB, main_db=MAIN_DB):
        self.boards = self._load_boards(nc_db)
        self.sockets = Counter(b["socket"] for b in self.boards if b["socket"])
        self.chipsets = Counter(b["chipset"] for b in self.boards if b["chipset"])
        self.boards_by_socket = defaultdict(list)
        for b in self.boards:
            if b["socket_key"]:
                self.boards_by_socket[b["socket_key"]].append(b)
        # 晶片組 → 插槽（從板子自己推，不靠記憶）
        self.chipset_socket = defaultdict(Counter)
        for b in self.boards:
            if b["chipset"] and b["socket_key"]:
                self.chipset_socket[b["chipset"]][b["socket_key"]] += 1
        self.main_db = main_db if os.path.exists(main_db) else None

    @staticmethod
    def _load_boards(path):
        out = []
        if not os.path.exists(path):
            return out
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for model, specs in con.execute(
                "select model, specs from hw_component where category='motherboards'"):
            s = json.loads(specs or "{}")
            summary = str(s.get("pcie_summary") or "")
            out.append({
                "model": model,
                "manufacturer": s.get("manufacturer") or "",
                "socket": s.get("socket") or "",
                "socket_key": snorm(s.get("socket")),
                "chipset": str(s.get("chipset") or ""),
                "form_factor": s.get("form_factor") or "",
                "ram_type": s.get("ram_type") or "",
                "ram_slots": to_int(s.get("ram_slots")),
                "ram_max_gb": to_int(s.get("ram_max_gb")),
                "m2": to_int(s.get("m2_count")),
                "m2_summary": str(s.get("m2_summary") or ""),
                "u2": to_int(s.get("u2_ports")),
                "sata": to_int(s.get("sata_6g_ports")) + to_int(s.get("sata_3g_ports")),
                "pcie_summary": summary,
                "x16_slots": count_slots(summary, 16),
                "pcie_lanes_gen5": bool(re.search(r"PCIe 5\.0", summary)),
                "pcie_flagged": bool(s.get("pcie_check")),
                "ecc": bool(s.get("ecc_support")),
                "raid": bool(s.get("raid_support")),
                "wifi": str(s.get("wifi") or ""),
                "ethernet": str(s.get("ethernet") or ""),
                "main_power": str(s.get("main_power") or ""),
                "cpu_power": str(s.get("cpu_power") or ""),
                "year": s.get("release_year"),
                "year_source": s.get("year_source") or "",
                "usb4": "USB4" in str(s.get("usb_headers") or "") + "USB4" in str(s.get("rear_io") or ""),
            })
        con.close()
        return out

    # ── Q1／Q2：CPU ↔ 主機板 ────────────────────────────────────────
    # 型號家族 ↔ 插槽的合理性：同一列的格子可能混進別的產品線
    # （Table of AMD processors 的家族列同時寫 AM4／FP6／SP3），
    # 直接取第一個像插槽的值會把「Threadripper PRO 5995WX」說成 SP3。
    PLAUSIBLE = [
        (re.compile(r"threadripper", re.I), re.compile(r"(str4|strx4|swrx8|str5|trx4|trx5)", re.I)),
        (re.compile(r"epyc", re.I), re.compile(r"(sp[3-8])", re.I)),
        (re.compile(r"ryzen", re.I), re.compile(r"(am[45]|fp[5-8]|fl1)", re.I)),
        (re.compile(r"(xeon|core|pentium|celeron|atom)", re.I), re.compile(r"(lga|socket\s?\d{3,4}|bga)", re.I)),
    ]

    @classmethod
    def plausible(cls, model, socket):
        """這個插槽對這個型號家族說得通嗎？找不到規則時回 True（不亂擋）。"""
        for fam, ok in cls.PLAUSIBLE:
            if fam.search(model or ""):
                return bool(ok.search(socket or ""))
        return True

    SOCKET_VALUE = re.compile(
        r"^\s*(?:(?:socket|lga|f|sp|tr)\s?)?"
        r"(lga\s?\d{3,4}(?:-\d)?(?:\s?\(v\d\))?|am\d\+?|fm\d\+?|s(?:wr)?x?\s?\d{1,4}"
        r"|sp\d|tr\d|kg\d{3}|bga\d{3,4}|f\s?\(1207\))\s*$", re.I)

    def cpu_socket(self, model):
        """從型號庫找出這顆 CPU 的插槽；找不到回傳 None（不猜）。

        順序很重要：先認「有標籤的欄位」（lib-cpu 每欄都有名字），
        再退回「整列裡找一個長得像插槽的值」——我們的 wiki 表格列沒有欄名，
        直接看欄名會錯配（例如把 L1 cache 當插槽）。
        """
        if not self.main_db:
            return None
        con = sqlite3.connect(f"file:{self.main_db}?mode=ro", uri=True)
        # 型號可能被寫成 "i7-12700K"、"Core i7 12700K"、"12700K"。
        # 型號庫的 norm 是「去品牌前綴」後的字串，而不同匯入管線去前綴的深度不一樣
        # （'12700k'／'i712700k'／'corei712700k' 都存在），所以：
        #   精確鍵 ＋ 「以數字字母尾端結尾」的 LIKE 都要試。
        key = re.sub(r"[^0-9a-z]+", "", str(model).lower())
        keys = {key, re.sub(r"^[a-z]+(?=\d)", "", key)}
        tails = set()
        m = re.search(r"(\d{3,5}[a-z]{1,4})$", key)
        if m:
            tails.add(m.group(1))
        conds = ["m.norm=?" for _ in keys] + ["m.norm LIKE ?" for _ in tails]
        params = sorted(keys) + ["%" + t for t in sorted(tails)]
        rows = con.execute(
            """SELECT m.headers, m.cells, m.kind, m.page, m.url FROM hw_model m
               WHERE m.kind IN ('lib-cpu','wiki-cpu')
                 AND (""" + " OR ".join(conds) + """)
               ORDER BY CASE WHEN m.kind='lib-cpu' THEN 0 ELSE 1 END
               LIMIT 20""", params).fetchall()
        con.close()
        for headers, cells, kind, page, url in rows:
            hh = json.loads(headers or "[]")
            cc = json.loads(cells or "[]")
            # 1) 有標籤、而且值本身確實像插槽
            if len(hh) == len(cc):
                for h, c in zip(hh, cc):
                    if ("socket" in str(h).lower() and self.SOCKET_VALUE.match(str(c))
                            and self.plausible(model, str(c))):
                        return str(c).strip(), page, url
            # 2) 沒標籤：整列找像插槽的值，先過合理性檢查
            for c in cc:
                if self.SOCKET_VALUE.match(str(c)) and self.plausible(model, str(c)):
                    return str(c).strip(), page, url
        # 3) 這一代的頁面自己說了（hw_socket_page），而且只說過一個插槽才採用
        sock = self._socket_by_page(rows, model)
        if sock:
            return sock[0], sock[1], sock[2]
        # 4) 桌機世代規則：只在上面都拿不到時用，而且回傳時標明是推定
        return self._socket_by_generation(model)

    # 行動版 SKU 的字尾：那些是 BGA 焊死在主機板上，「世代頁說什麼插槽」對它們無意義
    # （Zen_4 頁講的是桌機 AM5，拿去回答 7945HX 就是錯答案）。
    MOBILE_TAIL = re.compile(r"(?:hx3d|hx|hs|u|p|h)$", re.I)

    def _socket_by_page(self, rows, model):
        """世代頁層級的事實：Alder Lake→LGA 1700。只在「該頁只提過一個插槽」時採用。"""
        if not self.main_db or not rows:
            return None
        if self.MOBILE_TAIL.search(re.split(r"\s+", str(model or "").strip())[-1] or "") \
                or "ryzen ai" in (model or "").lower():
            return None
        # 頁名要正規化：lib 匯入的頁名是「lib:Alder Lake」（空白），wiki 抓的是「Alder_Lake」
        pages = {re.sub(r"^lib:", "", r[3] or "").replace(" ", "_")
                 for r in rows if r[3]}
        if not pages:
            return None
        try:
            con = sqlite3.connect(f"file:{self.main_db}?mode=ro", uri=True)
            for page in sorted(pages):
                r = con.execute("SELECT socket,sockets,segments FROM hw_socket_page WHERE page=?",
                                (page,)).fetchone()
                if not r:
                    continue
                sockets = json.loads(r[1] or "[]")
                seg = json.loads(r[2] or "{}")
                # 分段插槽：只在「世代頁」（不是 List of…）用，且要對得上家族產品線
                if seg and not page.lower().startswith(("list_of", "table_of")):
                    for key in self._segment_keys(model):
                        for label, sock in seg.items():
                            if key in label:
                                con.close()
                                return sock, page, ""
                if len(sockets) == 1:
                    con.close()
                    return sockets[0], page, ""
            con.close()
        except sqlite3.Error:
            return None
        return None

    @staticmethod
    def _segment_keys(model):
        """家族 → 該到資訊框哪個分段找插槽。"""
        m = (model or "").lower()
        if "threadripper pro" in m:
            return ("workstation", "high-end")
        if "threadripper" in m:
            return ("high-end", "workstation")
        if "epyc" in m or "opteron" in m:
            return ("server",)
        if "ryzen" in m or re.search(r"[3579]\d{3}(x3d|xx|hx)?", m):
            return ("mainstream desktop", "desktop")
        return ("desktop",)

    def boards_for_cpu(self, model, limit=8):
        sock = self.cpu_socket(model)
        if not sock:
            return {"found": False, "model": model, "reason":
                    "型號庫裡這顆 CPU 沒有 socket 欄位——不用同名插槽猜相容性"}
        value, page, url = sock
        key = snorm(value)
        boards = sorted(self.boards_by_socket.get(key, []),
                        key=lambda b: (b["year"] or 0), reverse=True)
        chipsets = sorted({b["chipset"] for b in boards if b["chipset"]})
        return {"found": True, "model": model, "socket": value, "socket_key": key,
                "from": page, "url": url, "board_count": len(boards),
                "chipsets": chipsets, "examples": [b["model"] for b in boards[:limit]],
                "note": self._socket_caveat(key)}

    # ── AMD 桌面型的世代 → 插槽（型號庫缺 socket 欄時的最後一條線）─────
    # 為什麼要這張表：List of AMD Ryzen processors 這種「一整頁跨多個插槽」的清單頁，
    # 表格裡的 Socket 欄是整組共用的 rowspan，少數列（例如 9800X3D）匯入後沒有值；
    # 而頁面級事實（hw_socket_page）同時出現 AM4／AM5／FP6／sTR5，不能拿來當答案。
    # 這裡只收「桌機、插槽唯一」的世代，行動版（H／HS／HX／U）一律不猜 —— 那些是 BGA，
    # 跟主機板相容性無關，猜錯只會製造假答案。
    GEN_SOCKET = [
        (re.compile(r"^ryzen [3579] (?:7[3-9]\d{2}|9[4-9]\d{2})(?:x3d|x|xt|f|g|ge|e)?$", re.I), "AM5"),
        (re.compile(r"^ryzen [3579] 8[3-7]\d{2}g(?:e)?$", re.I), "AM5"),
        (re.compile(r"^ryzen [3579] (?:5[5-9]\d{2}|5[12]\d{2}|4[67]\d{2})(?:x3d|x|xt|g|ge|)$", re.I), "AM4"),
        (re.compile(r"^ryzen [3579] 3[1-9]\d{2}(?:x|xt|g|ge)?$", re.I), "AM4"),
        (re.compile(r"^ryzen [3579] 2[1-7]\d{2}(?:x|g|ge)?$", re.I), "AM4"),
        (re.compile(r"^ryzen [3579] 1[4-8]\d{2}(?:x)?$", re.I), "AM4"),
        (re.compile(r"^threadripper (?:pro )?[79][0-9]{3}(?:wx|w|x)?$", re.I), "sTR5"),
        (re.compile(r"^threadripper (?:pro )?[35][0-9]{3}wx$", re.I), "sWRX8"),
        (re.compile(r"^threadripper (?:pro )?[35][0-9]{3}(?:x|xt)?$", re.I), "sTRX4"),
        (re.compile(r"^threadripper (?:pro )?[12][0-9]{3}(?:x|wx)?$", re.I), "TR4"),
    ]

    def _socket_by_generation(self, model):
        """型號 → 世代 → 插槽。回傳 (socket, '世代規則（推定）', '') 或 None。"""
        key = re.sub(r"\s+", " ", re.sub(r"^(?:amd|intel)\s+", "",
                                         str(model or "").strip(), flags=re.I)).strip()
        for pat, sock in self.GEN_SOCKET:
            if pat.match(key):
                return sock, "世代規則（推定）", ""
        return None

    @staticmethod
    def _socket_caveat(key):
        """同一插槽名跨多代／多電氣规格的已知陷阱：資料分不出來，就要講出來。"""
        notes = {
            "lga1151": "LGA1151 有兩代（6/7 代與 8/9/10 代 Coffee Lake），"
                       "名字相同但不相容；本資料的 socket 欄不區分，需以晶片組判斷。",
            "lga2011": "LGA2011／2011-3／2011-v4 互不相容（X79 vs X99）。",
            "lga1200": "LGA1200 同時支援 10 代與 11 代，但 11 代部分板要 BIOS。",
            "am4": "AM4 跨 Ryzen 1000–5000，但舊 300 系列需 BIOS 更新，"
                   "且部分 A320／X370 不在厂商支援清單。",
            "lga1700": "LGA1700 跨 12／13／14 代；600 系列晶片組上 13／14 代需 BIOS。",
        }
        return notes.get(key, "")

    def socket_lineage(self):
        """依板子數量排序的插槽清單（＝哪些世代在零售市場上真的存在）。"""
        return [{"socket": k, "boards": v} for k, v in self.sockets.most_common(20)]

    # ── Q5：工作站／HEDT 主機板的世代演進 ──────────────────────────
    # 一張「世代 → 插槽 → 配套晶片組」的固定序列（產業事實，不是推測），
    # 再從組件庫把每代的板數、年份、記憶體、PCIe 世代、擴充槽數**數出來**。
    WS_GENERATIONS = [
        ("LGA 1366", "X58", "Nehalem／Westmere Xeon"),
        ("LGA 2011", "X79", "Sandy／Ivy Bridge-E"),
        ("LGA 2011-3", "X99", "Haswell／Broadwell-E，DDR4 起點"),
        ("LGA 2066", "X299", "Skylake-X／Cascade Lake-X"),
        ("LGA 3647", "C621／C622／C627", "Skylake-SP／Cascade Lake-SP（雙路）"),
        ("LGA 4189", "C621A", "Ice Lake-SP／Cooper Lake，PCIe 4.0"),
        ("LGA 4677", "W790", "Sapphire Rapids（Xeon W-3400／2400），PCIe 5.0"),
        ("TR4", "X399", "Threadripper 1000／2000"),
        ("sTRX4", "TRX40", "Threadripper 3000，PCIe 4.0"),
        ("sWRX8", "WRX80", "Threadripper PRO 3000／5000WX，8 通道記憶體"),
        ("sTR5", "TRX50／WRX90", "Threadripper 7000（WRX90 = PRO 7000WX）"),
    ]

    def workstation_evolution(self):
        """每個工作站世代「市場上實際有幾張板、什麼規格」——數字全部來自組件庫。"""
        out = []
        for socket, chipsets, note in self.WS_GENERATIONS:
            key = snorm(socket)
            boards = [b for b in self.boards if (b["socket_key"] or "") == key]
            if not boards:
                out.append({"socket": socket, "boards": 0, "chipsets": [], "note": note})
                continue
            years = [b["year"] for b in boards if b["year"]]
            seen = Counter(b["ram_type"] for b in boards if b["ram_type"])
            out.append({
                "socket": socket, "boards": len(boards),
                "chipsets": chipsets,
                "chipsets_seen": sorted({b["chipset"] for b in boards if b["chipset"]}),
                "years": (min(years), max(years)) if years else None,
                "ram": seen.most_common(1)[0][0] if seen else "",
                "gen5": sum(1 for b in boards if b["pcie_lanes_gen5"]),
                "ecc": sum(1 for b in boards if b["ecc"]),
                "x16_max": max(b["x16_slots"] for b in boards),
                "m2_max": max(b["m2"] for b in boards),
                "u2": sum(1 for b in boards if b["u2"] >= 1),
                "note": note,
            })
        return out

    # ── Q3：超頻 ───────────────────────────────────────────────────
    def oc_boards(self, platform=None, limit=10):
        out = []
        for b in self.boards:
            ch = b["chipset"]
            intel = bool(INTEL_OC.search(ch)) and not b["platform_is_amd"]
            amd = bool(AMD_OC.search(ch)) and b["platform_is_amd"]
            if (intel or amd) and (not platform or platform in ch.lower()):
                out.append(b)
        return out

    # ── Q4：多 GPU／U.2／M.2／PCIe 同時滿足 ─────────────────────────
    def expansion(self, x16_min=2, m2_min=2, u2_min=1, gen5=False, ecc=False,
                  exclude_flagged=True, limit=12):
        rows = []
        for b in self.boards:
            if b["x16_slots"] < x16_min:
                continue
            if b["m2"] < m2_min:
                continue
            if b["u2"] < u2_min:
                continue
            if gen5 and not b["pcie_lanes_gen5"]:
                continue
            if ecc and not b["ecc"]:
                continue
            if exclude_flagged and b["pcie_flagged"]:
                continue          # 來源模板把 PCIe 世代預填 4.0 的那批，世代不可信
            rows.append(b)
        rows.sort(key=lambda b: (-b["x16_slots"], -b["m2"], -b["u2"]))
        funnel = {
            "x16>=2": sum(1 for b in self.boards if b["x16_slots"] >= 2),
            "m2>=2": sum(1 for b in self.boards if b["m2"] >= 2),
            "m2>=3": sum(1 for b in self.boards if b["m2"] >= 3),
            "u2>0": sum(1 for b in self.boards if b["u2"] > 0),
            "PCIe5.0 出洞": sum(1 for b in self.boards if b["pcie_lanes_gen5"]),
            "PCIe 世代被模板預填（不可信）": sum(1 for b in self.boards if b["pcie_flagged"]),
        }
        return rows, funnel, rows[:limit]

    def describe(self, b):
        return "；".join(x for x in (
            f"插槽 {b['socket']}" if b["socket"] else "",
            f"晶片組 {b['chipset']}" if b["chipset"] else "",
            f"板型 {b['form_factor']}" if b["form_factor"] else "",
            f"x16 插槽 {b['x16_slots']}" if b["x16_slots"] else "",
            f"M.2 {b['m2']}" if b["m2"] else "",
            f"U.2 {b['u2']}" if b["u2"] else "",
            f"SATA {b['sata']}" if b["sata"] else "",
            f"記憶體 {b['ram_type']}",
            f"網路 {b['ethernet']}" if b["ethernet"] else "",
        ) if x)

    def __len__(self):
        return len(self.boards)


# platform 判斷（Intel 板 vs AMD 板）：以插槽前綴為準，比晶片組字串可靠
def _tag_platforms(c):
    for b in c.boards:
        k = b["socket_key"]
        b["platform_is_amd"] = bool(re.match(r"(am\d|fm\d|s|tr|sp|kg|slotam)", k or "")) or \
                              bool(re.search(r"\b(amd|tr|x)\d{3}", b["chipset"], re.I))
    return c


def load(nc_db=NC_DB, main_db=MAIN_DB):
    return _tag_platforms(Components(nc_db, main_db))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = load()
    print(f"組件庫：{len(c)} 張主機板")
    print("插槽分佈（前 10）:", c.socket_lineage()[:10])
    for q in ("i5-12400F", "Ryzen 5 5600X", "Core i7-12700K", "Xeon W-2135"):
        r = c.boards_for_cpu(q)
        print("\n", q, "→", (f"{r['socket']}｜板子 {r['board_count']} 張｜晶片組 {r['chipsets'][:6]}"
                              f"｜{r['note']}" if r.get("found") else r["reason"]))
    rows, funnel, top = c.expansion()
    print("\n擴充漏斗:", funnel)
    print(f"同時滿足雙 x16＋M.2≥2＋U.2≥1（排除 PCIe 世代不可信）：{len(rows)} 張")
    for b in top[:5]:
        print("   ", b["model"], "→", c.describe(b))
