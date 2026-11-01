#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XinSpect-Ai 的檢索層：BM25 ＋ 中日韓二元切分 ＋ 跨語言術語橋接。

為什麼不用向量模型：
  這台機器只有 12 GB 的 TITAN Xp（Pascal，沒有 tensor core），跑 embedding 模型可以，
  但「檢索」在這個語料上的難點不是語意相近，而是**術語對得上**——`MSR 0x8B`、
  「封裝溫度」、「package temperature」指的是同一件事，而 bigram BM25 對這種
  逐字對得上的中文術語反而比稠密向量更穩。跨語言那一段用一張人工術語表補，
  補不到就退回「原文照搜」，不會假裝懂。

  純標準庫＋numpy，沒有訓練、沒有下載、沒有 GPU。索引可重建、可 diff。
"""

import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN = re.compile(r"[a-z0-9]+(?:[._\-][a-z0-9]+)*")


# ── 簡繁折疊 ────────────────────────────────────────────────────────────
# 文件同時有繁體與簡體版本，而使用者可能用另一邊的字問同一個概念
# （「命令行导出证据」對上的是繁體的「命令列匯出證據」）。不折疊的話，
# 簡體問題在繁體段落上的召回率會憑空掉一半——這不是語意問題，是字形問題。
#
# 做法：用 opencc 把「字元全集」逐一轉成簡體，建一張 char→char 的對照表，
# 之後 str.translate 就是純 C 速度的查表。沒有 opencc 就退回身分函式，
# 程式照跑，只是簡繁互查會差一點。
def _build_fold_table():
    try:
        from opencc import OpenCC
    except Exception:
        return None
    cc = OpenCC("t2s")
    chars = []
    seen = set()
    for cp in list(range(0x3400, 0x4DC0)) + list(range(0x4E00, 0xA000)) + \
              list(range(0xF900, 0xFB00)):
        ch = chr(cp)
        if ch in seen:
            continue
        seen.add(ch)
        chars.append(ch)
    # 分批轉：一次送四千字，opencc 的轉換器對超長字串沒有優勢
    mapping = {}
    for i in range(0, len(chars), 4000):
        batch = "".join(chars[i:i + 4000])
        conv = cc.convert(batch)
        if len(conv) != len(batch):
            # 長度不一致（罕用字被合併）就跳過這批，寧可漏幾百字也不要錯位
            continue
        for a, b in zip(batch, conv):
            if a != b:
                mapping[ord(a)] = b
    return mapping


_FOLD = _build_fold_table()


def fold(text):
    """把漢字折到簡體，讓簡繁兩種寫法落在同一個詞上。"""
    if not _FOLD:
        return text
    return text.translate(_FOLD)


# ── 兩岸用詞對照 ────────────────────────────────────────────────────────
# 字折疊解決「匯／汇」，解決不了「命令列／命令行」「記憶體／内存」——那是選詞不同，
# 不是字形不同。文件有簡體版，使用者用繁體問時，這張表讓兩邊落在同一個詞上。
# 只收這個語料裡真的會出現的技術用詞；收了就要雙向對稱，否則檢索會偏一邊。
REGIONAL = {
    "命令列": "命令行",
    "記憶體": "内存",
    "硬碟": "硬盘",
    "軟體": "软件",
    "網路": "网络",
    "預設": "默认",
    "資訊": "信息",
    "介面": "界面",
    "資料夾": "文件夹",
    "檔案": "文件",
    "連結": "链接",
    "程式": "程序",
    "執行": "执行",
    "支援": "支持",
    "最佳化": "优化",
    "偵測": "检测",
    "雜湊": "散列",
    "雜湊值": "哈希",
    "解析度": "分辨率",
    "頻寬": "带宽",
    "行動": "移动",
    "帳號": "账号",
    "連線": "连接",
    "匯出": "导出",
    "匯入": "导入",
    "存取": "访问",
    "伺服器": "服务器",
    "韌體": "固件",
    "驅動程式": "驱动程序",
    "快取": "缓存",
    "陣列": "数组",
    "函式": "函数",
    "物件": "对象",
    "欄位": "字段",
    "字元": "字符",
    "位元": "比特",
    "位元組": "字节",
    "擴充": "扩展",
    "選單": "菜单",
    "預覽": "预览",
    "簽章": "签名",
    "雜訊": "噪声",
    "封包": "数据包",
    "迴圈": "循环",
    "遞迴": "递归",
    "過濾": "过滤",
    "篩選": "筛选",
    "符合": "符合",
    "回報": "回报",
    "校驗": "校验",
    "唯讀": "只读",
    "寫入": "写入",
    "提權": "提权",
    "權限": "权限",
    "機密": "机密",
    "隱私": "隐私",
    "憑證": "凭证",
    "序號": "序号",
    "廠牌": "品牌",
    "規格": "规格",
    "腳位": "插槽",
    "散熱": "散热",
    "電壓": "电压",
    "功耗": "功耗",
    "時脈": "时钟",
    "封裝": "封装",
    "感測器": "传感器",
    "主機板": "主板",
    "晶片組": "芯片组",
    "顯示卡": "显卡",
    "固態硬碟": "固态硬盘",
    "機械硬碟": "机械硬盘",
    "記憶體插槽": "内存插槽",
    "作業系統": "操作系统",
    "應用程式": "应用程序",
    "說明文件": "说明文档",
    "版本沿革": "版本历史",
    "更新紀錄": "更新记录",
    "事實鍵": "事实键",
    "時間膠囊": "时间胶囊",
    "三源對帳": "三源对账",
    "資料主權": "数据主权",
    "可稽核": "可审计",
    "稽核": "审计",
    "授權": "授权",
    "貢獻": "贡献",
    "專案": "项目",
    "原始碼": "源代码",
    "開源": "开源",
    "建置": "构建",
    "編譯": "编译",
    "函式庫": "库",
    "相依": "依赖",
    "測試": "测试",
    "除錯": "调试",
    "錯誤": "错误",
    "例外": "异常",
    "發生": "发生",
    "處理": "处理",
    "處理器": "处理器",
    "執行緒": "线程",
    "核心數": "核心数",
    "頻率": "频率",
    "基礎": "基础",
    "效能": "性能",
    "延遲": "延迟",
    "吞吐": "吞吐",
    "穩定性": "稳定性",
    "量測": "测量",
    "讀值": "读数",
    "採樣": "采样",
    "離線": "离线",
    "線上": "在线",
    "雲端": "云端",
    "裝置": "设备",
    "週邊": "外设",
    "鍵盤": "键盘",
    "滑鼠": "鼠标",
    "螢幕": "屏幕",
    "電池": "电池",
    "電源": "电源",
    "風扇": "风扇",
    "匯流排": "总线",
    "匯流": "汇流",
    "變數": "变量",
    "旗標": "标志",
    "狀態": "状态",
    "布林": "布尔",
    "整數": "整数",
    "字串": "字符串",
    "清單": "列表",
    "目錄": "目录",
    "路徑": "路径",
    "雜湊表": "哈希表",
    "佇列": "队列",
    "堆疊": "栈",
    "中斷": "中断",
    "訊號": "信号",
    "註冊表": "注册表",
    "事件記錄": "事件日志",
    "藍屏": "蓝屏",
    "當機": "崩溃",
    "修復": "修复",
    "快照": "快照",
    "還原": "还原",
    "修補": "补丁",
    "漏洞": "漏洞",
    "攻擊": "攻击",
    "風險": "风险",
    "稽核記錄": "审计日志",
    "相容": "兼容",
    "建議": "建议",
    "需求": "需求",
    "安裝": "安装",
    "移除": "卸载",
    "便攜": "便携",
    "單一執行檔": "单文件",
    "免安裝": "免安装",
}

# 反向不必手寫：下面會把每一個鍵與值都收到同一個代表形，避免「換過去又換回來」的迴圈。
_CANON = {}
for _a, _b in REGIONAL.items():
    # 取字典序較短者為代表形只是為了穩定；重點是兩邊落在同一格
    _canonical = _b
    _CANON[_a] = _canonical
    _CANON.setdefault(_b, _canonical)

_CANON_KEYS = sorted(_CANON, key=len, reverse=True)


def regionalize(text):
    """把兩岸用詞換成同一個寫法（長詞優先，避免「硬碟」先被「硬」吃掉）。"""
    for a in _CANON_KEYS:
        if a in text:
            text = text.replace(a, _CANON[a])
    return text

# 功能詞不是證據：問「幾點」「天氣」時，真正決定相關性的是「幾點」「天氣」，
# 不是「現在」「如何」。把這些字拿掉，相關性判斷才不會被文法詞灌水。
STOPWORDS = {
    # 英文功能詞
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "how", "what", "when", "where", "why", "who", "which", "whom", "whose",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your", "our", "their", "its",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "about", "as", "and", "or",
    "but", "not", "no", "yes", "can", "could", "should", "would", "will", "shall", "may",
    "this", "that", "these", "those", "there", "here", "if", "then", "than", "so", "such",
    "tell", "please", "want", "need", "know", "give", "show", "explain", "help",
    # 中文疑問與功能詞
    "什麼", "什麼是", "怎麼", "怎樣", "如何", "為何", "為什麼", "哪裡", "哪個", "哪些",
    "請問", "告訴", "告訴我", "說明", "解釋", "可以", "能否", "是否", "有沒", "沒有",
    "現在", "今天", "明天", "昨天", "我的", "你的", "我們", "你們", "這個", "那個",
    "這些", "那些", "以及", "還有", "但是", "因為", "所以", "如果", "然後", "就是",
    "一下", "一樣", "一些", "一個", "一種", "這麼", "那麼", "多少", "幾個",
    "帮我", "幫我", "寫一", "一首", "一段", "推薦", "介紹",
}

# ── 跨語言術語橋接 ──────────────────────────────────────────────────────
# 左邊是英文（檢索時會被展開成右邊的中文寫法，反之亦然）。
# 只收「曦覽文件裡真的會出現」的術語；寧可少，不要亂翻——翻錯比不翻更糟。
GLOSSARY = {
    "cpuid": ["cpuid", "處理器識別"],
    "smbios": ["smbios", "系統管理基本輸入輸出"],
    "acpi": ["acpi", "進階組態與電源介面"],
    "msr": ["msr", "型號特定暫存器"],
    "nvme": ["nvme", "固態硬碟", "ssd"],
    "sata": ["sata", "硬碟"],
    "smart": ["smart", "健康", "自我監測"],
    "spd": ["spd", "記憶體模組", "記憶體"],
    "dimm": ["dimm", "記憶體模組", "記憶體插槽"],
    "pcie": ["pcie", "pci express", "匯流排"],
    "pci": ["pci", "匯流排"],
    "usb": ["usb", "通用序列匯流排"],
    "thunderbolt": ["thunderbolt", "雷電"],
    "tpm": ["tpm", "可信平台模組"],
    "secure boot": ["secure boot", "安全開機"],
    "uefi": ["uefi", "統一可延伸韌體介面"],
    "bios": ["bios", "韌體"],
    "firmware": ["firmware", "韌體"],
    "gpu": ["gpu", "顯示卡", "繪圖處理器"],
    "cpu": ["cpu", "處理器", "中央處理器"],
    "ram": ["ram", "記憶體"],
    "thermal": ["thermal", "溫度", "散熱"],
    "temperature": ["temperature", "溫度"],
    "power": ["power", "功耗", "電源"],
    "voltage": ["voltage", "電壓"],
    "frequency": ["frequency", "頻率", "時脈"],
    "clock": ["clock", "時脈", "頻率"],
    "benchmark": ["benchmark", "跑分", "效能測試"],
    "latency": ["latency", "延遲"],
    "bandwidth": ["bandwidth", "頻寬"],
    "throughput": ["throughput", "吞吐"],
    "snapshot": ["snapshot", "快照", "時間膠囊"],
    "time capsule": ["time capsule", "時間膠囊", "快照"],
    "fact key": ["fact key", "事實鍵"],
    "availability": ["availability", "可用性", "六態"],
    "provenance": ["provenance", "出處", "來源"],
    "evidence": ["evidence", "證據"],
    "reconciliation": ["reconciliation", "對帳", "三源對帳"],
    "privacy": ["privacy", "隱私"],
    "data sovereignty": ["data sovereignty", "資料主權"],
    "read-only": ["read-only", "唯讀"],
    "write": ["write", "寫入"],
    "driver": ["driver", "驅動程式", "驅動"],
    "kernel": ["kernel", "核心"],
    "event log": ["event log", "事件記錄"],
    "bsod": ["bsod", "藍屏", "當機"],
    "boot": ["boot", "開機"],
    "memory": ["memory", "記憶體"],
    "storage": ["storage", "儲存", "硬碟"],
    "disk": ["disk", "磁碟", "硬碟"],
    "battery": ["battery", "電池"],
    "audio": ["audio", "音訊"],
    "network": ["network", "網路"],
    "motherboard": ["motherboard", "主機板"],
    "chipset": ["chipset", "晶片組"],
    "license": ["license", "授權"],
    "release": ["release", "發佈", "版本"],
    "changelog": ["changelog", "版本沿革", "更新紀錄"],
    "install": ["install", "安裝"],
    "portable": ["portable", "便攜", "免安裝"],
    "elevation": ["elevation", "提權", "系統管理員"],
    "administrator": ["administrator", "系統管理員"],
    "sensor": ["sensor", "感測器"],
    "core": ["core", "核心"],
    "thread": ["thread", "執行緒"],
    "vendor": ["vendor", "廠商"],
    "serial number": ["serial number", "序號"],
    "hash": ["hash", "雜湊"],
    "export": ["export", "匯出"],
    "cli": ["cli", "命令列"],
    "rule": ["rule", "規則"],
    "diagnosis": ["diagnosis", "診斷"],
    "baseline": ["baseline", "基線"],
    "anomaly": ["anomaly", "異常"],
    "risk": ["risk", "風險"],
    "limitation": ["limitation", "限制"],
    "accuracy": ["accuracy", "準確"],
    "tolerance": ["tolerance", "誤差"],
}


def normalize(text):
    """統一的文本正規化：NFKC → 小寫 → 簡繁折疊 → 兩岸用詞統一。

    索引與查詢都必須走這一支，否則同一份文件在兩邊會被切成不同的詞。
    """
    return regionalize(fold(unicodedata.normalize("NFKC", text).lower()))


def tokenize(text):
    """英文詞（含點號鍵名）＋ CJK 二元切分。單一漢字另外保留一份，避免兩字詞被切碎。"""
    t = normalize(text)
    toks = []

    for m in LATIN.finditer(t):
        w = m.group(0)
        toks.append(w)
        # 鍵名前綴也有用：`ac.aml.boundary` → 也加 `ac.aml`
        if "." in w:
            parts = w.split(".")
            for i in range(1, len(parts)):
                toks.append(".".join(parts[:i]))
        # 極簡複數還原：temperatures → temperature
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            toks.append(w[:-1])

    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", t)
    for run in cjk_runs:
        if len(run) == 1:
            toks.append(run)
            continue
        for i in range(len(run) - 1):
            toks.append(run[i:i + 2])
        # 三字詞（「處理器」「記憶體」）在中文術語裡很常見
        if len(run) > 2:
            for i in range(len(run) - 2):
                toks.append(run[i:i + 3])
    return toks


def expand_query(query):
    """把查詢裡的術語展開成其他語言的對應寫法。回傳 (augmented_text, hits)。"""
    q = normalize(query)
    extra = []
    hits = []
    for en, alts in GLOSSARY.items():
        matched = False
        if en in q:
            matched = True
        if any(a in q for a in alts[1:] if a):
            matched = True
        if matched:
            hits.append(en)
            for a in alts:
                if a not in q:
                    extra.append(a)
    if extra:
        return q + "\n" + "\n".join(dict.fromkeys(extra)), hits
    return q, hits


def strip_stopwords(text):
    """拿掉功能詞，但**只按完整詞**拿。

    原本用 str.replace 逐個刪，結果單字母的英文功能詞把詞拆了：
    刪掉 "i" 之後 "install" 變成 " stall"，整題就再也命中不了。
    """
    text = normalize(text)
    out = []
    for m in re.finditer(r"[a-z0-9][a-z0-9._\-]*|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+|[^\s]", text):
        tok = m.group(0)
        if tok.isascii() and tok in STOPWORDS:
            out.append(" ")
        elif tok in STOPWORDS:
            out.append(" ")
        else:
            out.append(tok)
    text = " ".join(out)

    # 中文的複合功能詞（「什麼是」「告訴我」）沒有空白邊界，只能逐段刪；
    # 這裡先刪長詞，避免「什麼是」先被「什麼」吃掉之後留下一個「是」。
    for sw in sorted((s for s in STOPWORDS if not s.isascii()), key=len, reverse=True):
        if sw in text:
            text = text.replace(sw, " ")
    return text


class BM25Index:
    """倒排索引 ＋ BM25。整份索引存成 npz＋json，載入只要幾十毫秒。"""

    def __init__(self, k1=1.4, b=0.72):
        self.k1 = k1
        self.b = b
        self.doc_ids = []
        self.doc_lens = []
        self.postings = {}      # term -> list[(doc_idx, tf)]
        self.df = Counter()
        self.avgdl = 0.0
        self.docs = []          # 原始區塊，供回傳出處

    # ── 建索引 ──────────────────────────────────────────────────────
    def build(self, chunks):
        self.docs = chunks
        postings = defaultdict(list)
        lens = []
        for i, c in enumerate(chunks):
            toks = tokenize(c["text"] + "\n" + c["heading_path"])
            # 標題與鍵名前綴加權：同一個詞在標題出現算兩次
            toks += tokenize(c["heading_path"])
            tf = Counter(toks)
            lens.append(len(toks))
            for term, n in tf.items():
                postings[term].append((i, n))
                self.df[term] += 1
            self.doc_ids.append(c["id"])
        self.postings = dict(postings)
        self.doc_lens = np.asarray(lens, dtype=np.float32)
        self.avgdl = float(self.doc_lens.mean()) if len(lens) else 0.0
        return self

    # ── 查詢 ────────────────────────────────────────────────────────
    # ── 相關性閘門 ──────────────────────────────────────────────────
    def answerability(self, query, doc, min_run=2):
        """查詢的「內容字」有多少真的被這段文件**連續覆蓋**（不是零散命中）。

        為什麼要這一把而不是只看 BM25 分數、也不是看 n-gram 命中率：
        BM25 對「什麼是」「怎麼」這類高頻詞也給分，所以「什么量子力学」在曦覽文件上
        照樣能拿 22 分；而 n-gram 命中率會被切詞殘片灌水——「么是」「傳我」這種跨詞
        邊界的碎片 df 很低、idf 很高，一命中就讓分數看起來很相關，實際上一個概念都沒講到。

        所以這裡問的是更笨、更難作弊的問題：**把問題裡的內容字串，拿去文件中找最長的
        連續匹配，命中了幾成**。「量子力學」在文件裡找不到連續的「量子」二字，就一分都不給；
        「封裝溫度」找得到，就整段都算命中。
        """
        q = strip_stopwords(query)
        # 只留 CJK、英數與點號（鍵名），其餘視為分隔
        q = re.sub(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff.]+", " ", q)

        doc_text = normalize(doc["text"] + "\n" + doc["heading_path"])
        doc_tokens = set(tokenize(doc_text))

        total = 0.0
        hit = 0.0
        for seg in q.split():
            if not seg:
                continue
            if re.fullmatch(r"[0-9a-z.\-]+", seg):
                # 英數：整詞（或鍵名）命中才算；3 字元以下的不算證據（too、the 的殘留）
                if len(seg) < 3:
                    continue
                total += 2.0
                if seg in doc_tokens:
                    hit += 2.0
                continue

            # 中文段：逐字累積，用「最長連續匹配」蓋章
            i = 0
            n = len(seg)
            while i < n:
                if not CJK.match(seg[i]):
                    i += 1
                    continue
                best = 0
                for L in range(min(24, n - i), min_run - 1, -1):
                    if seg[i:i + L] in doc_text:
                        best = L
                        break
                if best:
                    total += best
                    hit += best
                    i += best
                else:
                    total += 1
                    i += 1
        if total <= 0:
            return 0.0
        return hit / total

    def coverage(self, query, doc):
        """查詢裡「有辨識度的詞」有多少真的出現在這段文件裡（idf 加權）。

        為什麼要有這個而不是只看 BM25 分數：BM25 對「什麼是」「怎麼」這類高頻詞也會給分，
        所以「什麼是量子力學」在曦覽文件上照樣能拿到 16 分——分數高低不是「有沒有講到」。
        這裡把 idf 太低的詞直接丟掉，只問「稀有詞命中幾成」，那才是「這段文件在講這件事」。
        """
        qtoks = set(tokenize(normalize(query))) - STOPWORDS
        doc_toks = set(tokenize(doc["text"] + "\n" + doc["heading_path"]))
        n_docs = max(len(self.docs), 1)
        total = 0.0
        hit = 0.0
        for t in qtoks:
            df = self.df.get(t, 0)
            # df=0 的詞一定不存在於語料，它只會是切詞切出來的殘片（「裝需求」「覽會不」）。
            # 把它算進分母，等於用一個不可能命中的詞把好問題判死。
            if df == 0:
                continue
            idf = math.log(1.0 + n_docs / (df + 1.0))
            if idf < 3.6:          # 太常見的詞：不算證據
                continue
            # 上限 6：單一罕見詞（df=1 的切詞殘片）不該一票決定整題的相關性
            idf = min(idf, 6.0)
            total += idf
            if t in doc_toks:
                hit += idf
        if total <= 0:
            return 0.0
        return hit / total

    def search(self, query, top_k=8, lang=None, expand=True, prefer_lang=None, prefer_bias=1.35):
        qtext, hits = expand_query(query) if expand else (normalize(query), [])
        q_tokens = Counter(tokenize(qtext))
        # 查詢詞權重：原查詢的詞比展開出來的詞重
        orig_tokens = set(tokenize(normalize(query)))
        n_docs = len(self.docs)
        if n_docs == 0:
            return []
        scores = np.zeros(n_docs, dtype=np.float32)
        for term, qtf in q_tokens.items():
            plist = self.postings.get(term)
            if not plist:
                continue
            df = self.df[term]
            # BM25 的 idf，用機率式變體避免負值
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            w = 1.0 if term in orig_tokens else 0.45
            norm = self.k1 * (1 - self.b + self.b * self.doc_lens / (self.avgdl or 1))
            idx = np.fromiter((p[0] for p in plist), dtype=np.int32, count=len(plist))
            tf = np.fromiter((p[1] for p in plist), dtype=np.float32, count=len(plist))
            contrib = idf * (tf * (self.k1 + 1)) / (tf + norm[idx])
            scores[idx] += w * qtf * contrib

        if lang:
            for i, d in enumerate(self.docs):
                if d["lang"] != lang:
                    scores[i] = 0.0
        elif prefer_lang:
            # 使用者用哪種語言問，就讓同語言的段落優先——但**不是**硬性過濾：
            # 中文文件對某些主題（例如英文白皮書才有的章節）可能沒有對應段，
            # 硬過濾會讓問題變成「查無資料」，而正確答案就在隔壁語言。
            for i, d in enumerate(self.docs):
                if d["lang"] != prefer_lang:
                    scores[i] *= 1.0 / prefer_bias

        k = min(top_k * 4, n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        results = []
        seen_text = set()
        for i in top:
            if scores[i] <= 0:
                break
            d = self.docs[int(i)]
            # 同文重複（重疊切塊）只留一次
            sig = d["text"][:120]
            if sig in seen_text:
                continue
            seen_text.add(sig)
            item = dict(d)
            item["score"] = float(scores[int(i)])
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    # ── 持久化 ──────────────────────────────────────────────────────
    def save(self, path_prefix):
        with open(path_prefix + ".docs.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        # postings 存成三段平行陣列，載入時用 np 快解
        terms = list(self.postings.keys())
        with open(path_prefix + ".postings.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for t in terms:
                plist = self.postings[t]
                f.write(json.dumps(
                    [t, [p[0] for p in plist], [p[1] for p in plist]],
                    ensure_ascii=False) + "\n")
        meta = {
            "k1": self.k1, "b": self.b,
            "avgdl": self.avgdl, "n_docs": len(self.docs),
            "doc_lens": self.doc_lens.tolist(),
        }
        with open(path_prefix + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    @classmethod
    def load(cls, path_prefix):
        self = cls()
        with open(path_prefix + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        self.k1 = meta["k1"]
        self.b = meta["b"]
        self.avgdl = meta["avgdl"]
        self.doc_lens = np.asarray(meta["doc_lens"], dtype=np.float32)
        with open(path_prefix + ".docs.jsonl", encoding="utf-8") as f:
            self.docs = [json.loads(line) for line in f if line.strip()]
        self.doc_ids = [d["id"] for d in self.docs]
        postings = {}
        df = Counter()
        with open(path_prefix + ".postings.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                term, idxs, tfs = json.loads(line)
                postings[term] = list(zip(idxs, tfs))
                df[term] = len(idxs)
        self.postings = postings
        self.df = df
        return self


def build_from_jsonl(jsonl_paths):
    """可傳單一路徑或路徑清單。曦覽文件與外部語料分成兩個檔，索引合併：
    檢索時兩邊一起比，才不會出現「問 PCIe 只找到曦覽文件、找不到 PCIe 規格」。
    """
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]
    chunks = []
    for p in jsonl_paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    return BM25Index().build(chunks)


def main():
    import argparse
    # 這台機器的主控台是 cp950：不先改 stdout，印出「›」就會炸掉整支程式。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--corpus", default=os.path.join(here, "..", "corpus", "corpus.jsonl"))
    ap.add_argument("--extra-corpus", nargs="*", default=None,
                    help="額外的語料檔（例如 corpus/corpus-external.jsonl）")
    ap.add_argument("--out", default=os.path.join(here, "index", "xinspect"))
    ap.add_argument("--query", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if args.query:
        idx = BM25Index.load(args.out)
        for r in idx.search(args.query, top_k=args.top):
            print(f'[{r["score"]:.2f}] {r["source"]}:{r["line_start"]} · {r["heading_path"][:60]}')
            print("   " + r["text"][:180].replace("\n", " "))
        return 0

    idx = build_from_jsonl([args.corpus] + (args.extra_corpus or []))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    idx.save(args.out)
    print(f"indexed {len(idx.docs)} chunks, vocab {len(idx.postings)}, avgdl {idx.avgdl:.1f}")
    print(f"saved -> {args.out}.{{docs.jsonl,postings.jsonl,meta.json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
