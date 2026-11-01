#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""XinSpect-Ai 推論服務：檢索 → 組提示 → 生成 → 附出處，OpenAI 相容介面。

設計上的三條硬規則，全部來自曦覽文件自己的立場：

1. **沒有出處就不回答。** 檢索不到足夠段落時，回「文件裡沒有寫到這件事」並說明查了哪裡，
   而不是用模型的常識補一個聽起來合理的答案。曦覽的整份文件就在講這件事。
2. **不冒充量測。** 任何「你這台電腦的溫度是…」都不可能出自這個網站——網站不掃描機器。
   系統提示與後處理都會擋掉這類回覆。
3. **三語同權。** 使用者用什麼語言問就用什麼語言答，且引用的段落可以是另一種語言的
   （文件本身是三語對譯，引用原文比翻譯更誠實）。

上游（實際產字的模型）由 --upstream 決定：
  ollama       → http://127.0.0.1:11434/v1      （本機）
  workers-ai   → 既有的 Cloudflare Worker        （官網現役路線）
  openai       → 任何 OpenAI 相容端點            （自填金鑰）
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rag"))

from bm25 import BM25Index  # noqa: E402
from entities import EntityStore  # noqa: E402

# 兩個索引、兩套門檻。曦覽自有文件與外部硬體語料的「該拒答」界線不同——
# 前者問題範圍窄（產品本身），後者是通用硬體知識，離題問題更容易沾到一兩個詞。
# 數字來自 eval/eval_split.py 的掃描（product 0.55 / external 0.6+bm25 22）。
DEFAULT_INDEX = os.path.join(HERE, "..", "rag", "index", "product")

LANGS = {"zh-Hant": "繁體中文", "zh-Hans": "简体中文", "en": "English"}

try:  # opencc 只用來判斷簡繁；沒裝就退回字集比對
    from opencc import OpenCC

    _t2s = OpenCC("t2s")
    _s2t = OpenCC("s2t")
except Exception:  # pragma: no cover
    _t2s = _s2t = None


def detect_lang(text):
    """粗分三語。判不出來（例如只有型號）就當繁體——官網的主語言。

    簡繁用 opencc 雙向轉換比對：轉完字串變了就是簡體。這比列一張「常用字差異表」可靠得多，
    那張表永遠列不完（「现」「点」「机」都在表外，就會把簡體判成繁體）。
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        if _t2s is not None:
            to_simplified = _t2s.convert(text)
            to_traditional = _s2t.convert(text)
            # 方向判定：繁體→簡體的轉換若能改動這段字，代表原文是繁體；反之亦然。
            # （先前把這個判斷寫反了，「現在幾點」會被判成簡體，介面語言就跟著錯。）
            changed_by_t2s = to_simplified != text
            changed_by_s2t = to_traditional != text
            if changed_by_t2s and not changed_by_s2t:
                return "zh-Hant"
            if changed_by_s2t and not changed_by_t2s:
                return "zh-Hans"
            # 兩邊都改不動（沒有簡繁差異字）或都改得動（混用）：比改動字數
            s_diff = sum(1 for a, b in zip(text, to_simplified) if a != b)
            t_diff = sum(1 for a, b in zip(text, to_traditional) if a != b)
            if s_diff == 0 and t_diff == 0:
                return "zh-Hant"
            return "zh-Hant" if s_diff > t_diff else "zh-Hans"
        simplified = set("这说时过还进个为发后里电机软网络术语关键线数据维优产业务级辆现点电脑检测")
        traditional = set("這說時過還進個為發後裡電機軟網絡術語關鍵線數據維優產業務級輛現點電腦檢測")
        s = sum(1 for ch in text if ch in simplified)
        t = sum(1 for ch in text if ch in traditional)
        return "zh-Hans" if s > t else "zh-Hant"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "zh-Hant"


# ── 提示詞 ──────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "zh-Hant": """你是「XinSpect-Ai」，曦覽（XinSpect）官方網站的說明助手。曦覽是一款在 Windows 上執行的原生硬體檢視、監控與安全稽核工具。

【你能做什麼】只根據下面提供的〔資料片段〕回答兩類問題：
  · 曦覽本身：它是什麼、讀哪些資料、每一把事實鍵從哪條路徑讀來、憑什麼算數、有哪些限制、版本沿革、隱私邊界、安裝與需求。
  · 電腦硬體與系統的一般知識（外部資料才有的部分）：PCI／USB／PCIe／記憶體／儲存／匯流排／處理器架構等。

【三條不能破的規則】
1. 每一個實質主張後面都要標出處。**照抄片段標頭裡方括號中的那一段**（例如 `README.md:10508`
   或 `Wikipedia/PCI Express 第 2 段`），不要自己編行號、也不要改寫成別的格式。
   片段裡沒有的內容不要補；若片段不足以回答，直接說「文件裡沒有寫到這件事」。
2. 你不能讀使用者的電腦。任何「你這台機器現在…」的問題，回答網站不掃描任何電腦，
   並指出曦覽桌面版能讀到哪些對應的事實鍵。不要把假設當成讀值，也不要引用文件裡的範例數字當成他的實測。
3. 曦覽的立場是「查不到不等於沒有」：讀不到、不支援、無權限是三種不同的狀態，不要混成一句「沒有」。

【資料可信度要分清】曦覽官方文件與 Wikipedia 是兩種材料。引用外部資料時要說明它來自外部，
不要說成曦覽官方說法；CC-BY-SA 的內容要標出來源頁。

【語氣】繁體中文，冷靜、具體、不用行銷語。寧可說「這份文件沒寫」，也不要給一個聽起來合理的猜測。回答長度以講清楚為準，通常 3～6 段。""",

    "zh-Hans": """你是「XinSpect-Ai」，曦览（XinSpect）官方网站的说明助手。曦览是一款在 Windows 上运行的原生硬件检视、监控与安全稽核工具。

【你能做什么】只根据下面提供的〔资料片段〕回答两类问题：
  · 曦览本身：它是什么、读哪些数据、每一把事实键从哪条路径读来、凭什么算数、有哪些限制、版本沿革、隐私边界、安装与需求。
  · 电脑硬件与系统的一般知识（外部资料才有的部分）：PCI／USB／PCIe／内存／存储／总线／处理器架构等。

【三条不能破的规则】
1. 每一个实质主张后面都要标出处。**照抄片段标题里方括号中的那一段**（例如 `README.md:10508`
   或 `Wikipedia/PCI Express 第 2 段`），不要自己编行号、也不要改写成别的格式。
   片段里没有的内容不要补；若片段不足以回答，直接说「文档里没有写到这件事」。
2. 你不能读用户的电脑。任何「你这台机器现在…」的问题，回答网站不扫描任何电脑，
   并指出曦览桌面版能读到哪些对应的事实键。不要把假设当成读值，也不要引用文档里的示例数字当成他的实测。
3. 曦览的立场是「查不到不等于没有」：读不到、不支持、无权限是三种不同的状态，不要混成一句「没有」。

【资料可信度要分清】曦览官方文档与 Wikipedia 是两种材料。引用外部资料时要说明它来自外部，
不要说成曦览官方说法；CC-BY-SA 的内容要标出来源页。

【语气】简体中文，冷静、具体、不用营销语。宁可说「这份文档没写」，也不要给一个听起来合理的猜测。回答长度以讲清楚为准，通常 3～6 段。""",

    "en": """You are "XinSpect-Ai", the assistant on the official XinSpect website. XinSpect is a native Windows hardware inspection, monitoring and security-audit tool.

[What you do] Answer from the passages below only, on two kinds of questions:
  - XinSpect itself: what it is, which data it reads, where each fact key comes from, why a value counts, its limits, release history, privacy boundaries, installation.
  - General computer hardware and systems knowledge (the external material): PCI/USB/PCIe, memory, storage, buses, microarchitecture.

[Three rules you may not break]
1. Cite after every substantive claim. **Copy the locator exactly as it appears in square brackets in the
   passage header** (e.g. `README.md:10508` or `Wikipedia/PCI Express paragraph 2`). Never invent line numbers
   and never reformat it. Do not add anything the passages do not contain; if they are insufficient, say plainly
   that the documentation does not cover it.
2. You cannot read the user's computer. For any "what is my machine doing right now" question, say the website
   never scans a computer, then point to the fact keys the desktop app would read. Never present an example
   number from the documentation as the user's measurement.
3. XinSpect's position is "not found is not the same as absent": unreadable, unsupported and unauthorised are
   three different states. Do not collapse them into "no".

[Separate the grades of evidence] XinSpect's own documentation and Wikipedia are different material. When you
use external material, say so and do not present it as an official XinSpect statement; attribute CC-BY-SA content.

[Tone] Calm, concrete, no marketing. Prefer "the documentation does not say" over a plausible guess. Three to six paragraphs unless the question needs more.""",
}


def locator_of(h):
    """這一塊的出處要怎麼寫給模型看。

    曦覽文件有真正的行號（`README.md:10508`），維基段落沒有——硬寫 `:0` 或
    `:line` 會讓模型照抄出一個假行號（實測就出現過「【wiki.pcie:2】」這種引用）。
    所以外部資料改用「頁名 + 第幾段／第幾張表」，那是讀者真的找得到的定位方式。
    """
    if h.get("origin") == "external" or h.get("_corpus") == "external":
        title = h.get("doc_title") or h.get("source", "")
        # doc_title 對規格表已經是「PCI Express 規格表」；這裡再補一次就會變成
        # 「規格表 規格表 8」——先把那個後綴拆掉再組。
        title = re.sub(r"\s*(規格表|规格表)$", "", title).strip()
        if h.get("table_index"):
            return f"Wikipedia/{title} 規格表 {h['table_index']}"
        if h.get("paragraph"):
            return f"Wikipedia/{title} 第 {h['paragraph']} 段"
        return f"Wikipedia/{title}"
    if h.get("line_start"):
        return f'{h["source"]}:{h["line_start"]}'
    return h.get("source", "")


def build_prompt(question, hits, lang, ui_lang=None):
    parts = []
    has_external = False
    for i, h in enumerate(hits, start=1):
        loc = locator_of(h)
        lic = h.get("license") or ""
        if h.get("origin") == "external" or h.get("_corpus") == "external":
            has_external = True
            tag = f"〔{lic}〕" if lic else ""
            parts.append(
                f'〔片段 {i}〕〔{loc}〕{tag}〔{h["heading_path"]}〕\n{h["text"]}')
        else:
            parts.append(
                f'〔片段 {i}〕〔{loc}〕〔{h["heading_path"]}〕\n{h["text"]}')
    context = "\n\n".join(parts)
    sys_prompt = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["zh-Hant"])

    if has_external:
        # 外部語料（Wikipedia 等）與曦覽文件的可信度等級不同，必須在提示裡講明白：
        # 曦覽文件是產品的一手說明，Wikipedia 是二手整理。回答時要說這一段是從哪來的。
        sys_prompt += {
            "zh-Hant": "\n\n【外部資料】這次的片段有一部分來自維基百科等外部來源（片段標題旁有授權標記）。"
                       "引用時要說明它來自外部資料，不要說成曦覽官方文件；CC-BY-SA 的內容要標出來源。",
            "zh-Hans": "\n\n【外部数据】这次的片段有一部分来自维基百科等外部来源（片段标题旁有授权标记）。"
                       "引用时要说明它来自外部数据，不要说成曦览官方文档；CC-BY-SA 的内容要标出来源。",
            "en": "\n\n[External material] Some passages come from Wikipedia or other external sources "
                  "(their licence is marked next to the heading). Say plainly that those come from "
                  "external material rather than the official XinSpect documentation, and attribute "
                  "CC-BY-SA content.",
        }[lang]

    if ui_lang and ui_lang != lang:
        sys_prompt += f"\n\n用{LANGS.get(lang, lang)}回答，但站方介面語言是{LANGS.get(ui_lang, ui_lang)}。"
    user = f"〔資料片段〕\n{context}\n\n〔問題〕\n{question}"
    return sys_prompt, user


# ── 後處理：把不該出現的東西擋掉 ────────────────────────────────────────
FAKE_READING = re.compile(
    r"(你(的)?(這台|目前|現在).{0,12}(溫度|電壓|頻率|功耗|風扇|使用率)\s*(是|為|約)\s*[0-9])"
    r"|(your (cpu|gpu|machine|pc).{0,20}(is|at) (currently )?[0-9]+\s*(°|c\b|mhz|ghz|w\b|%))",
    re.I)


# 「你有沒有掃我的電腦」類問題：使用者問的是「我這台機器現在怎樣」。
# 這種問題在文件裡一定有大量相關段落（所以通過得了相關性門檻），
# 但答案不可能是讀值——網站不掃描任何電腦。這類問題走專用回覆，不走一般問答。
MACHINE_SUBJECT = (
    # 简繁都要寫：只列繁體的話，「我这台电脑的内存有多少」整條規則不會響。
    r"我(這台|这台|的|家)(電腦|电脑|機器|机器|机台|主機|主机|筆電|笔电|桌機|桌机)"
    r"|(這|这|那)(台|部)(電腦|电脑|機器|机器|机台|主機|主机|筆電|笔电|桌機|桌机)"
    r"|我的\s?(cpu|gpu|顯示卡|显卡|显卡|記憶體|内存|硬碟|硬盘|固態|固态|主機板|主板|電源|电源|風扇|风扇|螢幕|屏幕|處理器|处理器)")

MACHINE_ASK = (
    # 「我的電腦配置是什麼」要攔得住。原本這裡只列「幾度／多少／多快」這種量度詞，
    # 結果配置與型號類的問法直接漏過去，模型撿到一段講「我的電腦」頁面的文件就回答，
    # 讀起來像它真的看過那台機器——這比答錯更糟。
    r"幾度|几度|多少|多快|幾瓦|几瓦|多大|多熱|多热|夠不夠|够不够|正常嗎|正常吗|健康嗎|健康吗"
    r"|有沒有|有没有|是多少|是什麼|是什么|是哪|哪一張|哪一张|哪一顆|哪一颗|哪款"
    r"|什麼型號|什么型号|什麼配置|什么配置|規格|规格|配置|型號|型号|跑幾分|跑几分|快不快|如何|怎樣|怎样|怎么样"
    r"|轉速|转速|\brpm\b")

MACHINE_SPECIFIC = re.compile(
    r"%s[^。？?\n]{0,20}(?:%s)" % (MACHINE_SUBJECT, MACHINE_ASK)
    + r"|my (pc|computer|machine|laptop|rig)[^.\n]{0,40}"
      r"(how (hot|cold|fast|many|much)|temperature|is it|doing|specs?|configuration)"
      r"|what(?:'s| is) my (pc|computer|machine|gpu|cpu|ram)"
      r"|how (hot|fast|much power) is my"
      r"|(what|which).{0,25}(gpu|cpu|ram|memory|storage).{0,15}do i have"
      r"|\bwhat are my (specs|specifications)\b",
    re.I)

# 相關性門檻：0.45 是在 eval/questions.json（30 題域內、18 題域外）上掃出來的
# 平衡點——域內召回 0.87、域外拒答 0.83、precision 0.90。調高會多拒掉真問題，
# 調低會開始回答「現在幾點」這種問題。改這個數字要重跑 eval/eval_retrieval.py。
MIN_ANSWERABILITY = 0.45


def looks_machine_specific(question):
    if not MACHINE_SPECIFIC.search(question):
        return False
    # 「曦覽在我這台要什麼配置」問的是系統需求，是文件有的內容；別把它當成讀值問題擋掉。
    if re.search(r"曦覽|曦览|XinSpect", question, re.I) and not re.search(
            r"幾度|溫度|温度|多熱|健康|正常嗎|跑幾分", question):
        return False
    return True


def postprocess(text, lang):
    """把「假裝讀到你的機器」的句子換成事實。只動這一種句形，其餘原文不動。"""
    if not FAKE_READING.search(text):
        return text
    note = {
        "zh-Hant": "\n\n（更正：本網站無法讀取你的電腦，上面那組數字不可能是你的實測值。"
                   "桌面版才會讀取即時讀值。）",
        "zh-Hans": "\n\n（更正：本网站无法读取你的电脑，上面那组数字不可能是你的实测值。"
                   "桌面版才会读取即时读值。）",
        "en": "\n\n(Correction: this website cannot read your computer; those numbers cannot be "
              "your measurements. Only the desktop app reads live values.)",
    }[lang]
    return text + note


# 答案裡自己寫出來的定位字串（檔案:行號）。外部語料的「Wikipedia/… 第 N 段」不在這裡處理。
RE_ANSWER_LOC = re.compile(r"([\w./\-]+\.(?:md|cs|json|txt|csproj|ps1|yml))[:：]L?(\d+)")


def repair_locators(text, cites):
    """把答案裡「檔案對得上、行號差幾行」的定位，改成真的檢索到的那一行。

    為什麼要修：小模型（1.7B）會把 939 寫成 941。差兩行看起來無害，但讀者點進去
    看到的是別的東西——出處的價值就在「點得到、對得上」，差一行就沒了。
    只在同一個檔案有被檢索到的時候修；檔案根本沒被檢索到的原樣留著，
    讓 eval/run_e2e.py 的「出處對得上」把它抓出來。
    """
    by_file = {}
    for c in cites or []:
        m = RE_ANSWER_LOC.search(c.get("locator") or "")
        if m:
            by_file.setdefault(m.group(1), set()).add(m.group(2))
    if not by_file:
        return text

    def fix(m):
        name, line = m.group(1), m.group(2)
        if name not in by_file or line in by_file[name]:
            return m.group(0)
        best = min(by_file[name], key=lambda x: abs(int(x) - int(line)))
        return f"{name}:{best}"

    return RE_ANSWER_LOC.sub(fix, text)


# ── 上游：Ollama / Workers AI / 任何 OpenAI 相容端點 ──────────────────────
class Upstream:
    def __init__(self, kind, base, model, key=None, timeout=180):
        self.kind = kind
        self.base = base.rstrip("/")
        self.model = model
        self.key = key
        self.timeout = timeout

    def _post(self, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base + "/chat/completions", data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        # Cloudflare 對 Python-urllib 的預設 UA 直接回 1010（browser integrity check），
        # 跟請求內容無關。帶一個正常 UA 就過得去，不是繞過什麼驗證。
        req.add_header("User-Agent", "XinSpect-AI/0.1 (+https://github.com/Xinglanclever/XinSpect)")
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def chat(self, messages, temperature=0.2, max_tokens=1600, stream=False):
        """回傳 (status, text_or_iterator)。錯誤一律轉成 (status, 錯誤字串)。"""
        try:
            resp = self._post({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            })
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            return e.code, detail
        except Exception as e:
            return 502, f"{type(e).__name__}: {e}"

        if not stream:
            raw = resp.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw)
                return 200, data["choices"][0]["message"]["content"]
            except Exception:
                return 502, raw[:400]

        return 200, self._iter_stream(resp)

    def chat_stream(self, messages, temperature=0.2, max_tokens=1600):
        """真正的逐字串流：邊收邊吐 (status, generator)。

        為什麼不把整段算完再切塊送去：那只是「假串流」，使用者要等完整段生成才看到字。
        上游若是 Ollama 或任何 OpenAI 相容端點都支援 SSE，直接轉出去最省事。
        """
        try:
            resp = self._post({
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            })
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            return e.code, detail
        except Exception as e:
            return 502, f"{type(e).__name__}: {e}"
        return 200, self._iter_stream(resp)

    @staticmethod
    def _iter_stream(resp):
        """把上游的 SSE 攤成純文字片段。

        要處理兩種上游：
          · 真的會串流的（Ollama 本機、任何正規 OpenAI 相容端點）——逐塊吐；
          · **不回串流的**（曦覽那支 Cloudflare Worker 就是這種：它在 README 裡明寫
            「一律不串流」，即使客端要求 stream=true 也回一整包 JSON）。
        第二種若不處理，這裡會一行 `data:` 都讀不到，前端就只會看到空白回答——
        實測就是這樣炸的。所以先看 content-type，非 SSE 就整包解析、一次吐出。
        """
        ctype = ""
        try:
            ctype = (resp.headers.get("Content-Type") or "").lower()
        except Exception:
            pass

        if "text/event-stream" not in ctype:
            raw = resp.read().decode("utf-8", "replace")
            piece = ""
            try:
                obj = json.loads(raw)
                piece = (obj.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            except Exception:
                # 不是 OpenAI 形狀（例如 Workers AI 原生格式）就整段當文字送出，
                # 至少使用者看得到內容，而不是一片空白。
                piece = raw.strip()
            if piece:
                yield piece
            return

        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                obj = json.loads(payload)
                delta = obj["choices"][0].get("delta", {})
                piece = delta.get("content") or ""
            except Exception:
                continue
            if piece:
                yield piece


# ── HTTP 服務 ───────────────────────────────────────────────────────────
class App:
    def __init__(self, product_idx, upstream, top_k=6, min_answerability=MIN_ANSWERABILITY,
                 external_idx=None, entities=None,
                 ext_min_answerability=0.6, ext_min_bm25=22.0):
        self.product = product_idx
        self.external = external_idx
        self.entities = entities or EntityStore()
        self.upstream = upstream
        self.top_k = top_k
        self.min_answerability = min_answerability
        self.ext_min_answerability = ext_min_answerability
        self.ext_min_bm25 = ext_min_bm25

    @property
    def index(self):
        """既有呼叫端（含 /health）用它拿筆數；回自有文件的索引。"""
        return self.product

    def prepare(self, question, lang=None, ui_lang=None, top_k=None):
        """把「要不要問模型」決定完，回傳兩種結果之一：

          {"early": {...}}          —— 查表命中／機器專屬問題／查無資料，直接回，不必叫模型
          {"prompt": (sys,user), "cites": [...]}  —— 需要生成，連同出處一起帶出去

        抽成這一支是為了串流：串流與非串流的差別只在「怎麼把模型的話送出去」，
        「該不該問、問什麼、引用哪些」完全一樣，不該寫兩份（寫兩份就會漂移）。
        """
        lang = lang or detect_lang(question)

        if looks_machine_specific(question):
            return {"early": self._machine_specific(question, lang)}

        # ── 第一層：硬體 ID 精確查找 ──────────────────────────────────
        # 「10de:1b80 是什麼」不該問語言模型，該查表。查到了就直接用查到的
        # 那一句話回答——比讓模型複述更準，也省一次推論（這台機器的推論很慢）。
        resolved = self.entities.resolve(question) if self.entities.available else []
        if resolved:
            return {"early": self._entity_answer(question, resolved, lang)}

        # ── 第二層：兩個語料庫各檢索一次，用各自的門檻 ──────────────────
        pool = []
        prod_hits = self.product.search(question, top_k=top_k or self.top_k, prefer_lang=lang)
        for h in prod_hits:
            if self.product.answerability(question, h) >= self.min_answerability:
                h = dict(h)
                h["_corpus"] = "product"
                pool.append(h)

        if self.external is not None:
            ext_hits = self.external.search(question, top_k=top_k or self.top_k)
            ext_strong = [dict(h) for h in ext_hits
                          if self.external.answerability(question, h) >= self.ext_min_answerability
                          and h["score"] >= self.ext_min_bm25]
            for h in ext_strong:
                h["_corpus"] = "external"
            # 外部語料只在自有文件答不了時才補位：
            # 問「曦覽怎麼讀溫度」時，Wikipedia 的「溫度」條目不該排在曦覽文件前面。
            if not pool:
                pool = ext_strong

        if not pool:
            msg = {
                "zh-Hant": "曦覽的文件裡沒有寫到這個問題，硬體 ID 對照表裡也沒有查到對應的項目。"
                           "我查了百科、技術白皮書、手冊與版本沿革，沒有對得上的段落——"
                           "所以我不會用猜的補一個答案。"
                           "你可以換個問法，或直接問曦覽讀哪些資料、有哪些事實鍵。",
                "zh-Hans": "曦览的文档里没有写到这个问题，硬件 ID 对照表里也没有查到对应的项目。"
                           "我查了百科、技术白皮书、手册与版本沿革，没有对得上的段落——"
                           "所以我不会用猜的补一个答案。"
                           "你可以换个问法，或直接问曦览读哪些数据、有哪些事实键。",
                "en": "Neither the XinSpect documentation nor the hardware ID tables cover this. "
                      "I searched the encyclopedia, the whitepaper, the manual, the changelog and "
                      "the PCI/USB/OUI tables and found no matching entry, so I will not invent an "
                      "answer. Try rephrasing, or ask which data XinSpect reads.",
            }[lang]
            return {"early": {"answer": msg, "citations": [], "lang": lang, "grounded": False,
                              "searched": [h["heading_path"] for h in prod_hits[:3]]}}

        pool.sort(key=lambda h: h["score"], reverse=True)
        strong = pool[: top_k or self.top_k]

        cites = []
        seen = set()
        per_source = {}
        for h in strong:
            # 同一頁的第 1、2、3 段是三個不同的片段，但對讀者來說是「同一個出處」。
            # 前端把三個同樣的維基連結排成一列沒有意義：按來源去重，且每個來源最多兩條
            # （一段散文 + 一張表是有資訊量的組合，三段散文不是）。
            key = h["source"]
            if per_source.get(key, 0) >= 2:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            sig = (key, h.get("paragraph"), h.get("table_index"))
            if sig in seen:
                continue
            seen.add(sig)
            cites.append({
                "source": h["source"],
                "line_start": h["line_start"],
                "line_end": h["line_end"],
                "heading_path": h["heading_path"],
                "locator": locator_of(h),
                "url": h["url"],
                "score": round(h["score"], 2),
                "license": h.get("license", ""),
                "corpus": h.get("_corpus", "product"),
            })

        sys_prompt, user = build_prompt(question, strong, lang, ui_lang)
        return {"prompt": (sys_prompt, user), "cites": cites, "lang": lang}

    def answer(self, question, lang=None, ui_lang=None, top_k=None):
        plan = self.prepare(question, lang, ui_lang, top_k)
        if "early" in plan:
            return plan["early"]

        sys_prompt, user = plan["prompt"]
        lang = plan["lang"]
        status, text = self.upstream.chat(
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user}])
        if status != 200:
            return {"error": f"上游回應 {status}：{text}", "lang": lang}

        text = repair_locators(postprocess(text, lang), plan["cites"])
        return {"answer": text, "citations": plan["cites"], "lang": lang, "grounded": True}

    def _entity_answer(self, question, resolved, lang):
        """ID 查到了就照查到的寫，不交給模型改寫。

        為什麼不讓模型複述：`10de:1b80 = NVIDIA GP104 [GeForce GTX 1080]` 是一個
        確定事實，模型複述時有機會把 GTX 1080 寫成 1080 Ti——這種錯誤版本號最難被發現。
        """
        lines = [r["text"] for r in resolved]
        if lang == "en":
            head = "This is a table lookup, not a guess:"
        elif lang == "zh-Hans":
            head = "这是查表结果，不是推测："
        else:
            head = "這是查表結果，不是推測："

        # 出處只寫在 citations 裡，讓前端統一畫成標籤。
        # 內文再列一次「出處」會讓同一條資訊出現兩遍。
        body = head + chr(10).join([""] + ["- " + l for l in lines])

        return {
            "answer": body,
            "citations": [{
                "source": r["kind"], "line_start": 0, "line_end": 0,
                "heading_path": r["name"],
                "locator": f'{r["kind"]} {r["id"]}',
                # 沒有 url 時不要給 null：前端會畫出 href=".../null" 的死連結。
                "url": r.get("url") or "",
                "score": 100.0, "license": r.get("license", ""), "corpus": "entities",
            } for r in resolved],
            "lang": lang, "grounded": True, "kind": "entity-lookup",
        }

    def _machine_specific(self, question, lang):
        """「我這台電腦現在幾度／我的配置是什麼」——網站答不了，但可以指出桌面版讀哪裡。

        這種問題最危險：文件裡有大量「溫度」「感測器」的段落，檢索一定命中，
        模型很可能順手把文件裡的**範例數字**寫成「你的讀值」。所以不走一般問答，
        直接給一段固定的、不可能被誤讀成讀值的回答，並附上真的講到讀取路徑的段落。

        兩種問法要分開答：問「溫度／快慢」給感測器那幾段；問「配置／型號」給
        「我的電腦」頁與 asset./cpu./gpu./mem./sb. 這幾組事實鍵。用同一段回答，
        問配置的人會拿到一段講溫度的文字，等於沒回答。
        """
        # 先看是不是在問「讀值」（溫度、瓦數、風扇轉速）。是的話走感測器那一段，
        # 即使句子裡出現了 CPU／顯示卡這種零件名——「我的 CPU 幾度」問的不是型號。
        is_reading = bool(re.search(
            r"幾度|几度|溫度|温度|多熱|多热|幾瓦|几瓦|跑幾分|跑几分|風扇|风扇|轉速|转速"
            r"|how hot|temperature|celsius|fan speed|\brpm\b", question, re.I))
        is_config = (not is_reading) and bool(re.search(
            r"配置|規格|规格|配備|配备|型號|型号|硬體清單|硬件清单|spec|config"
            r"|顯示卡|显卡|記憶體|内存|硬碟|硬盘|固態|固态|主機板|主板|處理器|处理器|電源|电源"
            r"|\b(gpu|cpu|ram|memory|storage|drive|board)\b", question, re.I))
        if is_config:
            queries = ("我的電腦 配置 硬體 清單 主機板 記憶體",
                       "系統資訊 主機板 記憶體 顯示卡 裝置清單")
            body = {
                "zh-Hant": (
                    "這個網站讀不到你的電腦，所以沒有你的配置清單——"
                    "任何看起來像「你的規格」的數字都不可能是你的實測。\n\n"
                    "你自己的配置在桌面版的「我的電腦」那一頁：CPU 型號與代數、主機板與 BIOS、"
                    "記憶體容量與插槽、顯示卡與驅動版本、儲存裝置、作業系統版本。"
                    "每一項都對應一把事實鍵（asset.identify／cpu.*／gpu.*／mem.*／sb.type*／os.*），"
                    "讀不到的會寫讀不到的原因，不會填 0。\n\n"
                    "如果你要問的是「曦覽要什麼配置才跑得動」，那是文件裡真的有寫的內容，"
                    "直接問「曦覽的系統需求」就有答案。"
                ),
                "zh-Hans": (
                    "这个网站读不到你的电脑，所以没有你的配置清单——"
                    "任何看起来像「你的规格」的数字都不可能是你的实测。\n\n"
                    "你自己的配置在桌面版的「我的电脑」那一页：CPU 型号与代数、主板与 BIOS、"
                    "内存容量与插槽、显卡与驱动版本、存储设备、操作系统版本。"
                    "每一项都对应一把事实键（asset.identify／cpu.*／gpu.*／mem.*／sb.type*／os.*），"
                    "读不到的会写读不到的原因，不会填 0。\n\n"
                    "如果你要问的是「曦览要什么配置才跑得动」，那是文档里真的有写的内容，"
                    "直接问「曦览的系统需求」就有答案。"
                ),
                "en": (
                    "This website cannot read your computer, so there is no spec sheet for it here, "
                    "and no number that looks like your configuration can be yours.\n\n"
                    "Your own configuration is on the desktop app's \"My PC\" page: CPU model and "
                    "generation, board and BIOS, memory size and slots, GPU and driver version, storage, "
                    "OS build. Each row maps to a fact key (asset.identify / cpu.* / gpu.* / mem.* / "
                    "sb.type* / os.*); anything unreadable reports the reason instead of a zero.\n\n"
                    "If you meant \"what does XinSpect need to run\", that is in the documentation: "
                    "ask for the system requirements."
                ),
            }[lang]
        else:
            queries = ("溫度 感測器 讀取 路徑", "temperature sensor read path")
            body = {
                "zh-Hant": (
                    "這個網站讀不到你的電腦。官網只呈現說明，不掃描任何機器——"
                    "所以「你這台現在幾度、跑多快、健康不健康」在這裡沒有答案，"
                    "任何看起來像讀值的數字都不可能是你的實測。\n\n"
                    "要拿到你自己機器的讀值，得在 Windows 上跑曦覽桌面版。"
                    "它把同一項性質分成硬體、韌體、作業系統三份陳述並列，讀不到的會給原因而不是給 0。"
                    "下面幾段是文件裡實際寫到讀取路徑的地方，可以對著看桌面版哪一頁會出現你要的數字。"
                ),
                "zh-Hans": (
                    "这个网站读不到你的电脑。官网只呈现说明，不扫描任何机器——"
                    "所以「你这台现在几度、跑多快、健康不健康」在这里没有答案，"
                    "任何看起来像读值的数字都不可能是你的实测。\n\n"
                    "要拿到你自己机器的读值，得在 Windows 上跑曦览桌面版。"
                    "它把同一项性质分成硬件、固件、操作系统三份陈述并列，读不到的会给原因而不是给 0。"
                    "下面几段是文档里实际写到读取路径的地方，可以对着看桌面版哪一页会出现你要的数字。"
                ),
                "en": (
                    "This website cannot read your computer. The site only describes the software; it never "
                    "scans a machine. So there is no answer here for \"how hot or fast is my PC right now\", "
                    "and no number that looks like a reading can be yours.\n\n"
                    "To get readings from your own machine, run the XinSpect desktop app on Windows. It lays "
                    "the hardware, firmware and OS accounts of the same property side by side, and gives a "
                    "reason instead of a zero when something cannot be read. The passages below are where the "
                    "documentation actually describes the read paths."
                ),
            }[lang]

        # 這幾個查詢是為了找出「桌面版怎麼讀這件事」的段落，不是為了回答問題本身
        hits = []
        for q in queries:
            hits.extend(self.index.search(q, top_k=3))
        seen = set()
        cites = []
        for h in hits:
            key = (h["source"], h["line_start"])
            if key in seen:
                continue
            seen.add(key)
            cites.append({
                "source": h["source"], "line_start": h["line_start"],
                "line_end": h["line_end"], "heading_path": h["heading_path"],
                "url": h["url"], "score": round(h["score"], 2),
            })
            if len(cites) >= 4:
                break

        return {"answer": body, "citations": cites, "lang": lang,
                "grounded": True, "kind": "machine-specific-config" if is_config else "machine-specific"}


class Handler(BaseHTTPRequestHandler):
    app = None            # type: App
    site_dir = None       # 有值才開靜態預覽
    max_prompt_chars = 24000
    ip_calls = {}
    # 預設 60 次/小時/IP。端到端評測要一次打完 78 題，所以留一個環境變數 override——
    # 只給本機評測用；正式部署不要設，否則這層防連發就等於沒有。
    ip_per_hour = int(os.environ.get("XINSPECT_RATE_LIMIT", "60"))

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        # 本機預覽時，"/" 是網站首頁而不是健康檢查——否則瀏覽器打開只會看到一段 JSON。
        if getattr(self, "site_dir", None) and path == "/":
            return self._serve_static(path)
        if path in ("/", "/health"):
            return self._json({"ok": True, "service": "xinspect-ai",
                               "model": self.app.upstream.model,
                               "upstream": self.app.upstream.kind,
                               "chunks": len(self.app.index.docs),
                               "externalChunks": len(self.app.external.docs) if self.app.external else 0,
                               "entities": self.app.entities.available})
        if path == "/v1/models":
            return self._json({"object": "list", "data": [
                {"id": "xinspect-ai", "object": "model", "owned_by": "xinspect"}]})
        # 本機預覽：把 web/preview/ 當靜態目錄送出，只在使用者明確開啟 --serve-site 時才有。
        if getattr(self, "site_dir", None):
            return self._serve_static(path)
        return self._json({"error": {"message": "no such path"}}, 404)

    def _serve_static(self, path):
        import mimetypes
        rel = "index-live.html" if path in ("/", "") else path.lstrip("/")
        # 防目錄穿越：一律解析成絕對路徑再確認還在站台資料夾底下
        target = os.path.realpath(os.path.join(self.site_dir, rel))
        root = os.path.realpath(self.site_dir)
        if not target.startswith(root + os.sep) and target != root:
            return self._json({"error": {"message": "forbidden"}}, 403)
        if not os.path.isfile(target):
            return self._json({"error": {"message": "not found"}}, 404)
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        data = open(target, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json({"error": {"message": "bad json"}}, 400)

        if path == "/v1/chat/completions":
            return self.chat_completions(body)
        if path == "/v1/xinspect/ask":
            return self.ask(body)
        return self._json({"error": {"message": "no such path"}}, 404)

    # 站方專用：回 JSON，帶 citations
    def ask(self, body):
        q = (body.get("question") or body.get("q") or "").strip()
        if not q:
            return self._json({"error": {"message": "question 是空的"}}, 400)
        if len(q) > 2000:
            return self._json({"error": {"message": "問題太長"}}, 400)
        if not self._allow():
            return self._json({"error": {"message": "使用太頻繁，請稍後再試。"}}, 429)

        if body.get("stream"):
            return self.ask_stream(q, body)
        result = self.app.answer(q, lang=body.get("lang"), ui_lang=body.get("uiLang"))
        status = 502 if "error" in result else 200
        return self._json(result, status)

    def ask_stream(self, q, body):
        """SSE 串流。事件型別有三種：

          event: meta     → 這一輪的語言與出處（先送，前端可以先畫出「正在查」）
          event: delta    → 逐字內容
          event: done     → 結束（帶 grounded 與最終出處）

        查表／拒答這類不需要模型的情況，整段內容一次當一個 delta 送出——
        前端不必為此寫兩條路徑。
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

        def emit(event, payload):
            frame = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            self.wfile.write(frame.encode("utf-8"))
            self.wfile.flush()

        try:
            plan = self.app.prepare(q, lang=body.get("lang"), ui_lang=body.get("uiLang"))
            if "early" in plan:
                early = plan["early"]
                emit("meta", {"lang": early.get("lang"), "citations": early.get("citations", []),
                              "kind": early.get("kind"), "grounded": early.get("grounded")})
                emit("delta", {"text": early.get("answer", "")})
                emit("done", {"grounded": early.get("grounded", False),
                              "citations": early.get("citations", [])})
                return

            sys_prompt, user = plan["prompt"]
            lang = plan["lang"]
            emit("meta", {"lang": lang, "citations": plan["cites"], "grounded": True})

            status, stream = self.app.upstream.chat_stream(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": user}])
            if status != 200:
                emit("error", {"message": f"上游回應 {status}：{stream}"})
                return

            buf = []
            for piece in stream:
                buf.append(piece)
                emit("delta", {"text": piece})

            # 後處理（例如「假裝讀到你的機器」的更正）在串流下不能用「換掉整段」的方式做，
            # 前端已經收到原文了。所以只送**追加的那一句**：postprocess 對這種句形
            # 一定是「原文 + 更正」，把尾巴那一段當額外的 delta 送出即可。
            raw = "".join(buf)
            fixed = postprocess(raw, lang)
            extra = fixed[len(raw):] if fixed.startswith(raw) else ""
            if extra:
                emit("delta", {"text": extra})
            emit("done", {"grounded": True, "citations": plan["cites"],
                          "corrected": bool(extra)})
        except Exception as e:
            try:
                emit("error", {"message": f"{type(e).__name__}: {e}"})
            except Exception:
                pass

    def _allow(self):
        """每個來源 IP 每小時的次數上限。

        為什麼放在服務這一側而不是只靠 Worker：這個服務可能直接掛在網站同源後面，
        那時前面沒有 Cloudflare 的限流可用。用一個行程內的 dict 就夠——單一實例、
        重啟即清空，正好符合「這不是帳務系統，只是防連發」的定位。
        """
        ip = self.client_address[0]
        now = time.time()
        window = Handler.ip_calls.setdefault(ip, [])
        window[:] = [t for t in window if now - t < 3600]
        if len(window) >= self.ip_per_hour:
            return False
        window.append(now)
        return True

    # OpenAI 相容：桌面版／任何客端可以直接指過來
    def chat_completions(self, body):
        messages = body.get("messages") or []
        if not messages:
            return self._json({"error": {"message": "messages 是空的"}}, 400)
        question = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                question = (m.get("content") or "").strip()
                break
        if not question:
            return self._json({"error": {"message": "找不到使用者訊息"}}, 400)
        question = question[: self.max_prompt_chars]

        lang = body.get("lang") or detect_lang(question)
        result = self.app.answer(question, lang=lang)
        if "error" in result:
            return self._json({"error": {"message": result["error"]}}, 502)

        text = result["answer"]
        if result.get("citations"):
            sep = "\n\n出處：\n" if lang != "en" else "\n\nSources:\n"
            text += sep + "\n".join(
                f'- `{c["source"]}:{c["line_start"]}` {c["heading_path"]}'
                for c in result["citations"][:5])

        created = int(time.time())
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()
            # 本服務先檢索再生成，並非真串流；這裡把已完成的一段切成小塊送出，
            # 讓客端的逐字介面照樣有東西可畫——並在最後附上出處。
            chunk = 24
            for i in range(0, len(text), chunk):
                frame = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                         "created": created, "model": "xinspect-ai",
                         "choices": [{"index": 0, "delta": {"content": text[i:i + chunk]},
                                      "finish_reason": None}]}
                self.wfile.write(("data: " + json.dumps(frame, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            done = {"id": f"chatcmpl-{created}", "object": "chat.completion.chunk",
                    "created": created, "model": "xinspect-ai",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(("data: " + json.dumps(done, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        return self._json({
            "id": f"chatcmpl-{created}",
            "object": "chat.completion",
            "created": created,
            "model": "xinspect-ai",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "xinspect": {"lang": result["lang"], "grounded": result.get("grounded"),
                         "citations": result.get("citations", [])},
        })


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=DEFAULT_INDEX)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--upstream", default="ollama", choices=["ollama", "workers-ai", "openai"])
    ap.add_argument("--base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--key", default=os.environ.get("XINSPECT_UPSTREAM_KEY"))
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--min-answerability", type=float, default=MIN_ANSWERABILITY)
    ap.add_argument("--external-index", default=os.path.join(HERE, "..", "rag", "index", "external"),
                    help="外部硬體語料的索引；不存在就只用曦覽文件")
    ap.add_argument("--no-external", action="store_true", help="停用外部語料（只答曦覽文件）")
    ap.add_argument("--entities-db", default=os.path.join(HERE, "..", "corpus", "entities.db"))
    ap.add_argument("--serve-site", default=None,
                    help="把一個目錄當靜態網站送出（本機預覽官網改動用），預設關閉")
    args = ap.parse_args()

    defaults = {
        "ollama": ("http://127.0.0.1:11434/v1", "qwen3:4b"),
        "workers-ai": ("https://xinspect-ai.xinspect-tools.workers.dev/v1", "auto"),
        "openai": ("https://api.openai.com/v1", "gpt-4o-mini"),
    }[args.upstream]
    base = args.base or defaults[0]
    model = args.model or defaults[1]

    idx = BM25Index.load(args.index)
    ext = None
    if not args.no_external and os.path.exists(args.external_index + ".meta.json"):
        ext = BM25Index.load(args.external_index)

    app = App(idx, Upstream(args.upstream, base, model, args.key),
              top_k=args.top_k, min_answerability=args.min_answerability,
              external_idx=ext, entities=EntityStore(args.entities_db))
    Handler.app = app
    if args.serve_site:
        Handler.site_dir = os.path.abspath(args.serve_site)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"XinSpect-Ai listening on http://{args.host}:{args.port}")
    print(f"  product : {len(idx.docs)} chunks")
    print(f"  external: {len(ext.docs) if ext else 0} chunks")
    print(f"  entities: {app.entities.stats() if app.entities.available else '（未載入）'}")
    print(f"  upstream: {args.upstream} {base} model={model}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
