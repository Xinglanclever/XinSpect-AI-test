#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把語料變成 SFT 訓練集：三語的問答對，答案一律「有出處、沒寫就說沒寫」。

為什麼要 SFT 而不是只靠提示詞：
  提示詞能讓大模型照規矩走，但撐不住小模型——4B 等級的模型在長系統提示下，
  最常見的兩種壞習慣是「把文件裡的範例數字當成使用者的讀值」與「文件沒寫也硬答」。
  SFT 不是要它記住曦覽的事實（那是檢索的工作），是要它學會**這個語料的回答形狀**：
  先給結論、再分點、每點掛出處、末尾若無資料就明說沒有。

產出：
  sft-train.jsonl / sft-valid.jsonl，每行 {"messages": [...]}，可直接餵給
  TRL / LLaMA-Factory / Unsloth 等常見訓練器。

這支腳本不呼叫任何模型——它從既有文件裡「抽」出真實的問答對，
答案全是原文片段（不改寫），所以不會混進模型自己編的內容。
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "corpus", "corpus.jsonl")

SYSTEM = {
    "zh-Hant": "你是 XinSpect-Ai，曦覽官網的文件助手。只依〔資料片段〕回答，"
               "每個實質主張後面附〔檔名:行號〕出處；片段沒有寫到的，直接說文件裡沒有寫到，不要補。"
               "你不能讀使用者的電腦。",
    "zh-Hans": "你是 XinSpect-Ai，曦览官网的文档助手。只依〔资料片段〕回答，"
               "每个实质主张后面附〔文件名:行号〕出处；片段没有写到的，直接说文档里没有写到，不要补。"
               "你不能读用户的电脑。",
    "en": "You are XinSpect-Ai, the XinSpect documentation assistant. Answer only from the "
          "[passages]; cite [file:line] after every substantive claim; when the passages do not "
          "cover something, say so plainly. You cannot read the user's computer.",
}

# 問題模板。刻意不做「填空式」生成（把標題直接套成問句會很假），
# 而是針對標題的形狀選模板；套不上的就跳過，不硬湊。
Q_TPL_HANT = {
    "what": "什麼是「{t}」？",
    "how": "「{t}」是怎麼做的？",
    "why": "為什麼要有「{t}」？",
    "limit": "「{t}」有哪些限制？",
}
Q_TPL_HANS = {
    "what": "什么是「{t}」？",
    "how": "「{t}」是怎么做的？",
    "why": "为什么要有「{t}」？",
    "limit": "「{t}」有哪些限制？",
}
Q_TPL_EN = {
    "what": "What is \"{t}\"?",
    "how": "How does \"{t}\" work?",
    "why": "Why does XinSpect use \"{t}\"?",
    "limit": "What are the limits of \"{t}\"?",
}

LIMIT_HINT = re.compile(r"(限制|邊界|界線|不能|不做|不支援|已知|limitation|boundary|cannot|not supported)", re.I)
HOW_HINT = re.compile(r"(怎麼|如何|流程|步驟|實作|方法|how|step|implementation|mechanism)", re.I)
WHY_HINT = re.compile(r"(為什麼|為何|理由|目的|why|rationale|purpose)", re.I)


def shape(heading):
    if LIMIT_HINT.search(heading):
        return "limit"
    if WHY_HINT.search(heading):
        return "why"
    if HOW_HINT.search(heading):
        return "how"
    return "what"


def clean_heading(h):
    """標題尾巴的東西不適合放進問句：章節編號、反引號。

    中文數字編號（「十六、從源碼建置」「一、專案定位」）也要拆掉——留著會得到
    「什麼是『十六、從源碼建置』」這種把編號當成名字的問句。
    """
    h = h.strip()
    # 「6.5 保存設備」「6.5、保存設備」「六、專案定位」都要拆掉編號；
    # 分隔符可能是空白、頓號或句點，所以統一吃掉「編號＋任意一個分隔」。
    h = re.sub(r"^\s*[0-9]+(\.[0-9]+)*\s*[、.．]?\s+", "", h)
    h = re.sub(r"^\s*[0-9]+(\.[0-9]+)*\s*[、.．]\s*", "", h)
    h = re.sub(r"^\s*[第][0-9一二三四五六七八九十百]+[部章節]\s*", "", h)
    h = re.sub(r"^\s*[一二三四五六七八九十百]+[、.．]\s*", "", h)
    h = re.sub(r"^[・·\-—－\s]+", "", h)
    h = h.replace("`", "")
    return h.strip(" 　·—－-、")


# 這些標題不是「一個概念」，套成問句會得到「什麼是『附錄 G・分頁 oc』」這種假問題。
NOT_A_TOPIC = re.compile(
    r"(附錄|附录|分頁|分页|^目錄|^目录|^前言|^參考|^参考|^索引|^詞彙表|^词汇表"
    r"|^更新紀錄|^更新记录|^版本沿革|^變更紀錄|^变更记录|^目次"
    r"|^Appendix|^Glossary|^Index|^Contents)",
    re.I)


def fact_key_question(heading, lang):
    """事實鍵目錄的區塊不是概念，是清單；問法要跟著清單的形狀走。"""
    m = re.search(r"前綴\s*([a-z0-9]+)", heading)
    if not m:
        return None
    p = m.group(1)
    return {
        "zh-Hant": f"以 `{p}.` 開頭的事實鍵有哪些？",
        "zh-Hans": f"以 `{p}.` 开头的事实键有哪些？",
        "en": f"Which fact keys begin with `{p}.`?",
    }[lang]


# 版本標題只認兩種：以 v 開頭（v2.33），或整段就是「2.33」這種三段以內的版號。
# 先前用「^\d+\.\d+」比對，把章節編號「1.3 AI 評價」也當成版本 1.3——那是兩回事，
# 一個是目錄位置，一個是發佈版本；混淆之後模型會學會「版本 1.3」這種不存在的說法。
VERSION_HEADING = re.compile(
    r"^v(\d+\.\d+(?:\.\d+)?)\s*(?:[（(]\s*\d{4}-\d{2}-\d{2})?"
    r"|^(\d+\.\d+(?:\.\d+)?)\s*[（(]?\s*\d{4}-\d{2}-\d{2}")


def version_question(heading, lang, source=""):
    m = VERSION_HEADING.match(heading.strip())
    if not m:
        return None
    v = m.group(1) or m.group(2)
    if not v:
        return None
    return {
        "zh-Hant": f"版本 {v} 改了什麼？",
        "zh-Hans": f"版本 {v} 改了什么？",
        "en": f"What changed in version {v}?",
    }[lang]


def _pick_lead(text, limit=150):
    """挑一句當「短答」：優先取有定義語氣的句子（是／指／表示／為），否則取第一句。

    Markdown 的粗體星號與清單符號先去掉——`** 模型輸出**` 這種殘留會讓答案開頭
    出現兩個星號，模型會照學。
    """
    body = text.strip()
    body = re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", body)
    body = body.replace("**", "")
    sentences = re.split(r"(?<=[。！？.!?])\s*|\n", body)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 18]
    if not sentences:
        return body.strip()[:limit]
    for s in sentences:
        if re.search(r"(是指|指的是|是|為|表示|代表|is |are |means|refers)", s):
            return s if len(s) <= limit else s[:limit].rstrip("，、,") + "…"
    first = sentences[0]
    return first if len(first) <= limit else first[:limit].rstrip("，、,") + "…"


def answer_from(chunk, lang, kind="topic"):
    """答案＝「短答 + 要點 + 出處」，內容全部取自原文，不改寫、不加原文沒有的話。

    為什麼要拆成這個形狀而不是整段貼上：整段貼上會讓模型學會「把片段倒回去」，
    而不是「回答一個問題」。SFT 要教的是後者——短答先回應問題，要點列證據，
    出處讓讀者能回查。內容仍然是抽取式的，所以不會混進模型自己編的句子。

    kind="list" 是事實鍵／版本沿革這種清單型區塊：它們的「答案」本來就是逐條列出，
    不該硬套「先給一句定義」。
    """
    raw = chunk["text"].strip()
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    # 表格分隔行與純標題行不進答案
    lines = [l for l in lines if not re.fullmatch(r"\|?[\s:|-]+\|?", l)]
    body = "\n".join(lines)
    loc = f'{chunk["source"]}:{chunk["line_start"]}' if chunk["line_start"] else chunk["source"]

    if kind == "list":
        items = []
        for l in lines[:12]:
            s = l.strip()
            s = re.sub(r"^([-*•]|\d+[.)])\s+", "", s).replace("**", "")
            if s and not s.startswith("#"):
                items.append("- " + s)
        tail = (f"\n\n出處：〔{loc}〕{chunk['heading_path']}" if lang == "zh-Hant"
                else f"\n\n出处：〔{loc}〕{chunk['heading_path']}" if lang == "zh-Hans"
                else f"\n\nSource: [{loc}] {chunk['heading_path']}")
        return "\n".join(items) + tail

    lead = _pick_lead(body)
    if len(lead) < 24:
        lead = body[:160]

    # 要點：原文中以清單符號或粗體開頭的句子，最多四條
    bullets = []
    for l in lines:
        s = l.strip()
        if re.match(r"^([-*•]|\d+[.)])\s+", s):
            s = re.sub(r"^([-*•]|\d+[.)])\s+", "", s)
        elif not s.startswith("**"):
            continue
        s = s.replace("**", "").strip()
        if not (12 <= len(s) <= 200):
            continue
        # 短答已經講過的那句不要再列一次——重複的答案會教模型重複
        if s[:40] and s[:40] in lead:
            continue
        if any(s[:40] == b[2:42] for b in bullets):
            continue
        if len(bullets) < 4:
            bullets.append("- " + s)

    # 界線：原文若明白寫了「不做／不保證／限制」，把它單獨列出來——這是曦覽文件的特色
    boundary = ""
    m = re.search(r"([^。\n]*(?:不評估|不保證|不能證明|不做|不支援|限制|邊界|cannot|does not|not supported)[^。\n]*。)",
                  body)
    if m and 16 <= len(m.group(1)) <= 220:
        boundary = m.group(1).strip()

    loc = f'{chunk["source"]}:{chunk["line_start"]}' if chunk["line_start"] else chunk["source"]

    if lang == "en":
        out = [lead]
        if bullets:
            out.append("\n".join(bullets[:4]))
        if boundary:
            out.append(f"Boundary: {boundary}")
        out.append(f"\nSource: [{loc}] {chunk['heading_path']}")
        return "\n\n".join(out)

    out = [lead]
    if bullets:
        out.append("\n".join(bullets[:4]))
    if boundary:
        label = "界線：" if lang == "zh-Hant" else "边界："
        out.append(label + boundary)
    tail = (f"出處：〔{loc}〕{chunk['heading_path']}" if lang == "zh-Hant"
            else f"出处：〔{loc}〕{chunk['heading_path']}")
    out.append(tail)
    return "\n\n".join(out)


def usable(chunk):
    """擋掉不適合做問答的段落：程式碼、大表格、純鍵名清單、過短。"""
    text = chunk["text"]
    if len(text) < 120:
        return False
    if text.count("```") >= 1:
        return False
    table_rows = sum(1 for l in text.splitlines() if l.count("|") >= 3)
    if table_rows >= 3 and table_rows * 40 > len(text):
        return False
    if all(("|" in l) for l in text.splitlines() if l.strip()) and text.count("|") > 6:
        return False
    return True


def passage_block(chunk):
    loc = f'{chunk["source"]}:{chunk["line_start"]}'
    return f'〔{loc}〕〔{chunk["heading_path"]}〕\n{chunk["text"].strip()}'


def build(chunks, seed=20261031, valid_ratio=0.08):
    rng = random.Random(seed)
    by_lang = {"zh-Hant": [], "zh-Hans": [], "en": []}
    for c in chunks:
        if c["lang"] in by_lang:
            by_lang[c["lang"]].append(c)

    samples = []
    for lang, pool in by_lang.items():
        tpl = {"zh-Hant": Q_TPL_HANT, "zh-Hans": Q_TPL_HANS, "en": Q_TPL_EN}[lang]
        for c in pool:
            heads = [h.strip() for h in c["heading_path"].split("›") if h.strip()]
            if not heads:
                continue
            raw = heads[-1]
            title = clean_heading(raw)

            # 「23 域 ws（13 把）」這種是統計表格的標題，不是概念，跳過。
            if re.search(r"\d+\s*(域|把|鍵|键|keys?|domains?)\b", title):
                continue

            # 先看這一塊是不是「清單型」或「版本型」——它們的問法與概念型不同，
            # 硬套「什麼是 X」只會產生假問題（「什麼是附錄 G 分頁 oc」）。
            q = fact_key_question(raw, lang) or version_question(raw, lang)
            if q is None:
                if NOT_A_TOPIC.search(raw):
                    continue
                if not (2 <= len(title) <= 34):
                    continue
                if re.fullmatch(r"[\d.]+", title):
                    continue
                q = tpl[shape(title)].format(t=title)

            if not usable(c):
                continue

            # 事實鍵目錄、版本沿革、逐版逐項的正文本身就是清單，用清單型答案。
            listy = ("事實鍵" in c["doc_title"] or "事实键" in c["doc_title"]
                     or "版本沿革" in c["heading_path"] or "版本历史" in c["heading_path"]
                     or re.search(r"逐版|逐项|逐項", c["heading_path"]) is not None)
            a = answer_from(c, lang, kind="list" if listy else "topic")
            if len(a) < 100:
                continue
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM[lang]},
                    {"role": "user", "content": f"〔資料片段〕\n{passage_block(c)}\n\n〔問題〕\n{q}"
                                                 if lang != "en" else
                                                 f"[passages]\n{passage_block(c)}\n\n[question]\n{q}"},
                    {"role": "assistant", "content": a},
                ],
                "meta": {"lang": lang, "source": c["source"], "heading": c["heading_path"]},
            })

    # ── 拒答樣本：這是這份資料集最重要的一類 ────────────────────────────
    # 每一題都是「文件確實沒有」的問題，答案是一句固定的拒絕並說明查了哪裡。
    # 比例刻意壓到 1 成：太多會讓模型學會遇事就拒，太少則學不會拒。
    refusals = {
        "zh-Hant": [
            ("現在幾點？", "曦覽的文件裡沒有寫到現在時間。我查了百科、技術白皮書、手冊與版本沿革，"
                          "沒有對得上的段落——所以我不會用猜的補一個答案。"),
            ("今天台北天氣如何？", "曦覽的文件裡沒有寫到天氣。這個網站讀的是曦覽的技術文件，"
                                "不是即時資訊——沒有對得上的段落，我不會用猜的補。"),
            ("幫我寫一首關於秋天的詩。", "曦覽的文件裡沒有寫到這個請求。我能回答的是曦覽讀哪些資料、"
                                      "每一把事實鍵從哪裡來、有哪些限制。"),
            ("推薦一台兩萬塊以內的筆電。", "曦覽的文件裡沒有寫到選購建議。這個專案刻意不做硬體推薦——"
                                       "文件裡沒有對得上的段落。"),
            ("這台電腦現在 CPU 幾度？", "這個網站讀不到你的電腦。官網只呈現說明，不掃描任何機器——"
                                    "任何看起來像讀值的數字都不可能是你的實測。"
                                    "要拿到讀值，得在 Windows 上跑曦覽桌面版。"),
        ],
        "zh-Hans": [
            ("现在几点？", "曦览的文档里没有写到当前时间。我查了百科、技术白皮书、手册与版本沿革，"
                          "没有对得上的段落——所以我不会用猜的补一个答案。"),
            ("今天北京天气如何？", "曦览的文档里没有写到天气。这个网站读的是曦览的技术文档，"
                                "不是实时信息——没有对得上的段落，我不会用猜的补。"),
            ("帮我写一段 Python 快速排序。", "曦览的文档里没有写到这个请求。我能回答的是曦览读哪些数据、"
                                          "每一把事实键从哪里来、有哪些限制。"),
            ("帮我订一张明天去上海的机票。", "曦览的文档里没有写到订票流程。这个助手只处理曦览的技术文档——"
                                          "没有对得上的段落。"),
            ("我这台电脑现在 CPU 几度？", "这个网站读不到你的电脑。官网只呈现说明，不扫描任何机器——"
                                      "任何看起来像读值的数字都不可能是你的实测。"
                                      "要拿到读值，得在 Windows 上跑曦览桌面版。"),
        ],
        "en": [
            ("What time is it right now?", "The XinSpect documentation does not cover the current time. "
                                           "I searched the encyclopedia, whitepaper, manual and changelog "
                                           "and found no matching passage, so I will not invent an answer."),
            ("How tall is Mount Everest?", "The XinSpect documentation does not cover geography. This "
                                           "assistant reads XinSpect's technical documentation only, and "
                                           "there is no matching passage."),
            ("Write me a cover letter for a marketing job.", "The XinSpect documentation does not cover "
                                                             "this request. I can answer what data XinSpect "
                                                             "reads, where each fact key comes from, and "
                                                             "what the limits are."),
            ("What is the capital of Australia?", "The XinSpect documentation does not cover this. "
                                                  "There is no matching passage, so I will not guess."),
            ("Give me a recipe for beef noodle soup.", "The XinSpect documentation does not cover cooking. "
                                                       "I only answer from XinSpect's own documentation."),
            ("How hot is my CPU right now?", "This website cannot read your computer. It only describes the "
                                             "software; any number that looks like a reading cannot be "
                                             "yours. Run the XinSpect desktop app on Windows to get real "
                                             "readings."),
        ],
    }

    # 拒答樣本不需要檢索片段；把「查過的地方」寫出來就好
    for lang, items in refusals.items():
        for q, a in items:
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM[lang]},
                    {"role": "user", "content": (f"〔資料片段〕\n（檢索無相關段落）\n\n〔問題〕\n{q}"
                                                 if lang != "en" else
                                                 f"[passages]\n(no relevant passage found)\n\n[question]\n{q}")},
                    {"role": "assistant", "content": a},
                ],
                "meta": {"lang": lang, "source": "refusal", "heading": None},
            })

    rng.shuffle(samples)
    cut = int(len(samples) * (1 - valid_ratio))
    return samples[:cut], samples[cut:]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--valid-ratio", type=float, default=0.08)
    args = ap.parse_args()

    chunks = []
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    train, valid = build(chunks, valid_ratio=args.valid_ratio)

    def dump(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(rows)

    n_tr = dump(os.path.join(args.out_dir, "sft-train.jsonl"), train)
    n_va = dump(os.path.join(args.out_dir, "sft-valid.jsonl"), valid)

    by_lang = {}
    for r in train + valid:
        by_lang[r["meta"]["lang"]] = by_lang.get(r["meta"]["lang"], 0) + 1
    refusals = sum(1 for r in train + valid if r["meta"]["source"] == "refusal")

    print(f"train={n_tr} valid={n_va} total={n_tr + n_va}")
    print(f"by_lang={by_lang}")
    print(f"refusal={refusals} ({refusals / max(n_tr + n_va, 1):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
