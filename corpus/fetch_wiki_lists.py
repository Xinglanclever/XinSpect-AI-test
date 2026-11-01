#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Wikipedia 上「型號清單」頁面的 wikitext 抓下來，存成 .wiki 檔。

為什麼要自己抓，而不是只吃既有語料：
  既有語料（documents.jsonl／raw）只有 288 份文件，很多型號清單沒被收進去。
  但「i7-6700K 的插槽是什麼」這種問題，答案就在這些清單的表格裡；
  沒有原始表格，就只能讓模型憑記憶講——那是我們最不想發生的事。

為什麼抓 wikitext 而不是渲染後的 HTML：
  渲染版會把 rowspan 展開、把註腳編號塞進型號裡；wikitext 是原始資料，
  `rowspan="2" | i7-12700K` 這種寫法自己解才拿得到完整的型號。

禮貌原則（Wikipedia API 是公共資源）：
  - 每頁間隔 --delay 秒（預設 2），請求帶可辨識的 User-Agent；
  - 失敗（429／IncompleteRead／timeout）退避重試，最多 --retries 次；
  - 已存在且非空的 .wiki 檔直接跳過，所以中斷後重跑會接續。

用法：
  python corpus/fetch_wiki_lists.py --out corpus/raw-fetched
  python corpus/fetch_wiki_lists.py --out corpus/raw-fetched --dry-run   # 只列清單
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse

API = "https://en.wikipedia.org/w/api.php"
UA = ("XinSpectAI/0.1 (documentation assistant; "
      "+https://github.com/Xinglanclever/XinSpect)")

# 這個網路環境下 Python 的 OpenSSL 握手會被中途切斷（SSLEOFError），
# 但 curl（Windows 走 Schannel）每次都成功，所以一律用 curl 抓，Python 只負責解析。
CURL = ["curl", "-sS", "--compressed", "--max-time", "150",
        "-A", UA, "-H", "Accept: application/json"]


def _curl_json(url, retries):
    last = None
    for attempt in range(retries):
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            r = subprocess.run(CURL + ["-o", tmp, "-w", "%{http_code}", url],
                               capture_output=True, text=True, timeout=180)
            code = (r.stdout or "").strip()
            if code == "200":
                with open(tmp, encoding="utf-8") as fh:
                    return json.loads(fh.read())
            last = f"HTTP {code or '???'} {r.stderr.strip()[:120]}"
        except Exception as e:                      # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        time.sleep(3 + attempt * 5)
    raise RuntimeError(f"重試 {retries} 次仍失敗：{last}")


def api(params, timeout=90, retries=5):
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    return _curl_json(API + "?" + urllib.parse.urlencode(params), retries)


# 這些前綴底下的頁面就是「型號清單」。Table of 系列是 AMD 的萬年大表。
# 「List of Arc／List of Apple」這種短前綴會撈到 Arctic Monkeys、Archbishop…，
# 所以只留夠長、明確指向硬體的前綴；缺的頁放 EXPLICIT。
PREFIXES = [
    "List of Intel", "List of AMD", "List of Nvidia", "List of Radeon",
    "List of GeForce", "List of Qualcomm Snapdragon",
    "Table of AMD", "Table of Intel", "Table of Nvidia",
]
# 直接指定、前綴撈不到的頁：世代／家族頁本身就是一整張型號表。
EXPLICIT = [
    "Apple silicon", "Epyc",
    # AMD 世代頁（「Ryzen 5000」這種標題，不在 List of 前綴底下）
    "Ryzen 1000", "Ryzen 2000", "Ryzen 3000", "Ryzen 4000", "Ryzen 5000",
    "Ryzen 6000", "Ryzen 7000", "Ryzen 8000", "Ryzen 9000", "Ryzen AI 300",
    # Intel 世代頁
    "Alder Lake", "Raptor Lake", "Meteor Lake", "Lunar Lake", "Arrow Lake",
    "Panther Lake", "Tiger Lake", "Ice Lake (microprocessor)", "Coffee Lake",
    "Kaby Lake", "Comet Lake", "Rocket Lake", "Skylake", "Broadwell",
    "Haswell (microarchitecture)", "Ivy Bridge", "Sandy Bridge",
    # GPU 各世代頁（型號表在「GeForce 30 series」這類文章裡，不在 List of 底下）
    "GeForce 10 series", "GeForce 16 series", "GeForce 20 series",
    "GeForce 30 series", "GeForce 40 series", "GeForce 50 series",
    "Radeon RX 400 series", "Radeon RX 500 series", "Radeon RX Vega series",
    "Radeon RX 5000 series", "Radeon RX 6000 series", "Radeon RX 7000 series",
    "Radeon RX 9000 series",
    # 架構頁：Zen／RDNA 的 SKU 表常只放在這裡
    "Zen 1", "Zen 2", "Zen 3", "Zen 4", "Zen 5",
    "RDNA 2", "RDNA 3", "RDNA 4",
]

# 不該抓的：不是型號（架構、代號、腳位、晶片組、比較表、週邊產品）。
BAD_TITLE = re.compile(
    r"(microarchitecture|codename|slot|socket|chipset|comparison|instruction"
    r"|feature|branding|naming|architecture|manufacturing|\bssds?\b|\bmodem"
    r"|3d vision|\bgames?\b|keynote|satellite|intellivision|devices using"
    r"|list of intel cpus$|list of intel processors$|list of amd processors$"
    r"|list of nvidia gpus$|list of amd gpus$|list of radeon gpus$)", re.I)

# 最後仍要含硬體關鍵字才收（重定向後的標題也要過這關）。用詞界避免
# 「Intelligence」「Intellivision」被當成 Intel。
TITLE_OK = re.compile(
    r"(\bintel\b|\bamd\b|\bnvidia\b|\bradeon\b|geforce|qualcomm|snapdragon"
    r"|apple silicon|\bxeon\b|\bepyc\b|\bryzen\b|\bathlon\b|\bopteron\b"
    r"|\bphenom\b|\bsempron\b|\bturion\b|\bduron\b|\batom\b|\bceleron\b"
    r"|\bpentium\b|quadro|\btesla\b|\brtx\b|\bgtx\b|\barc\b|firepro"
    r"|processor|microprocessor|graphics processing units|\bgpu\b|\bcpu\b"
    r"|\blake\b|\bzen\b|threadripper)", re.I)


def enumerate_titles():
    titles = []
    for prefix in PREFIXES:
        try:
            d = api({"action": "query", "list": "allpages", "apprefix": prefix,
                     "aplimit": "500", "apnamespace": "0"})
        except RuntimeError as e:
            print(f"  {prefix:20s} FAIL — {e}")
            continue
        got = [p["title"] for p in d["query"]["allpages"]]
        print(f"  {prefix:20s} {len(got):4d}")
        titles.extend(got)
        time.sleep(1)
    titles.extend(EXPLICIT)
    # 去重、去掉不是型號清單的
    seen, out = set(), []
    for t in titles:
        k = t.lower()
        if k in seen or BAD_TITLE.search(t):
            continue
        seen.add(k)
        out.append(t)
    return sorted(out)


def resolve_redirects(titles, batch=50):
    """一次 50 個標題問「最後停在哪一頁」，把重複的清單頁收斂成一份。"""
    final, alias = [], {}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        d = api({"action": "query", "titles": "|".join(chunk), "redirects": "1"})
        q = d.get("query", {})
        norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
        redir = {n["from"]: n["to"] for n in q.get("redirects", [])}
        for t in chunk:
            p = norm.get(t, t)
            p = redir.get(p, p)
            alias.setdefault(p, []).append(t)
            final.append(p)
        time.sleep(1)
    return sorted(set(final)), alias


def keep(title):
    """型號清單頁才留；架構／腳位／比較表與重定向後的無關頁都丟。"""
    low = title.lower()
    # 只認「清單頁」這種形狀；EXPLICIT 是我們點名要的世代／家族頁。
    if not (low.startswith("list of") or low.startswith("table of")
            or low.startswith("ryzen") or low.startswith("epyc")
            or low.startswith("threadripper") or low.endswith(" lake")
            or low.endswith(" series") or "(microprocessor)" in low
            or re.match(r"^(?:zen|rdna)\s*\d$", low)
            or low in {t.lower() for t in EXPLICIT}):
        return False
    return bool(TITLE_OK.search(title)) and not BAD_TITLE.search(title)


def fetch_one(title, retries=4):
    err = None
    for attempt in range(retries):
        try:
            d = api({"action": "parse", "page": title, "prop": "wikitext"},
                    retries=2)
            if "error" in d:
                return None, d["error"].get("code", "api-error")
            text = d["parse"]["wikitext"]
            # parse 不會自動跟進重定向；遇到就再追一層（最多 3 層，防迴圈）。
            hops = 0
            while text.lstrip().lower().startswith("#redirect") and hops < 3:
                hops += 1
                mlink = re.search(r"\[\[([^\]|#]+)", text)
                if not mlink:
                    break
                d2 = api({"action": "parse", "page": mlink.group(1).strip(),
                          "prop": "wikitext"}, retries=2)
                if "error" in d2:
                    return None, d2["error"].get("code", "api-error")
                d, text = d2, d2["parse"]["wikitext"]
            return text, None
        except Exception as e:                      # noqa: BLE001 —— 網路錯誤種類太多
            err = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(4 + attempt * 6)
    return None, err


def fetch_rendered(title, retries=3):
    """渲染後的 HTML：模板轉嵌的表格（{{AMD Radeon RX 6000 series}}）只在這看得到。"""
    err = None
    for attempt in range(retries):
        try:
            d = api({"action": "parse", "page": title, "prop": "text",
                     "disableeditsection": "1"}, retries=2)
            if "error" in d:
                return None, d["error"].get("code", "api-error")
            return d["parse"]["text"], None
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            time.sleep(4 + attempt * 6)
    return None, err


def safe_name(title):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", title) + ".wiki"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("列出清單頁……")
    titles = [t for t in enumerate_titles() if keep(t)]
    final, alias = resolve_redirects(titles)
    final = [t for t in final if keep(t)]
    print(f"\n{len(titles)} 個標題 → 去除重定向／無關頁後 {len(final)} 個頁面")
    if args.dry_run:
        for t in final:
            print("   ", t)
        return 0

    ok = skip = fail = html_n = 0
    for i, title in enumerate(final, 1):
        path = os.path.join(args.out, safe_name(title))
        if os.path.exists(path) and os.path.getsize(path) > 2000:
            text = None
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read(400000)
            skip += 1
        else:
            text, err = fetch_one(title, args.retries)
            if not text:
                fail += 1
                print(f"[{i}/{len(final)}] FAIL {title} — {err}")
                time.sleep(args.delay * 2)
                continue
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            ok += 1
            print(f"[{i}/{len(final)}] ok {len(text)/1000:7.0f} KB  {title}")
            time.sleep(args.delay)

        # 頁面本身幾乎沒有表格 → 型號表很可能是模板轉嵌的，再抓一份渲染後的 HTML。
        hpath = os.path.splitext(path)[0] + ".html"
        if (text is not None and text.count("{|") < 3
                and not (os.path.exists(hpath) and os.path.getsize(hpath) > 2000)):
            html_text, herr = fetch_rendered(title)
            if html_text:
                with open(hpath, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(html_text)
                html_n += 1
                print(f"         +html {len(html_text)/1000:6.0f} KB")
            else:
                print(f"         +html FAIL {title} — {herr}")
            time.sleep(args.delay)

    print(f"\n完成：新抓 {ok}、略過 {skip}、HTML {html_n}、失敗 {fail} → {args.out}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
