#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 XinSpect 倉庫的說明文件切成「帶出處的區塊」，供檢索與引用。

為什麼要這支：
  曦覽的文件本身是這個 AI 唯一可信的知識來源。要能回答「這把鍵從哪裡讀、憑什麼算數」，
  就必須讓每一個回答片段都帶著「出自哪個檔案的第幾行」，否則 AI 只是在講聽起來合理的話。
  所以這裡不做摘要、不改寫、不刪字——只切塊，並把行號記下來。

輸出一律 UTF-8 JSONL（每行一個區塊）與一份 manifest，供 rag/ 下使用。
"""

import argparse
import json
import os
import re
import sys

# ── 語料來源：倉庫內的路徑 → (語言, 標題) ────────────────────────────────
# 只收「說明性文件」。程式碼另由 fact-key 與 changelog 兩支抽取器處理成結構化區塊。
DOCS = [
    ("README.md", "zh-Hant", "曦覽 XinSpect 百科"),
    ("README.zh-CN.md", "zh-Hans", "曦览 XinSpect 百科"),
    ("README.en.md", "en", "XinSpect Encyclopedia"),
    ("手冊.md", "zh-Hant", "曦覽 XinSpect 手冊"),
    ("WHITEPAPER.md", "zh-Hant", "曦覽 XinSpect 技術白皮書"),
    ("WHITEPAPER.zh-CN.md", "zh-Hans", "曦览 XinSpect 技术白皮书"),
    ("WHITEPAPER.en.md", "en", "XinSpect Technical Whitepaper"),
    ("CONTRIBUTING.md", "zh-Hant", "貢獻指南"),
    ("SECURITY.md", "zh-Hant", "安全性政策"),
    ("SUPPORT.md", "zh-Hant", "支援"),
    ("CODE_OF_CONDUCT.md", "zh-Hant", "行為準則"),
    ("docs/DEPLOYMENT.md", "zh-Hant", "部署"),
    ("docs/DATA-SOVEREIGNTY.md", "zh-Hant", "資料主權"),
    ("docs/FEATURE-REFERENCE.md", "zh-Hant", "功能參考"),
    ("docs/MEASUREMENT-METHODOLOGY.md", "zh-Hant", "量測方法論"),
    ("docs/spec/METHODOLOGY.md", "zh-Hant", "量測方法論（規格）"),
    ("docs/spec/LIMITATIONS.md", "zh-Hant", "已知限制（規格）"),
    ("docs/spec/CERTIFICATION-PLAN.md", "zh-Hant", "認證計畫"),
    ("docs/spec/query-language.v1.md", "zh-Hant", "事實查詢語言 v1"),
    ("docs/spec/corpus-upload.v1.md", "zh-Hant", "語料貢獻包格式 v1"),
    ("docs/RELEASE-NOTES-3.05.md", "zh-Hant", "3.05 發佈說明"),
    ("docs/RELEASE-NOTES-3.04.md", "zh-Hant", "3.04 發佈說明"),
    ("docs/RELEASE-NOTES-3.03.md", "zh-Hant", "3.03 發佈說明"),
]

REPO = "https://github.com/Xinglanclever/XinSpect/blob/main/"

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
LINK_ONLY = re.compile(r"\[[^\]]*\]\([^)]*\)")


def slugify(text, seen):
    """與 GitHub 相同的錨點規則：小寫、去標點、空格轉連字號、重複加序號。"""
    s = text.strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = s.strip("-")
    if not s:
        s = "section"
    n = seen.get(s, 0)
    seen[s] = n + 1
    return s if n == 0 else f"{s}-{n}"


def is_noise(block):
    """目錄、徽章列、純連結列不是知識，收了只會稀釋檢索。"""
    body = block.strip()
    if not body:
        return True
    if len(body) < 24:
        return True
    # 目錄：幾乎整段都是 markdown 內部連結
    stripped = LINK_ONLY.sub("", body)
    if len(stripped) < len(body) * 0.35:
        return True
    # 徽章列：連續的 shields.io 圖片
    if body.count("img.shields.io") >= 2:
        return True
    return False


def split_markdown(path, text, lang, title):
    """逐行掃描，維護「標題路徑」，把正文累積成區塊。回傳 list[dict]。"""
    lines = text.splitlines()
    chunks = []
    stack = []          # [(level, text, anchor)]
    seen_anchors = {}
    buf = []
    buf_start = 0
    in_fence = False

    def flush(end_line):
        nonlocal buf, buf_start
        if not buf:
            return
        body = "\n".join(buf)
        start = buf_start
        buf = []
        if is_noise(body):
            return
        heading_path = " › ".join(t for _, t, _ in stack) or title
        anchor = stack[-1][2] if stack else ""
        chunks.append({
            "source": path,
            "lang": lang,
            "doc_title": title,
            "heading_path": heading_path,
            "anchor": anchor,
            "line_start": start,
            "line_end": end_line,
            "text": body.strip(),
            "url": f"{REPO}{path}#L{start}-L{end_line}",
        })

    for idx, raw in enumerate(lines, start=1):
        if FENCE.match(raw):
            in_fence = not in_fence
            buf.append(raw)
            continue

        if not in_fence:
            m = HEADING.match(raw)
            if m:
                flush(idx - 1)
                level = len(m.group(1))
                htext = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, htext, slugify(htext, seen_anchors)))
                continue

        if not raw.strip():
            if buf:
                flush(idx - 1)
            buf_start = idx + 1
            continue

        if not buf:
            buf_start = idx
        buf.append(raw)

    flush(len(lines))
    return chunks


def chunk_long(chunk, max_chars=1400, overlap=180):
    """把過長的區塊在段落邊界切開，重疊少許上下文讓跨段落的說明不會斷頭。"""
    text = chunk["text"]
    if len(text) <= max_chars:
        return [chunk]

    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    cur = []
    cur_len = 0

    def emit():
        nonlocal cur, cur_len
        if not cur:
            return
        body = "\n\n".join(cur)
        if not is_noise(body):
            piece = dict(chunk)
            piece["text"] = body
            piece["part"] = len(out) + 1
            out.append(piece)
        # 保留尾巴一點重疊
        tail = []
        tlen = 0
        for p in reversed(cur):
            if tlen + len(p) > overlap:
                break
            tail.insert(0, p)
            tlen += len(p)
        cur = tail
        cur_len = tlen

    for p in paras:
        if len(p) > max_chars:
            # 單一段落就超長（通常是表格）：硬切，對齊換行
            for i in range(0, len(p), max_chars):
                cur.append(p[i:i + max_chars])
                cur_len += len(p[i:i + max_chars])
                emit()
            continue
        if cur_len + len(p) > max_chars:
            emit()
        cur.append(p)
        cur_len += len(p)
    if cur and "\n\n".join(cur) != "\n\n".join(paras[-len(cur):]) or (cur and not out):
        emit()
    return out or [chunk]


# ── 事實鍵目錄：把 1,166 把鍵變成可檢索的結構化區塊 ──────────────────────
def build_fact_keys(repo):
    path = os.path.join(repo, "Services", "FactKeyCatalog.cs")
    if not os.path.exists(path):
        return [], {"error": "找不到 FactKeyCatalog.cs"}
    text = open(path, encoding="utf-8-sig").read()
    keys = re.findall(r'^\s*"([a-z0-9][a-z0-9._-]*)"\s*,?\s*$', text, re.M)
    keys = sorted(set(keys))

    # 依第一段前綴分組；每組一塊，方便「以 ac.* 開頭的鍵有哪些」這類問題
    groups = {}
    for k in keys:
        prefix = k.split(".")[0]
        groups.setdefault(prefix, []).append(k)

    chunks = []
    for prefix in sorted(groups):
        members = groups[prefix]
        body = (
            f"事實鍵前綴 `{prefix}.` 共 {len(members)} 把：\n"
            + "\n".join(f"- `{k}`" for k in members)
        )
        chunks.append({
            "source": "Services/FactKeyCatalog.cs",
            "lang": "zh-Hant",
            "doc_title": "事實鍵目錄",
            "heading_path": f"事實鍵目錄 › 前綴 {prefix}",
            "anchor": "factkey-" + prefix,
            "line_start": 0,
            "line_end": 0,
            "text": body,
            "url": f"{REPO}Services/FactKeyCatalog.cs",
        })

    meta = {
        "count": len(keys),
        "prefixes": {p: len(v) for p, v in sorted(groups.items())},
        "keys": keys,
    }
    return chunks, meta


# ── 版本沿革：從 ChangelogCatalog.cs 抽出每一筆 ───────────────────────────
def build_changelog(repo):
    path = os.path.join(repo, "Nav", "ChangelogCatalog.cs")
    if not os.path.exists(path):
        return [], {"error": "找不到 ChangelogCatalog.cs"}
    text = open(path, encoding="utf-8-sig").read()

    entries = []
    for m in re.finditer(r"new ChangeEntry\s*\{(.*?)\n\s*\},", text, re.S):
        block = m.group(1)
        ver = re.search(r'Version\s*=\s*"([^"]*)"', block)
        date = re.search(r'Date\s*=\s*"([^"]*)"', block)
        title = re.search(r'Title\s*=\s*"([^"]*)"', block)
        items = re.findall(r'^\s*"((?:[^"\\]|\\.)*)",?\s*$', block, re.M)
        entries.append({
            "version": ver.group(1) if ver else "",
            "date": date.group(1) if date else "",
            "title": title.group(1) if title else "",
            "items": [i.replace('\\"', '"') for i in items if i.strip()],
        })

    chunks = []
    for e in entries:
        if not e["version"]:
            continue
        body = [f"版本 {e['version']}（{e['date'] or '日期未留下紀錄'}）", "", e["title"]]
        body += ["", "逐項改動："] + [f"- {i}" for i in e["items"]]
        chunks.append({
            "source": "Nav/ChangelogCatalog.cs",
            "lang": "zh-Hant",
            "doc_title": "版本沿革",
            "heading_path": f"版本沿革 › {e['version']}",
            "anchor": f"v{e['version']}",
            "line_start": 0,
            "line_end": 0,
            "text": "\n".join(body),
            "url": f"{REPO}Nav/ChangelogCatalog.cs",
        })
    return chunks, {"count": len(chunks), "latest": entries[0] if entries else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=r"C:\Users\Administrator\XinSpect")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    args = ap.parse_args()

    chunks = []
    missing = []

    for rel, lang, title in DOCS:
        path = os.path.join(args.repo, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        text = open(path, encoding="utf-8-sig").read()
        for c in split_markdown(rel, text, lang, title):
            chunks.extend(chunk_long(c))

    fact_chunks, fact_meta = build_fact_keys(args.repo)
    chunks.extend(fact_chunks)

    log_chunks, log_meta = build_changelog(args.repo)
    chunks.extend(log_chunks)

    # 穩定 id：來源＋行號＋序號，重跑同輸入得到同 id
    for i, c in enumerate(chunks):
        c["id"] = f'{c["source"].replace("/", "_")}#{c.get("part", 1)}-{i:05d}'

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_lang = {}
    for c in chunks:
        by_lang[c["lang"]] = by_lang.get(c["lang"], 0) + 1
    chars = sum(len(c["text"]) for c in chunks)

    manifest = {
        "chunks": len(chunks),
        "chars": chars,
        "by_lang": by_lang,
        "sources": sorted({c["source"] for c in chunks}),
        "missing": missing,
        "fact_keys": fact_meta,
        "changelog": log_meta,
    }
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"chunks={len(chunks)} chars={chars} by_lang={by_lang}")
    print(f"missing={missing}")
    print(f"fact_keys={fact_meta.get('count')} changelog={log_meta.get('count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
