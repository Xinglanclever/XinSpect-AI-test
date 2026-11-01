#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端評測：對「已經跑起來的服務」問完整題庫，看它答不答、拒得對不對、出處對不對得上。

跟 eval_split.py 的差別：那一支量的是檢索（BM25 分數與門檻），這一支量的是**成品**——
檢索、閘門、提示、生成全部串起來之後的表現。生成模型換了就要重跑一次。

用法：
  python run_e2e.py --endpoint http://127.0.0.1:8793/v1/xinspect/ask
  python run_e2e.py --endpoint http://127.0.0.1:8793/v1/xinspect/ask --limit 10
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# 答案裡出現的定位字串，例如 README.md:10508 或 Wikipedia/PCI Express 第 23 段
RE_LOCATOR = re.compile(r"([\w./\-]+\.(?:md|cs|json|txt|csproj|ps1|yml))[:：]L?(\d+)")
RE_WIKI = re.compile(r"(Wikipedia/[^\s，。；）)]+?)\s*第\s*\d+\s*段")
RE_SPEC = re.compile(r"(Wikipedia/[^\s，。；）)]+?)\s*規格表\s*\d+")


def ask(endpoint, question, timeout=600):
    body = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data, time.time() - t
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}"}, time.time() - t
    except Exception as e:                     # 連線中斷、逾時
        return {"error": type(e).__name__ + ": " + str(e)}, time.time() - t


def locators_in(text):
    return {(m.group(1), m.group(2)) for m in RE_LOCATOR.finditer(text)}


def cite_keys(citations):
    """把 citations 轉成可比對的鍵：('README.md', '10508') 這種。"""
    keys = set()
    for c in citations or []:
        loc = c.get("locator") or ""
        m = RE_LOCATOR.search(loc)
        if m:
            keys.add((m.group(1), m.group(2)))
        elif "Wikipedia/" in loc:
            keys.add(("wiki", loc.strip()))
    return keys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8793/v1/xinspect/ask")
    ap.add_argument("--limit", type=int, default=0, help="每一類只跑前 N 題（除錯用）")
    ap.add_argument("--out", default=os.path.join(HERE, "e2e-results.json"))
    args = ap.parse_args()

    files = [os.path.join(HERE, "questions.json"),
             os.path.join(HERE, "questions-external.json"),
             os.path.join(HERE, "questions-hw.json")]
    items = []
    for path in files:
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        for it in d.get("in_domain", []):
            items.append(("in", it))
        for it in d.get("out_domain", []):
            items.append(("out", it))
    if args.limit:
        keep_in = [x for x in items if x[0] == "in"][: args.limit]
        keep_out = [x for x in items if x[0] == "out"][: args.limit]
        items = keep_in + keep_out

    print(f"題目 {len(items)} 題 → {args.endpoint}\n")
    results = []
    for i, (kind, item) in enumerate(items, 1):
        q = item["q"]
        data, dt = ask(args.endpoint, q)
        if "error" in data:
            print(f"[{i:3d}] ERROR {q}  → {data['error']}")
            results.append({"kind": kind, "q": q, "error": data["error"], "seconds": dt})
            continue

        answer = (data.get("answer") or "").strip()
        # 硬體題看的是「答案裡有沒有那個值」：型號查不到時小模型最常見的失敗
        # 是用同系列的另一顆頂替（問 9800X3D 答 7800X3D），這種答案讀起來很順。
        expect = item.get("expect_all") or []
        flat = re.sub(r"\s+", "", answer)
        # 兩邊都要去掉空白再比：「Zen 4」 vs 答案裡的「Zen4Yes」
        value_ok = (all(re.sub(r"\s+", "", e) in flat for e in expect)
                    if expect else None)
        cites = data.get("citations") or []
        grounded = bool(data.get("grounded"))
        answer_locs = locators_in(answer)
        # 答案裡自己寫的定位字串，有多少比例真的出現在 citations 裡
        valid = 0
        if answer_locs:
            keys = cite_keys(cites)
            valid = sum(1 for a in answer_locs
                        if any(a[0].split("/")[-1] == k[0].split("/")[-1] and a[1] == k[1]
                               for k in keys))
        results.append({
            "kind": kind, "q": q, "lang": item.get("lang"),
            "expect_all": expect, "value_ok": value_ok,
            "grounded": grounded, "kind_tag": data.get("kind"),
            "answer": answer, "citations": [
                {"locator": c.get("locator"), "url": c.get("url")} for c in cites],
            "answer_locators": sorted("%s:%s" % a for a in answer_locs),
            "locators_valid": valid, "locators_total": len(answer_locs),
            "seconds": round(dt, 1),
        })
        mark = "ok " if (grounded if kind == "in" else not grounded) else "!! "
        print(f"[{i:3d}] {mark}{dt:5.1f}s  {q[:46]}")

    in_dom = [r for r in results if r["kind"] == "in" and "error" not in r]
    out_dom = [r for r in results if r["kind"] == "out" and "error" not in r]
    answered = [r for r in in_dom if r["grounded"] and r["answer"]
                and not str(r.get("kind_tag") or "").startswith("machine-specific")]
    # 「我的電腦現在幾度／我的配置是什麼」不是文件問題，走的是機器專屬那條固定回答，
    # 它在 grounded 上是 true（我們有把握說「網站讀不到」），但對評測來說算「正確地沒答」。
    refused = [r for r in out_dom
               if not r["grounded"] or str(r.get("kind_tag") or "").startswith("machine-specific")]
    checked = [r for r in in_dom if r.get("expect_all")]
    value_hit = sum(1 for r in checked if r.get("value_ok"))
    with_loc = [r for r in in_dom if r["locators_total"] > 0]
    valid_loc = sum(r["locators_valid"] for r in with_loc)
    total_loc = sum(r["locators_total"] for r in with_loc)
    lat = [r["seconds"] for r in results if "error" not in r]

    print("\n" + "=" * 62)
    print(f"應該回答的 {len(in_dom)} 題：答了 {len(answered)}（{len(answered)/max(len(in_dom),1):.0%}）")
    print(f"應該拒答的 {len(out_dom)} 題：拒了 {len(refused)}（{len(refused)/max(len(out_dom),1):.0%}）")
    if checked:
        print(f"帶期望值的 {len(checked)} 題：值寫對 {value_hit}（{value_hit/len(checked):.0%}）")
    if total_loc:
        print(f"出處對得上：{valid_loc}/{total_loc}（{valid_loc/total_loc:.0%}）"
              f"—— 來自 {len(with_loc)} 個有寫出處的回答")
    if lat:
        print(f"延遲：中位數 {statistics.median(lat):.1f}s，最慢 {max(lat):.1f}s")
    gaps = [r for r in in_dom if not r["grounded"]]
    if gaps:
        print("\n答不出來的（多半是門檻錯殺，值得看是不是問法差太多）：")
        for r in gaps[:8]:
            print("  ·", r["q"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"endpoint": args.endpoint, "results": results}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n逐題結果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())