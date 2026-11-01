#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兩個語料庫分別量：曦覽自有文件、外部硬體語料。

為什麼要分開量：這兩批資料的「該拒答」界線不一樣。
曦覽文件的問題範圍很窄（產品本身），所以 0.45 就足以切開；外部語料是通用硬體知識，
「什麼是 PCIe 6.0」與「怎麼烤麵包」之間的距離比想像中近——兩者都可能命中一兩個詞。
分開量才知道門檻該設在哪，而不是拿一個數字硬套兩批資料。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rag"))

from bm25 import BM25Index  # noqa: E402


def measure(idx, query):
    hits = idx.search(query, top_k=5, prefer_lang=None)
    if not hits:
        return {"ans": 0.0, "ans_best3": 0.0, "bm25": 0.0}
    return {
        "ans": idx.answerability(query, hits[0]),
        "ans_best3": max(idx.answerability(query, h) for h in hits[:3]),
        "bm25": hits[0]["score"],
    }


def run(idx, qfile, gate, label="", verbose=False):
    qs = json.load(open(qfile, encoding="utf-8"))
    tp = fn = tn = fp = 0
    misses = []
    for item in qs["in_domain"]:
        sig = measure(idx, item["q"])
        if gate(sig):
            tp += 1
        else:
            fn += 1
            misses.append(("MISS", item["q"], sig))
    for item in qs["out_domain"]:
        sig = measure(idx, item["q"])
        if gate(sig):
            fp += 1
            misses.append(("FALSE-ANSWER", item["q"], sig))
        else:
            tn += 1
    r = tp / max(tp + fn, 1)
    rj = tn / max(tn + fp, 1)
    p = tp / max(tp + fp, 1)
    print(f"{label:28s} recall={r:.2f} reject={rj:.2f} precision={p:.2f}"
          f"  (tp={tp} fn={fn} tn={tn} fp={fp})")
    if verbose:
        for kind, q, sig in misses:
            print(f"    {kind:13s} {q[:60]}   ans={sig['ans']:.2f} best3={sig['ans_best3']:.2f}")
    return r, rj, p


def sweep(idx, qfile, name):
    print(f"\n===== {name} =====")
    best = None
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        r, rj, p = run(idx, qfile, lambda s, t=t: s["ans_best3"] >= t, f"best3>={t}")
        # 目標函數刻意壓在拒絕率上：答錯的代價遠大於答不出來
        score = rj * 2 + r + p
        if best is None or score > best[0]:
            best = (score, t, r, rj, p)
    print("對照（只看 BM25）：")
    for t in [15, 20, 25, 30]:
        run(idx, qfile, lambda s, t=t: s["bm25"] >= t, f"bm25>={t}")
    _, t, *_ = best
    print(f"\n選定 best3>={t} 的失誤：")
    run(idx, qfile, lambda s: s["ans_best3"] >= t, "FINAL", verbose=True)
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    product = BM25Index.load(os.path.join(HERE, "..", "rag", "index", "product"))
    external = BM25Index.load(os.path.join(HERE, "..", "rag", "index", "external"))

    t_prod = sweep(product, os.path.join(HERE, "questions.json"), "曦覽自有文件")
    t_ext = sweep(external, os.path.join(HERE, "questions-external.json"), "外部硬體語料")

    print(f"\n建議門檻：product={t_prod}  external={t_ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
