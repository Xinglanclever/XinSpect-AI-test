#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量測檢索層：域內召回率與域外拒答率。

兩件事分開量，因為它們的失敗方式完全不同：
  · 域內沒召回（false negative）→ 使用者問得到的事被說成「文件沒寫」，體驗差但不算說謊。
  · 域外卻回答（false positive）→ 模型用文件拼出一個聽起來對的答案，那是這個專案最不能犯的錯。

所以下面的權重刻意壓在「拒絕率」上：寧可多說幾次「沒寫」，也不要編。
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rag"))

from bm25 import BM25Index  # noqa: E402

INDEX = os.path.join(HERE, "..", "rag", "index", "xinspect")
QUESTIONS = os.path.join(HERE, "questions.json")


def measure(idx, query):
    hits = idx.search(query, top_k=5, prefer_lang=None)
    if not hits:
        return {"ans": 0.0, "ans_best3": 0.0, "bm25": 0.0}, hits
    ans = idx.answerability(query, hits[0])
    ans3 = max(idx.answerability(query, h) for h in hits[:3])
    return {"ans": ans, "ans_best3": ans3, "bm25": hits[0]["score"]}, hits


def run(idx, gate, label="", verbose=False):
    qs = json.load(open(QUESTIONS, encoding="utf-8"))
    tp = fn = tn = fp = 0
    misses = []
    for item in qs["in_domain"]:
        sig, _ = measure(idx, item["q"])
        if gate(sig):
            tp += 1
        else:
            fn += 1
            misses.append(("MISS", item["q"], sig))
    for item in qs["out_domain"]:
        sig, _ = measure(idx, item["q"])
        if gate(sig):
            fp += 1
            misses.append(("FALSE-ANSWER", item["q"], sig))
        else:
            tn += 1
    recall = tp / max(tp + fn, 1)
    reject = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    print(f"{label:30s} recall={recall:.2f} reject={reject:.2f} precision={precision:.2f}"
          f"  (tp={tp} fn={fn} tn={tn} fp={fp})")
    if verbose:
        for kind, q, sig in misses:
            print(f"    {kind:13s} {q}   ans={sig['ans']:.2f} best3={sig['ans_best3']:.2f} bm25={sig['bm25']:.1f}")
    return recall, reject, precision


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = BM25Index.load(INDEX)

    print("── 只看 answerability（內容字連續覆蓋率）──")
    rows = []
    for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        r, rj, p = run(idx, lambda s, t=t: s["ans_best3"] >= t, f"best3>={t}")
        rows.append((rj * 2 + r + p, t, r, rj, p))

    print("\n── 對照：只看 BM25 分數 ──")
    for t in [12, 16, 20, 25]:
        run(idx, lambda s, t=t: s["bm25"] >= t, f"bm25>={t}")

    print("\n── 只取第一名 vs 前三名最大值 ──")
    for t in [0.4, 0.5]:
        run(idx, lambda s, t=t: s["ans"] >= t, f"top1>={t}")

    rows.sort(reverse=True)
    _, best_t, *_ = rows[0]
    print(f"\n── 選定 best3>={best_t} 的失誤明細 ──")
    run(idx, lambda s: s["ans_best3"] >= best_t, "FINAL", verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
