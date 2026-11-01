#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把「曦覽文件問答」與「硬體型號問答」併成一份訓練集。

為什麼要分開產生、再合併：
  文件題的答案形狀是「短答＋要點＋〔檔名:行號〕」，硬體題是「值＋〔來源：頁名〕」；
  兩邊的模板差很多，塞進同一支腳本會互相污染（硬體題會學到「要點」格式，
  文件題會學到「一行帶過」）。分開寫、合併時再打散，模型才會學到
  「看問題決定形狀」。

合併時做的事：
  1. 去重（同一個問題與答案只留一筆；三語各自獨立，不算重複）
  2. 文件資料照舊的 train／valid 切分（那份切分已經用過，換掉會讓前後評測不可比）
  3. 硬體資料的 8% 進 valid——只用文件 valid 評估，會看不到硬體題退步
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def key_of(rec):
    msgs = rec.get("messages") or []
    users = [m.get("content", "") for m in msgs if m.get("role") == "user"]
    asst = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
    return (users[0] if users else "", asst[0] if asst else "")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-train", default=os.path.join(ROOT, "train", "sft-train.jsonl"))
    ap.add_argument("--doc-valid", default=os.path.join(ROOT, "train", "sft-valid.jsonl"))
    ap.add_argument("--hw", default=os.path.join(ROOT, "train", "sft-hw.jsonl"))
    ap.add_argument("--out-train", default=os.path.join(ROOT, "train", "sft-train-v2.jsonl"))
    ap.add_argument("--out-valid", default=os.path.join(ROOT, "train", "sft-valid-v2.jsonl"))
    ap.add_argument("--hw-valid", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()

    random.seed(args.seed)
    doc_train, doc_valid, hw = load(args.doc_train), load(args.doc_valid), load(args.hw)
    print(f"文件 train {len(doc_train)}／valid {len(doc_valid)}；硬體 {len(hw)}")

    seen = set()
    def keep(rec):
        k = key_of(rec)
        if not k[0] or k in seen:
            return False
        seen.add(k)
        return True

    train = [r for r in doc_train if keep(r)]
    valid = [r for r in doc_valid if keep(r)]
    hw = [r for r in hw if key_of(r) not in seen and keep(r)]

    random.shuffle(hw)
    n_val = max(1, int(len(hw) * args.hw_valid))
    valid += hw[:n_val]
    train += hw[n_val:]
    random.shuffle(train)
    random.shuffle(valid)

    for path, rows in ((args.out_train, train), (args.out_valid, valid)):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for rec in rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"合併後：train {len(train)}、valid {len(valid)}（硬體進 valid {n_val}）")
    print(f"  → {args.out_train}\n  → {args.out_valid}")


if __name__ == "__main__":
    raise SystemExit(main())
