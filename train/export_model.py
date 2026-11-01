#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把訓練出來的 LoRA adapter 併回底座，存成「可以直接轉 GGUF」的 fp16 目錄。

為什麼要合併而不是出貨 adapter：
  · Ollama / LM Studio / llama.cpp 對「adapter」的支援各不相同，對「一個 gguf」則一致。
  · 交付物要能被不懂 PEFT 的人雙擊就跑，所以先合併再量化。
adapter 本身仍然留著（train/out/lora），要換底座重練時還有得用。

用法：
  python export_model.py --base ../models/Qwen3-1.7B --adapter out/lora --out merged
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(HERE, os.pardir, "models", "Qwen3-1.7B"))
    ap.add_argument("--adapter", default=os.path.join(HERE, "out", "lora"))
    ap.add_argument("--out", default=os.path.join(HERE, "merged"))
    args = ap.parse_args()

    for p, what in ((args.base, "底座"), (args.adapter, "adapter")):
        if not os.path.isdir(p):
            print(f"找不到{what}：{p}")
            return 1

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"底座：{args.base}")
    print(f"adapter：{args.adapter}")
    # 合併不需要 GPU：CPU 上做是小時的差別，而 12 GB 的卡留給訓練。
    print("載入底座（fp16, CPU）…")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True)
    print("套上 adapter…")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("合併權重…")
    merged = model.merge_and_unload()
    merged.eval()

    os.makedirs(args.out, exist_ok=True)
    print(f"寫出 {args.out} …")
    merged.save_pretrained(args.out, safe_serialization=True, max_shard_size="2GB")

    tok = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tok.save_pretrained(args.out)

    # 記下來源，日後要追「這個 gguf 是哪一版資料、哪一個底座」不必翻對話紀錄。
    meta = {
        "base": os.path.abspath(args.base),
        "adapter": os.path.abspath(args.adapter),
        "merged": os.path.abspath(args.out),
        "dtype": "float16",
    }
    with open(os.path.join(args.out, "MERGE-INFO.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    total = sum(os.path.getsize(os.path.join(args.out, n))
                for n in os.listdir(args.out) if n.endswith(".safetensors"))
    print(f"完成：{total / 1e9:.2f} GB safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
