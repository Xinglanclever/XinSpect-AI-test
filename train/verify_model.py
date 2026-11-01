#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""訓練完之後，用三種語言各問一題，看它有沒有學會「曦覽的講法」。

這不是評測——三題看不出好壞，只看得出「有沒有壞掉」：
  · 還在不在講人話（不是 '!!!!'、不是空的）
  · 有沒有用曦覽的詞（事實鍵、六態、只讀、出處……）
  · 被問到「我的電腦」時會不會說「網站不掃描任何電腦」

用法：
  python verify_model.py --model merged
  python verify_model.py --model merged --adapter out/lora     # 未合併時直接掛 adapter
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

PROBES = [
    ("zh-Hant", "曦覽是什麼？它會不會上傳我的資料？"),
    ("zh-Hans", "曦览和一般的硬件检测软件有什么不同？"),
    ("en", "Does XinSpect upload my data? What does it read?"),
    ("zh-Hant", "我這台電腦現在 CPU 幾度？"),
]

SYSTEM_BY_LANG = {
    "zh-Hant": (
        "你是「XinSpect-Ai」，曦覽（XinSpect）官網的說明助手。曦覽是 Windows 上的原生硬體檢視、"
        "監控與安全稽核工具。只根據文件回答曦覽相關問題；文件沒寫的就說沒寫。"
        "你不能讀使用者的電腦，任何「我這台機器現在…」的問題要說明網站不掃描任何電腦。"
        "用繁體中文回答。"
    ),
    "zh-Hans": (
        "你是「XinSpect-Ai」，曦览（XinSpect）官网的说明助手。曦览是 Windows 上的原生硬件检视、"
        "监控与安全稽核工具。只根据文档回答曦览相关问题；文档没写的就说没写。"
        "你不能读用户的电脑，任何「我这台机器现在…」的问题要说明网站不扫描任何电脑。"
        "用简体中文回答。"
    ),
    "en": (
        "You are \"XinSpect-Ai\", the assistant on the official XinSpect website. XinSpect is a native "
        "Windows hardware inspection, monitoring and security-audit tool. Answer only from the documentation; "
        "if it does not cover something, say so. You cannot read the user's computer: for any "
        "\"what is my machine doing right now\" question, say the website never scans a computer. "
        "Answer in English."
    ),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "merged"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--cpu", action="store_true", help="用 CPU 跑（沒有 GPU 或 GPU 被佔用時）")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() and not args.cpu else torch.float32
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"載入 {args.model}（{device}, {str(dtype).split('.')[-1]}）…")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)

    if args.adapter:
        from peft import PeftModel
        print(f"掛上 adapter {args.adapter} …")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    for lang, q in PROBES:
        msgs = [{"role": "system", "content": SYSTEM_BY_LANG[lang]},
                {"role": "user", "content": q}]
        try:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print("\n" + "=" * 70)
        print(f"[{lang}] {q}")
        print("-" * 70)
        print(text if text else "（空回答）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())