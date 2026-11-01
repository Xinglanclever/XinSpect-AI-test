#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QLoRA 微調：在 12 GB 的 TITAN Xp 上訓練一個「曦覽語氣」的小模型。

先讀這段，再決定要不要跑：

  **這台機器可以把 QLoRA 跑起來，但不該期待它取代檢索。**
  TITAN Xp 是 Pascal（2017），沒有 bf16、沒有 tensor core，官方 PyTorch 的
  cu118 之後的版本對 sm_61 支援越來越差。實測結論：

    · 4B 模型 4-bit QLoRA、序列長度 1024、batch 1、梯度累積 8 —— 可行，
      但一步大約 4～8 秒，2,500 筆訓練資料跑 3 個 epoch 要十幾小時。
    · 7B 以上在這個 VRAM 上要開 offload，慢到沒有實用價值。
    · 就算訓完，它也不會比「BM25 檢索 + 大模型」更準——事實來自檢索，微調只改語氣。

  所以這支腳本的定位是：**讓你自己驗證這件事**，而不是交付一個「已訓練的模型」。
  真要上線，用 Cloudflare Workers AI 或租一張 24 GB 卡（A10/4090）跑同一個腳本。

用法（需要先裝依賴，見 requirements-train.txt）：
  python qlora.py --data sft-train.jsonl --valid sft-valid.jsonl --out ./out --epochs 3

沒有 GPU 也能跑（--device cpu），只是慢到只能驗證流程通不通。
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def check_env():
    """先檢查環境，缺什麼一次講清楚——不要讓使用者在訓練到一半才發現。

    回傳 (info, problems, warnings)：problems 是「跑不了」，warnings 是「能跑但要知道」。
    兩者混在一起會讓一句 sm_61 的提醒變成致命錯誤——那正是之前的 bug。
    """
    problems = []
    warnings = []
    try:
        import torch
        info = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
        if torch.cuda.is_available():
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = torch.cuda.get_device_capability(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            major, minor = info["capability"]
            if major < 8:
                warnings.append(
                    f"GPU 是 sm_{major}{minor}（Pascal/Turing 之前）：不支援 bf16，"
                    "bitsandbytes 的 4-bit 也可能退回慢路徑。可以用，但別期待速度。")
    except ImportError:
        info = {}
        problems.append("沒有 torch：pip install torch --index-url https://download.pytorch.org/whl/cu118")

    for mod, hint in (
        ("transformers", "pip install transformers"),
        ("peft", "pip install peft"),
        ("datasets", "pip install datasets"),
        ("accelerate", "pip install accelerate"),
    ):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"沒有 {mod}：{hint}")

    # bitsandbytes 只有「4-bit」那條路需要。Windows 上它的 wheel 常編不起來，
    # 所以沒有它不算環境壞掉——退回 fp16 LoRA，1.7B 在 12 GB 上綽綽有餘。
    try:
        __import__("bitsandbytes")
        info["bnb"] = True
    except ImportError:
        info["bnb"] = False

    return info, problems, warnings


def load_pairs(path, tokenizer, max_len):
    """把 {"messages":[...]} 轉成 prompt/completion 兩段式的訓練樣本。

    為什麼不用 chat template 直接整段算 loss：整段算 loss 會讓模型把
    「system 提示 + 使用者問題」也當成要學的東西。這裡只對 assistant 那一段算 loss，
    才是「學會怎麼回答」而不是「學會怎麼背題目」。
    """
    from datasets import Dataset

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            msgs = rec.get("messages") or []
            if len(msgs) < 3:
                continue
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            answer = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if not user or not answer:
                continue

            prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            completion = answer + "<|im_end|>"
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

            # 太長就砍使用者問題那一段（它是最不重要的），不要砍答案
            if len(p_ids) + len(c_ids) > max_len:
                keep = max(64, max_len - len(c_ids) - 8)
                p_ids = p_ids[-keep:]
            ids = (p_ids + c_ids)[:max_len]
            labels = ([-100] * len(p_ids) + c_ids)[:max_len]

            rows.append({
                "input_ids": ids,
                "labels": labels,
                "attention_mask": [1] * len(ids),
            })
    return Dataset.from_list(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "sft-train.jsonl"))
    ap.add_argument("--valid", default=os.path.join(HERE, "sft-valid.jsonl"))
    ap.add_argument("--base", default=os.path.join(HERE, os.pardir, "models", "Qwen3-1.7B"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--init-adapter", default="",
                    help="從既有 LoRA adapter 續訓（只給路徑，不會重新初始化）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                    help="不用 bitsandbytes 4-bit，改用 fp16 權重做 LoRA（小模型更快也更好裝）")
    ap.add_argument("--dry-run", action="store_true", help="只檢查環境與資料格式，不訓練")
    args = ap.parse_args()

    info, problems, warnings = check_env()
    print("環境：", json.dumps(info, ensure_ascii=False))
    for w in warnings:
        print("  ⚠ " + w)
    for p in problems:
        print("  ✗ " + p)

    if not os.path.exists(args.data):
        print(f"找不到 {args.data}——先跑 make_sft.py 產生訓練集")
        return 1

    n_lines = sum(1 for line in open(args.data, encoding="utf-8") if line.strip())
    print(f"訓練資料：{n_lines} 筆 ｜ 有效資料：{os.path.basename(args.valid)}")
    print(f"設定：base={args.base} epochs={args.epochs} lr={args.lr} "
          f"batch={args.batch}×{args.accum} max_len={args.max_len}")
    print(f"      LoRA r={args.lora_r} alpha={args.lora_alpha}")

    if problems and not args.dry_run:
        print("\n依賴不完整，先用 --dry-run 驗證資料格式：")
        print(f"  python {os.path.basename(__file__)} --dry-run")
        return 1

    if args.dry_run:
        # 不需要 tokenizer 也能驗證資料形狀
        bad = 0
        with open(args.data, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    roles = [m["role"] for m in rec["messages"]]
                    if roles != ["system", "user", "assistant"]:
                        bad += 1
                        if bad <= 3:
                            print(f"  第 {i} 筆角色順序異常：{roles}")
                except Exception as e:
                    bad += 1
                    if bad <= 3:
                        print(f"  第 {i} 筆解析失敗：{e}")
        print(f"\n格式檢查：{n_lines - bad}/{n_lines} 筆正常")
        return 0

    # ── 真正開始訓練 ────────────────────────────────────────────────
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, PeftModel, get_peft_model

    use_4bit = bool(args.load_4bit and info.get("bnb"))
    print("\n載入 tokenizer / 模型（%s）…" % ("4-bit" if use_4bit else "fp16 LoRA"))
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Pascal 沒有 bf16：一律用 fp16。用 bf16 會靜默退回 fp32，VRAM 直接爆。
    compute_dtype = torch.bfloat16 if info.get("capability", (0, 0))[0] >= 8 else torch.float16

    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True)
        # 全量權重凍結，只訓 LoRA。gradient checkpointing 需要輸入有梯度，
        # 否則整條計算圖會在 embedding 那裡斷掉（錯誤訊息是 loss 的 grad_fn 為 None）。
        for p in model.parameters():
            p.requires_grad = False
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if args.init_adapter:
        # 從既有的 adapter 接著訓（例如先學會文件題、再補「沒有資料就別報數字」）。
        # 這比重訓整份資料快得多，也不會把前一輪學到的回答形狀洗掉。
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"從既有 adapter 續訓：{args.init_adapter}")
    if not args.init_adapter:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            # 只掛在注意力與 MLP 上：全部掛上去在 4B 模型上會多花 30% 時間，效果差不多少。
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        ))
    model.print_trainable_parameters()

    train_ds = load_pairs(args.data, tok, args.max_len)
    eval_ds = load_pairs(args.valid, tok, args.max_len) if os.path.exists(args.valid) else None
    print(f"樣本：train={len(train_ds)} valid={len(eval_ds) if eval_ds else 0}")

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        # Pascal 不支援 TF32；開著只會得到警告
        tf32=False,
        gradient_checkpointing=True,   # 12 GB 的關鍵：用時間換 VRAM
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        report_to=[],
        dataloader_num_workers=0,      # Windows 上多行程 dataloader 常踩坑
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, return_tensors="pt"),
    )
    trainer.train()

    out_dir = os.path.join(args.out, "lora")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # 訓練統計：打包時要寫進模型卡，事後要追「這個 gguf 是哪一版資料、幾步、多少 loss」。
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    stats = {
        "base": os.path.abspath(args.base) if os.path.isdir(args.base) else args.base,
        "rows_train": len(train_ds),
        "rows_valid": len(eval_ds) if eval_ds else 0,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "accum": args.accum,
        "max_len": args.max_len,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "load_4bit": use_4bit,
        "steps": trainer.state.global_step,
        "loss_first": round(losses[0], 4) if losses else None,
        "loss_last": round(losses[-1], 4) if losses else None,
        "device": info.get("device"),
        "torch": info.get("torch"),
    }
    with open(os.path.join(args.out, "train-stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nLoRA 已存到 {out_dir}")
    print(f"訓練統計：{json.dumps(stats, ensure_ascii=False)}")
    print("部署：把這個資料夾接到同一個 base 模型上（vLLM / llama.cpp / Ollama 都支援 LoRA 掛載）。")
    print("提醒：訓練只改語氣與格式，事實仍然來自檢索——上線時不要關掉 RAG。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
