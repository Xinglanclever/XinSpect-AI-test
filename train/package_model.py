#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把量化好的 gguf 包成一個「拿到就能跑」的 zip。

zip 裡放的是一個 Ollama 模型（Modelfile ＋ gguf）＋ 三種跑法 ＋ 一份把話講清楚的說明。
說明裡最重要的一段是「它不是什麼」：這個模型學的是曦覽講話的方式與格式，
事實仍然由 server/ 的檢索餵進來。單獨拿它去問事實，它會用自己的記憶回答——
那是 1.7B 的記憶，不是曦覽文件。

用法：
  python package_model.py --gguf dist/xinspect-ai-1.7b-q4_k_m.gguf --out dist
"""

import argparse
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

# 系統提示：給「單獨使用」的人看的版本。接上 server/ 之後，真正的提示由伺服器
# 依語言組（含〔資料片段〕），這一份只是讓模型在沒有檢索時仍然守住同一套規矩。
SYSTEM = """你是 XinSpect-Ai，曦覽官網的硬體與文件助手。規格要照型號庫或〔資料片段〕回答並附來源；沒有那筆資料就直說查不到，不要憑記憶給數字，也不要編出處。不要把問題再抄一遍，也不要複述規則本身。你不能讀使用者的電腦。"""

TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

MODELFILE = """# XinSpect-Ai 1.7B — ChatML 模板（Qwen3 家族）
# 用法：ollama create xinspect-ai -f Modelfile
FROM ./{gguf}

TEMPLATE \"\"\"{template}\"\"\"

SYSTEM \"\"\"{system}\"\"\"

PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
"""

CARD = """# XinSpect-Ai 1.7B（Q4_K_M）

用曦覽自己的文件微調出來的小模型。**這是我們自己的權重**：
底座 Qwen3-1.7B（Apache-2.0）＋ 在曦覽文件上訓練的 LoRA，合併後量化成 Q4_K_M。

## 先讀這段：它不是什麼

它學的是**曦覽怎麼講話**——用什麼詞、怎麼標出處、什麼時候說「文件裡沒寫」——
**不是**把 7,982 段文件背進權重。單獨拿它問「HEST 表解了哪些欄位」，
它會用自己的記憶回答，那是 1.7B 的記憶，不是曦覽文件。

**實測到的具體壞習慣**：沒有〔資料片段〕時，它仍然會硬生一段「出處：〔…〕」，
檔名和行號都可能是編的（訓練資料裡每一筆答案都有出處，它把「一定要有出處」學過頭了）。
所以：**要它講事實，就接上 `server/` 的檢索**。伺服器會把命中的片段放進提示，
並且在回傳前把對不上的行號修掉（`repair_locators`）。接上檢索之後，
實測 36 個寫了出處的回答，行號 36/36 都對得上真正檢索到的那一行。

## 跑起來

### 1. Ollama（最簡單）

```cmd
run-ollama.cmd
```

或手動：

```cmd
ollama create xinspect-ai -f Modelfile
ollama run xinspect-ai
```

### 2. LM Studio

把 `{gguf}` 拖進 LM Studio 的 Models 目錄，選 Q4_K_M，載入即用。
系統提示請貼 `SYSTEM-PROMPT.txt` 的內容。

### 3. llama.cpp

```cmd
llama-cli -m {gguf} -p "你是 XinSpect-Ai…" -n 512
```

## 接回官網服務

```cmd
server\\start.cmd --upstream ollama --model xinspect-ai:1.7b
```

`/health` 會顯示 `upstream: ollama`，之後所有回答都由這個模型產字，
事實仍然來自 BM25 檢索與 entities.db 查表。

## 規格

| 項目 | 值 |
|---|---|
| 底座 | Qwen3-1.7B（Apache-2.0） |
| 微調 | LoRA r=16, alpha=32, fp16；主訓練 5,626 筆（文件 1,953＋硬體 3,673），再兩輪加固 2,070 筆（拒答／無資料時不報數字） |
| 量化 | Q4_K_M（{size}） |
| 上下文 | 8,192 tokens（Modelfile 預設） |
| 語言 | 繁體中文／简体中文／English |
| 訓練資料 | 曦覽自有文件；硬體題只用**事實欄位**（核心數／時脈／插槽…）並用自家模板改寫，未貼入第三方原文；使用者自備型號表（證據等級 A/B）另佔 1,365 筆 |

## 授權

* 權重：繼承 Qwen3-1.7B 的 **Apache-2.0**。
* 訓練資料：曦覽自有文件（見 `https://github.com/Xinglanclever/XinSpect`）；硬體欄位事實取自 Wikipedia 與自備型號表，答案句子為自撰並附來源頁名。
* 這個 zip 不含任何第三方語料的原文。

## 訓練來源

{provenance}
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, os.pardir, "dist"))
    ap.add_argument("--name", default="XinSpect-Ai-1.7B-Q4_K_M")
    args = ap.parse_args()

    gguf = os.path.abspath(args.gguf)
    if not os.path.isfile(gguf):
        print(f"找不到 gguf：{gguf}")
        return 1

    out = os.path.abspath(args.out)
    stage = os.path.join(out, args.name)
    os.makedirs(stage, exist_ok=True)

    base = os.path.basename(gguf)
    size_mb = os.path.getsize(gguf) / 1e6

    # 大檔案用硬連結／複製都可以；zip 之後只留 zip，暫存目錄會刪掉。
    shutil.copy2(gguf, os.path.join(stage, base))

    with open(os.path.join(stage, "Modelfile"), "w", encoding="utf-8", newline="\n") as f:
        f.write(MODELFILE.format(gguf=base, template=TEMPLATE, system=SYSTEM))

    with open(os.path.join(stage, "SYSTEM-PROMPT.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(SYSTEM + "\n")

    with open(os.path.join(stage, "run-ollama.cmd"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("@echo off\r\n"
                "REM 把這個資料夾的 gguf 註冊成 Ollama 模型，然後直接對話。\r\n"
                "setlocal\r\n"
                "cd /d \"%~dp0\"\r\n"
                "where ollama >nul 2>&1 || (echo 找不到 ollama，先安裝 https://ollama.com/download & pause & exit /b 1)\r\n"
                "ollama create xinspect-ai -f Modelfile || (pause & exit /b 1)\r\n"
                "echo.\r\n"
                "echo 完成。之後可以：ollama run xinspect-ai\r\n"
                "ollama run xinspect-ai\r\n")

    provenance = ""
    info = os.path.join(HERE, "merged", "MERGE-INFO.json")
    if os.path.exists(info):
        with open(info, encoding="utf-8") as f:
            provenance = "```json\n" + f.read().strip() + "\n```"
    stats = os.path.join(HERE, "out", "train-stats.json")
    if os.path.exists(stats):
        with open(stats, encoding="utf-8") as f:
            provenance += "\n\n```json\n" + f.read().strip() + "\n```"
    if not provenance:
        provenance = "（訓練統計未留下，重跑 `train/qlora.py` 時會寫出 `train/out/train-stats.json`。）"

    with open(os.path.join(stage, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(CARD.format(gguf=base, size=f"{size_mb:.0f} MB", provenance=provenance))

    zip_path = os.path.join(out, args.name + ".zip")
    print(f"打包 {zip_path} …")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(os.listdir(stage)):
            path = os.path.join(stage, name)
            # 權重已經是量化過的熵很高，DEFLATE 幾乎壓不動；但仍然值得，因為同一個 zip 裡還有文字檔。
            z.write(path, arcname=f"{args.name}/{name}")
            print(f"  + {name}  {os.path.getsize(path) / 1e6:.1f} MB")

    print(f"\n完成：{zip_path}  {os.path.getsize(zip_path) / 1e9:.2f} GB")
    print(f"暫存目錄保留在 {stage}（要重打包時不必再複製 gguf）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
