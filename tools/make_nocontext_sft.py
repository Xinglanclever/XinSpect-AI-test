#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「沒有資料就別報數字」的加固樣本。

為什麼要補這一課：
  v2 訓練集裡每一筆答案都帶著值＋出處，模型學到的形狀是「一定會有一個數字和一個出處」。
  於是單獨使用（沒有接檢索、也沒有型號庫那一列）時，它會很順地回答
  「9800X3D 的 L3 是 25 MB，參考來源：lib:List of AMD processors with 3D graphics」——
  值是錯的，出處是編的。接上 server/ 不會發生（查不到就走查表或拒答），
  但打包出去給別人用 ollama 跑的時候，這正是最容易出事的地方。

這裡只做三件事：
  1. 規格題在**沒有給那一列**時，拒絕報數字（並說明要怎麼查）
  2. 一般知識題在沒有資料時，只講結構與方向、不給數字、不寫出處
  3. 使用者自己的機器問題，維持「讀不到」的回答

用法：
  python tools/make_nocontext_sft.py --out train/sft-nocontext.jsonl --repeat 3
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))

SYSTEM = {
    "zh-Hant": "你是 XinSpect-Ai，曦覽官網的硬體與文件助手。規格要照型號庫或〔資料片段〕回答並附來源；"
               "沒有那筆資料就直說查不到，不要憑記憶給數字，也不要編出處。"
               "不要把問題再抄一遍，也不要複述規則本身。你不能讀使用者的電腦。",
    "zh-Hans": "你是 XinSpect-Ai，曦览官网的硬件与文档助手。规格要照型号库或〔资料片段〕回答并附来源；"
               "没有那条数据就直说查不到，不要凭记忆给数字，也不要编出处。"
               "不要把问题再抄一遍，也不要复述规则本身。你不能读用户的电脑。",
    "en": "You are XinSpect-Ai, the XinSpect hardware and documentation assistant. Quote specs only "
          "from the model database or the [passages], with the source attached. If the data is not "
          "there, say so: do not answer from memory, do not invent a source, do not restate the "
          "question, and do not recite these rules. You cannot read the user's computer.",
}

# 同一個行為用多種問法，避免模型只認得一種句型
SPEC_Q = {
    "zh-Hant": [
        "{m} 有幾核心、幾執行緒？", "{m} 的 TDP 是多少？", "{m} 的 L3 快取多大？",
        "{m} 用哪個插槽？", "{m} 什麼時候推出、賣多少錢？", "{m} 的加速時脈是多少？",
        "幫我查 {m} 的規格", "{m} 的規格先給我",
    ],
    "zh-Hans": [
        "{m} 有几个核心、几个线程？", "{m} 的 TDP 是多少？", "{m} 的 L3 缓存多大？",
        "{m} 用哪个插槽？", "{m} 什么时候发布、卖多少钱？", "{m} 的加速频率是多少？",
        "帮我查 {m} 的规格", "{m} 的规格先给我",
    ],
    "en": [
        "How many cores does the {m} have?", "What is the TDP of the {m}?",
        "How much L3 cache does the {m} have?", "Which socket does the {m} use?",
        "When was the {m} released and what did it cost?", "What is the boost clock of the {m}?",
        "Look up the specs of the {m}", "Give me the spec sheet for the {m}",
    ],
}

# 型號庫裡有、也有沒有的都要練：模型不能因為「聽過這個名字」就開始報數字。
MODELS = [
    "Ryzen 7 9800X3D", "Core i9-14900K", "GeForce RTX 4090", "EPYC 9654",
    "Ryzen 5 5600X", "Core Ultra 7 265K", "Threadripper PRO 5995WX",
    "i9-15900K", "Ryzen 9 11950X", "GeForce RTX 6090", "Xeon Platinum 9999",
]

ANSWER_NO_DATA = {
    "zh-Hant": ("沒有 {m} 這一筆資料，我不報數字。\n"
                "把完整型號給我（含 K／X3D／PRO 這類後綴），我查到那一列就照抄，並附來源頁。"),
    "zh-Hans": ("没有 {m} 这一条数据，我不报数字。\n"
                "把完整型号给我（带 K／X3D／PRO 这类后缀），我查到那一行就照抄，并附来源页。"),
    "en": ("I do not have a row for the {m}, so I will not quote numbers.\n"
           "Send the full SKU (with K／X3D／PRO suffixes) and I will copy that row with its source."),
}

# 一般知識在**沒有資料**時，只給方向、不給數字、不寫出處。
# 1.7B 的通用知識不可靠（問 DDR4 位寬會答 32-bit），給了數字就是編。
GENERAL = [
    ({"zh-Hant": "PCIe 4.0 x16 的頻寬是多少？", "zh-Hans": "PCIe 4.0 x16 的带宽是多少？",
      "en": "What is the bandwidth of PCIe 4.0 x16?"},
     {"zh-Hant": "這題要具體數字，而我這邊沒有可引用的規格表，所以不給數值。\n"
                 "方向是：PCIe 頻寬＝每條 lane 的速率×lane 數，4.0 與 5.0 差在每條 lane 的速率；"
                 "要準確數字請給我主機板或 SSD 的型號，我查那一列。",
      "zh-Hans": "这题要具体数字，而我这边没有可引用的规格表，所以不给数值。\n"
                 "方向是：PCIe 带宽＝每条 lane 的速率×lane 数，4.0 与 5.0 差在每条 lane 的速率；"
                 "要准确数字请给我主板或 SSD 的型号，我查那一行。",
      "en": "This needs a number, and I have no spec table to cite here, so I will not give one.\n"
            "The shape of the answer: PCIe bandwidth is per-lane rate times lane count, and 4.0 "
            "versus 5.0 differs in the per-lane rate. For exact figures, send the board or SSD "
            "model and I will look up the row."}),
    ({"zh-Hant": "DDR5 跟 DDR4 差在哪？", "zh-Hans": "DDR5 和 DDR4 差在哪？",
      "en": "How is DDR5 different from DDR4?"},
     {"zh-Hant": "差別在架構而不是單一數字：供電管理搬到模組上、單條拆成兩個子通道、"
                 "頻率與單條容量的起點更高。\n"
                 "具體位寬／時序／支援頻率我這邊沒有可引用的資料，不報數字；"
                 "要這些請給我主機板型號，我查那一列。",
      "zh-Hans": "差别在架构而不是单一数字：供电管理搬到模块上、单条拆成两个子通道、"
                 "频率与单条容量的起点更高。\n"
                 "具体位宽／时序／支持频率我这边没有可引用的资料，不报数字；"
                 "要这些请给我主板型号，我查那一行。",
      "en": "The difference is architectural, not a single number: power management moved onto "
            "the module, each module split into two sub-channels, and the starting data rate and "
            "capacity went up.\n"
            "I have nothing to cite for the exact widths, timings or supported speeds, so I will "
            "not quote them; send the board model and I will look up its row."}),
    ({"zh-Hant": "什麼是 TDP？", "zh-Hans": "什么是 TDP？",
      "en": "What is TDP?"},
     {"zh-Hant": "TDP 是散熱設計功耗：給散熱器與主機板供電的一個共同基準，不是耗電上限；"
                 "實際功耗會隨負載與加速策略浮動。\n"
                 "個別型號的 TDP 要查型號庫那一列，把型號給我我就照抄並附來源。",
      "zh-Hans": "TDP 是散热设计功耗：给散热器与主板供电的一个共同基准，不是耗电上限；"
                 "实际功耗会随负载与加速策略浮动。\n"
                 "个别型号的 TDP 要查型号库那一行，把型号给我我就照抄并附来源。",
      "en": "TDP is a thermal design target: a shared baseline for cooler and VRM design, not a "
            "power cap; real draw moves with load and boost policy.\n"
            "A specific SKU's TDP has to come from the model database row; send the model and I "
            "will copy it with the source."}),
]

MACHINE = [
    ({"zh-Hant": "我這台的記憶體跑多少？", "zh-Hans": "我这台的内存跑多少？",
      "en": "What speed is the RAM in my machine running at?"},
     {"zh-Hant": "我讀不到你的電腦，所以不報數字。\n"
                 "記憶體時脈要由桌面版讀 SMBIOS／SPD 才拿得到；網站這邊不會掃描任何機器。",
      "zh-Hans": "我读不到你的电脑，所以不报数字。\n"
                 "内存频率要由桌面版读 SMBIOS／SPD 才拿得到；网站这边不会扫描任何机器。",
      "en": "I cannot read your machine, so no number.\n"
            "DRAM speed comes from SMBIOS/SPD on the desktop app; the website does not scan "
            "anything."}),
]


def ex(lang, q, a):
    return {"messages": [{"role": "system", "content": SYSTEM[lang]},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "train", "sft-nocontext.jsonl"))
    ap.add_argument("--repeat", type=int, default=3,
                    help="同一批樣本重複幾次：這個行為要壓過『一定有數字』的既有傾向")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    random.seed(args.seed)

    out = []
    for lang in ("zh-Hant", "zh-Hans", "en"):
        for model in MODELS:
            for qt in SPEC_Q[lang]:
                out.append(ex(lang, qt.format(m=model),
                              ANSWER_NO_DATA[lang].format(m=model)))
        for qs, ans in GENERAL + MACHINE:
            out.append(ex(lang, qs[lang], ans[lang]))

    base = len(out)
    out = out * max(1, args.repeat)
    random.shuffle(out)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"寫出 {len(out)} 筆（模板 {base} × {args.repeat}）→ {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
