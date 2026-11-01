# 訓練資料（SFT）

這個目錄是 **v5 模型實際吃進去的全部對話資料**，原封不動。
放出來的理由很實際：`corpus/` 與 `rag/index/` 那些語料不在這個 repo（授權不同），
沒有資料的話，`train/` 裡那些腳本只是看起來能重現而已。

格式一律是每行一筆 JSON：

```json
{"messages": [
  {"role": "system",    "content": "…（三語各一套，與推論時一字不差）"},
  {"role": "user",      "content": "問題"},
  {"role": "assistant", "content": "答案＋〔來源〕"}
]}
```

## 清單

| 檔案 | 筆數 | 內容 | 用在哪一版 |
|---|---|---|---|
| `sft-train.jsonl` | 1,953 | 曦覽文件題（短答＋要點＋〔檔名:行號〕） | v1：3 epochs／735 步 |
| `sft-valid.jsonl` | 170 | 上面的驗證分割 | v1 |
| `sft-hw.jsonl` | 3,996 | 硬體事實題（xlsx 型號索引＋型號庫欄位＋拒答） | **中間產物**，未單獨訓練，已併入 v2 |
| `sft-train-v2.jsonl` | 5,626 | 主訓練集：文件 1,953 ＋ 硬體 3,673 | v2：2 epochs／704 步 |
| `sft-valid-v2.jsonl` | 489 | 主驗證集（硬體題刻意壓在 8%，才看得到硬體題有沒有退步） | v2 |
| `sft-nocontext.jsonl` | 828 | 「沒有那筆資料就不報數字」的三語樣本 | **中間產物**，已併入 fix 系列 |
| `sft-fix-train.jsonl` | 2,128 | 第一輪加固：拒答邊界 | v3：1 epoch／266 步 |
| `sft-fix-valid.jsonl` | 160 | 對應驗證集 | v3 |
| `sft-fix2-train.jsonl` | 1,928 | 第二輪加固 | v4：1 epoch／241 步 |
| `sft-fix2-valid.jsonl` | 140 | 對應驗證集 | v4 |
| `sft-fix3-train.jsonl` | 2,070 | 第三輪加固（回放＋無上下文拒答） | v5：1 epoch／259 步，從 `out-v4/lora` 續訓 |
| `sft-fix3-valid.jsonl` | 140 | 對應驗證集 | v5 |

v2 主訓練集的語種與出處分布：

| 項目 | 數字 |
|---|---|
| 繁體中文／简体中文／English | 2,241／1,724／1,661 |
| 答案帶〔來源〕／不帶 | 4,368／1,258 |

後面的 1,258 筆是**故意**的：訓練集如果每一筆都有出處，模型會學成「一定要有出處」，
單獨使用時就編檔名編行號（v4 就是這樣退步的）。不帶出處的那幾筆教它講「查不到」。

## 每筆答案的來源是哪一種

1. **曦覽自有文件**——答案裡的〔檔名:行號〕對得上主專案的文件。
2. **Wikipedia 型號／晶片組表的欄位事實**——只取數值與欄位名（96 MB、LGA 1700、SP5、
   `Zen 4 Yes`），句子由 `tools/make_hw_sft.py` 的自家模板產生，結尾附
   `〔來源：lib:List of AMD processors with 3D graphics〕` 這頁名。
   **沒有收錄任何整段第三方原文。**
3. **使用者自備的型號索引**（`CPU_型號索引`／`GPU_型號索引`）——440 筆，
   標「證據等級 A／B」，代表這一欄是原表直接寫的（A）還是需要推斷（B）。

## 授權

* 這些資料隨著 repo 以 **Apache-2.0** 釋出（程式碼與權重同）。
* 其中的硬體數值是**事實**，不受著作權保護；句子是本專案自撰。
  引用時請連原始來源一起標（Wikipedia 對應頁面採 CC BY-SA 4.0，見 `NOTICE.md`）。
* **沒有** CC BY-NC-SA 的組件庫資料（BuildCores 那一份未收錄、未訓練、未發佈），
  已逐檔掃描確認沒有 `BY-NC`／`entities-nc`／`BuildCores` 字樣。
* 內文中出現的 `Administrator`、`BUILTIN\Administrators` 是文件在講 Windows
  內建帳號與權限時的工作內容，不是這台機器的路徑或帳號洩漏。

## 重現

```bash
# 1) 硬體事實題（需要 entities.db）
python tools/make_hw_sft.py
# 2) 無資料拒答加固
python tools/make_nocontext_sft.py --repeat 3
# 3) 合併＋去重＋8% 硬體進 valid
python tools/merge_sft.py
# 4) 訓練（TITAN Xp 12 GB，fp16 LoRA，沒有 bitsandbytes）
python train/qlora.py --data data/sft-train-v2.jsonl --valid data/sft-valid-v2.jsonl \
  --epochs 2 --no-4bit --out train/out-v2
python train/qlora.py --data data/sft-fix3-train.jsonl --valid data/sft-fix3-valid.jsonl \
  --init-adapter train/out-v4/lora --epochs 1 --lr 3e-5 --no-4bit --out train/out-v5
```

步驟 1–3 需要曦覽文件與 `entities.db`，那兩樣不在這個 repo；
所以這裡的 **JSONL 才是這份發佈裡唯一可驗證的訓練輸入**。

## 這批資料的已知缺點

* 硬體題的句子全部出自同一批模板，讀起來很整齊，真實使用者的問法比這亂。
  評測裡「應該拒答的 45 題只拒了 32」有一部分就是這種問法差距。
* 文件題的答案偏向「摘錄式」，模型學會了直接引一段，而不先回應問題。
* 「欄位不存在」的問法（例如 CPU 保固年限）**沒有**對應的拒答樣本，
  下一輪要補：問句指向沒有那一欄的型號時，要回「這份資料沒有保固這個欄位」。
