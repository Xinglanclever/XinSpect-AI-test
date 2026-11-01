# 來源與授權

本 repo 的程式碼與模型權重以 **Apache-2.0** 釋出（見 `LICENSE`）。

## 模型

* 底座：**Qwen3-1.7B**，© Alibaba Cloud，Apache-2.0。
  本 repo 只發佈在其上訓練的 LoRA 與合併量化後的權重，不重複發佈底座原檔。

## 訓練與評測資料

| 來源 | 授權 | 用在哪 |
|---|---|---|
| 本 repo `data/` 的 SFT JSONL | Apache-2.0（隨 repo 釋出） | v1–v5 的訓練輸入原檔；內容組成見下表 |
| 曦覽（XinSpect）自有文件 | 專案自有 | 文件題 1,953 筆、固定回答、系統提示 |
| Wikipedia 硬體型號／晶片組表的**欄位事實** | CC BY-SA 4.0 | 硬體題 2,600 筆（只取數值與欄位名，句子自撰，答案標頁名） |
| 使用者自備 xlsx 型號表（證據等級 A/B） | 使用者自有 | 1,365 筆，未含原文 |
| 主機板組件庫 BuildCores 衍生資料 | **CC BY-NC-SA 4.0** | **未收錄、未訓練、未發佈**；程式碼僅保留 `XINSPECT_ALLOW_NC=1` 閘門 |

**沒有**使用 TechPowerUp、PCHardware 等拒絕往來來源；也沒有把任何第三方頁面的
整段文字放進訓練集或這個 repo。

## 評測題庫

`eval/questions*.json` 由本專案自行撰寫；其中硬體題的期望值由 `entities.db`
多數決產生（`eval/make_questions_hw.py`）。題庫不含第三方原文。

## 引用

若你要在文章裡引用曦覽的硬體事實，請引用原始來源（Wikipedia 對應頁面），
而不是這個模型。模型只是把事實講成句子的那一層。
