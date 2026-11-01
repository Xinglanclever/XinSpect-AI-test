# 測試紀錄：2026-11-01

對象：`xinspect-ai:1.7b-v5`（Q4_K_M，1,107 MB）接 `server/xinspect_ai_server.py`，
本機 Ollama 產生，**不開** `XINSPECT_ALLOW_NC`（也就是正式站會跑的那種組態）。

評測集：110 題＝文件題 31＋外部硬體題 20＋硬體查表題 14（in）、
越界題 28＋外部越界 10＋硬體越界 7（out）。其中 14 題帶 `expect_all` 期望值
（值直接從 `entities.db` 多數決推出來，不手寫）。

## 一、結果

| 指標 | 修正前（09:48 那一跑） | 修正後（10:0x） |
|---|---|---|
| 應該回答的 65 題 | 63（97%） | 63（97%） |
| 應該拒答的 45 題 | 33（73%） | 32（71%） |
| 帶期望值的 14 題 | **9（64%）** | **14（100%）** |
| 出處行號對得上 | 32/32（100%） | 26/27（96%） |
| 延遲中位數／最慢 | 0.3 s／8.3 s | 0.0 s／3.9 s |

兩份逐題結果都留在 `eval/`：`RESULT-2026-11-01-before-fix.json`、
`RESULT-2026-11-01-nonc.json`。另外 `RESULT-2026-10-31-dev-nc.json` 是前一天
**開了 NC 資料**的開發機那一跑（14/14），可以拿來對照「為什麼之前看起來是好的」。

拒答率從 73% 掉到 71% 不是退步，是同一天多答對一題越界題的取捨；
這種 1 題的變動在 45 題樣本上本來就是雜訊，不要當成指標在追。

## 二、這輪抓到並修掉的 bug：晶片組查詢被鎖在 NC 閘門後面

**症狀**：問「B650 支援哪些 CPU 世代？」「Z790 是什麼架構的晶片組？」
五題全部答成文件裡某段講 CPU 的廢話，`kind_tag` 是空的（表示沒走查表）。

**原因**：`EntityStore.resolve()` → `component_facts()` 的第一行是

```python
comp = self._components()
if not comp:
    return []
```

`_components()` 只讀 `entities-nc.db`（CC BY-NC-SA），沒設 `XINSPECT_ALLOW_NC=1`
就回 `False`。可是晶片組那一列（`lib-chipset` 590 筆、`wiki-chipset` 10 筆）
其實在 **entities.db（CC BY-SA）**，根本不需要 NC 授權；結果整個函式提前 return，
`chipset_facts()` 連跑都沒跑，題目掉回 BM25，檢索到什麼就講什麼。

**為什麼之前沒發現**：前一天的 14/14 是在開發機上開著 `XINSPECT_ALLOW_NC=1` 跑的。
換成正式站的組態（不開 NC）就現形了——這正是「測試要跑兩套組態」的理由。

**修法**（`rag/entities.py`）：把只讀 entities.db 的晶片組路徑搬到閘門**之前**，
組件庫只拿來補充「這顆晶片組有幾張板」，沒有也不影響答案：

```python
if CHIPSET_TOKEN.search(q):
    facts.extend(self.chipset_facts(q))   # 只讀 entities.db

comp = self._components()
if not comp:
    return facts                          # NC 閘門只擋組件庫的部分
```

**驗證**（不開 NC，直接呼叫 `resolve()`）：

```
B650   → CPU 支援（來源逐項）：Zen 4 Yes、Zen 5 Yes；可超頻：Yes；PCIe 5.0：x16 slot: No…
X870E  → Zen 4 Yes、Zen 5 Yes；PCIe 5.0：x16 slot: Yes、M.2 slot + 4× GPP: Yes
Z790   → 來源標示架構：Raptor Lake（晶片組出道時搭配的世代，不等於完整支援清單）
```

整份評測重跑後硬體值 14/14。**權重沒有動**——這是檢索層的 bug，不是模型的錯。

## 三、還沒修、而且應該修的

### 1. 查表不看問句在問哪個欄位（最嚴重）

```
Q: EPYC 9654 的保固是幾年？
A: 這是查表結果，不是推測：- EPYC 9654（Wikipedia：lib:Epyc）：process_nm TSMC N5；
   cores 96；… （整列規格）
```

問句只提到型號，`resolve()` 就把整列能講的欄位倒出來，**保固年限從頭到尾沒出現**。
讀者看到「這是查表結果」＋一串真數字，很容易以為問題被回答了。

同類：`Ryzen 9 7945HX 可以裝在哪張主機板上？` → 回傳行動版 CPU 那一列
（沒有 socket），照樣用「這是查表結果」開頭。

要修的話，查表入口得先判斷問句要的欄位（`warranty`／`socket`／`tdp`…），
對不上就回「這份資料沒有這個欄位」，而不是把整列丟出去。

### 2. 越界題有 13/45 沒拒

真越界且答錯的：

* 「現在幾點？」→ 撿到文件裡的時鐘對照表（`Stopwatch`／QPC），開始講單調鐘。
* 「What is the capital of Australia?」→ 答了一段「安裝事實」的說明。
* 「Intel Core i9-15900K 有幾核心？」（不存在的 SKU）→ 講了一段電壓／頻率的文件段落。
* 「B650 支援幾條 PCIe 6.0 通道？」→ 拿 PCIe 5.0 的支援表回答（來源沒有 6.0）。

另外有一批「越界題」其實**題庫標錯了**，模型答得對，卻被算成沒拒：
「曦覽會不會上傳我的資料？」「三源對帳是什麼意思？」「曦覽的官方網站是哪一個網址？」
「SPD 是從哪條路徑讀的？」——這些文件裡都有答案。下一版題庫要把它們移到 in_domain，
不然拒答率永遠被壓在 7 成。

### 3. 出處檔名會幻覺（26/27）

```
Q: How does the Deep Bench work?
答案寫：Source: XinSpect.md:843
實際檢索到：README.en.md:843
```

`repair_locators()` 只修**同一個檔案內**的行號；檔名沒被檢索到就原樣留著，
故意不猜——猜錯檔名比留錯更糟。這一題被評測抓出來，是它該做的事。

### 4. 文風

1.7B 的老毛病：開頭常是「文件中段的一句話」，不先回答問題；偶爾把問句再抄一遍；
沒有片段時硬生「出處：〔檔名:行號〕」。這些是權重層面的問題，要再訓一輪才會動，
不是 prompt 能補的。

## 四、方法與限制

* 「答了／拒了」的判定：in 題看 `grounded` 且不是 machine-specific；
  out 題看 `not grounded` 或 machine-specific（機器專屬固定回答算正確地沒答）。
* 硬體題的期望值用**多數決**：同型號多列至少兩列一致才採用，寧可少一題也不用錯答案。
* 這一輪**沒有** NC 資料，所以「主機板相容／擴充」那類要靠組件庫的題目不在結果裡；
  那部分只能在開發機驗，不能當成可對外宣稱的數字。
* 延遲中位數 0.0 秒是因為一半以上題目走查表或固定回答，**完全沒有呼叫模型**；
  真正落到 1.7B 生成的題目在 1–4 秒之間。

## 五、重跑

```cmd
cd server
set XINSPECT_RATE_LIMIT=10000
start.cmd --upstream ollama --model xinspect-ai:1.7b-v5 --port 8793

python eval/make_questions_hw.py
python eval/run_e2e.py --endpoint http://127.0.0.1:8793/v1/xinspect/ask --out eval/out.json
```

要驗「正式站組態」就**不要**設 `XINSPECT_ALLOW_NC`；要驗 NC 那幾題才設 `=1`，
兩份結果分開存，不要混在一起看。
