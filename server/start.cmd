@echo off
REM 啟動 XinSpect-Ai 服務（本機開發／預覽用）。
REM
REM 為什麼要這支而不是直接打一長串指令：
REM   · 這台機器的主控台是 cp950，Python 的 print 一碰到「›」就會炸掉整支程式，
REM     所以一定要設 PYTHONIOENCODING。
REM   · 索引有兩個（曦覽文件、外部硬體語料），路徑相對 server/ 很容易打錯。
REM   · 舊的行程若還在跑，port 會被佔住，新的根本起不來——先清掉再起。
REM
REM 用法：
REM   start.cmd                     本機預覽（含外部語料、靜態網站）
REM   start.cmd --no-external       只答曦覽文件（給官網正式用，範圍最窄、最不會越界）
REM   start.cmd --upstream ollama   改用本機模型

setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

REM 清掉佔用 8791 的舊行程（沒有就安靜跳過）
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8791 " ^| findstr LISTENING') do (
    taskkill /F /PID %%p >nul 2>&1
)

python -I xinspect_ai_server.py --port 8791 --upstream workers-ai --serve-site ..\web\preview %*
