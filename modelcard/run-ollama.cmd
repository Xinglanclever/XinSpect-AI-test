@echo off
REM 把這個資料夾的 gguf 註冊成 Ollama 模型，然後直接對話。
setlocal
cd /d "%~dp0"
where ollama >nul 2>&1 || (echo 找不到 ollama，先安裝 https://ollama.com/download & pause & exit /b 1)
ollama create xinspect-ai -f Modelfile || (pause & exit /b 1)
echo.
echo 完成。之後可以：ollama run xinspect-ai
ollama run xinspect-ai
