@echo off
:: StockWatch 一键启动脚本（Windows）
:: 双击运行，或在命令提示符 / PowerShell 中执行：start.bat [web|once|daemon|test]

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=web"

:: ── 1. Python 检查 ────────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到 Python。请先安装 Python 3.10+：https://www.python.org/downloads/
    echo    安装时勾选 "Add Python to PATH"
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set "PY_FULL=%%v"
for /f "tokens=1,2 delims=." %%a in ("%PY_FULL%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)
if %PY_MAJOR% LSS 3 (
    echo ❌ Python 版本过旧（需要 3.10+），当前：%PY_FULL%
    pause & exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo ❌ Python 版本过旧（需要 3.10+），当前：%PY_FULL%
    pause & exit /b 1
)

:: ── 2. 虚拟环境 ───────────────────────────────────────────────────────────────
if not exist ".venv\" (
    echo 🔧 首次运行，创建虚拟环境...
    python -m venv .venv
    echo 📦 安装依赖（首次约需 1-3 分钟）...
    .venv\Scripts\pip install --upgrade pip -q
    .venv\Scripts\pip install -r requirements.txt -q
    echo ✅ 依赖安装完成
)

set "PYTHON_VENV=.venv\Scripts\python.exe"

:: ── 3. 配置文件检查 ───────────────────────────────────────────────────────────
if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo ==========================================
    echo   📝 首次使用：请先填写配置文件 .env
    echo ==========================================
    echo.
    echo   必填项：
    echo     WATCHLIST=600519,000858       ^← 你的自选股（逗号分隔）
    echo     LLM_API_KEY=sk-xxxx           ^← AI 模型 API Key
    echo     LLM_BASE_URL=https://...      ^← API 地址
    echo     LLM_MODEL=your-model-name     ^← 模型名称
    echo.
    echo   免费本地模型（不需要 API Key）：
    echo     1. 安装 Ollama：https://ollama.com/download
    echo     2. 在命令行运行：ollama pull qwen2.5:7b
    echo     3. 在 .env 填：
    echo        LLM_BASE_URL=http://127.0.0.1:11434/v1
    echo        LLM_MODEL=qwen2.5:7b
    echo        LLM_API_KEY=  （留空）
    echo.
    echo   填好后重新双击 start.bat
    echo.
    :: 用记事本打开配置文件
    start notepad .env
    pause
    exit /b 0
)

:: ── 4. 启动 ───────────────────────────────────────────────────────────────────
echo.
if "%MODE%"=="web" goto :start_web
if "%MODE%"=="dashboard" goto :start_web
if "%MODE%"=="once" goto :start_once
if "%MODE%"=="daemon" goto :start_daemon
if "%MODE%"=="test" goto :start_test
echo 用法：start.bat [web^|once^|daemon^|test]
echo   web    → 打开 Web 控制台（默认）
echo   once   → 立即运行一次分析
echo   daemon → 启动定时守护进程
echo   test   → 自检连接
pause & exit /b 1

:start_web
set "PORT=8765"
echo 🚀 StockWatch Web 控制台启动中...
echo    浏览器访问：http://localhost:%PORT%
echo    停止服务：关闭此窗口或按 Ctrl+C
echo.
%PYTHON_VENV% main.py dashboard
goto :end

:start_once
echo ▶  立即运行一次完整分析...
echo.
%PYTHON_VENV% main.py once
goto :end

:start_daemon
echo 🔄 守护进程模式启动（早9:10 / 午12:30 / 收盘15:15 自动运行）
echo    停止服务：关闭此窗口或按 Ctrl+C
echo.
%PYTHON_VENV% main.py daemon
goto :end

:start_test
echo 🔍 运行自检...
%PYTHON_VENV% main.py test
goto :end

:end
if errorlevel 1 (
    echo.
    echo ❌ 运行出错，请查看上方错误信息。
    echo    常见原因：.env 配置有误 / 网络不通 / API Key 错误
    pause
)
endlocal
