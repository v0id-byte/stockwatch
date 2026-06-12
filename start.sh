#!/usr/bin/env bash
# StockWatch 一键启动脚本
# 用法：bash start.sh [web|once|daemon]
#   不带参数 → 启动 Web 控制台（推荐）
#   once     → 立即跑一次分析
#   daemon   → 后台守护进程模式（需要 .env 已配置）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-web}"

# ── 1. Python 检查 ────────────────────────────────────────────────────────────
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "❌ 未找到 Python 3。请先安装 Python 3.10+：https://www.python.org/downloads/"
    exit 1
fi

PY_VER=$($PYTHON -c "import sys; print(sys.version_info.major * 10 + sys.version_info.minor)")
if [ "$PY_VER" -lt 310 ]; then
    echo "❌ Python 版本过旧（需要 3.10+），当前版本：$($PYTHON --version)"
    exit 1
fi

# ── 2. 虚拟环境 ───────────────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "🔧 首次运行，创建虚拟环境..."
    $PYTHON -m venv .venv
    echo "📦 安装依赖（首次约需 1-3 分钟）..."
    .venv/bin/pip install --upgrade pip -q
    .venv/bin/pip install -r requirements.txt -q
    echo "✅ 依赖安装完成"
fi

PYTHON_VENV=".venv/bin/python"
if [ ! -f "$PYTHON_VENV" ]; then
    # Windows / 某些环境路径不同
    PYTHON_VENV=".venv/Scripts/python"
fi

# ── 3. 配置文件检查 ───────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "=========================================="
    echo "  📝 首次使用：请先填写配置文件 .env"
    echo "=========================================="
    echo ""
    echo "  必填项："
    echo "    WATCHLIST=600519,000858       ← 你的自选股（逗号分隔）"
    echo "    LLM_API_KEY=sk-xxxx           ← AI 模型 API Key"
    echo "    LLM_BASE_URL=https://...      ← API 地址（OpenAI 兼容接口）"
    echo "    LLM_MODEL=your-model-name     ← 模型名称"
    echo ""
    # 检测 Ollama 是否已安装，给出对应建议
    if command -v ollama &>/dev/null; then
        OLLAMA_RUNNING=false
        if curl -s http://127.0.0.1:11434/api/tags &>/dev/null 2>&1; then
            OLLAMA_RUNNING=true
        fi
        echo "  ✅ 检测到 Ollama 已安装（完全免费方案）："
        if $OLLAMA_RUNNING; then
            echo "     Ollama 正在运行，可直接使用以下配置："
        else
            echo "     请先启动 Ollama（另开终端运行：ollama serve）"
            echo "     然后拉取模型（首次约需几分钟）："
            echo "       ollama pull qwen2.5:7b"
        fi
        echo ""
        echo "     在 .env 中填入："
        echo "       LLM_PROVIDER=openai"
        echo "       LLM_BASE_URL=http://127.0.0.1:11434/v1"
        echo "       LLM_MODEL=qwen2.5:7b"
        echo "       LLM_API_KEY=  （留空）"
    else
        echo "  💡 没有 API Key？可以用免费本地模型（Ollama）："
        echo "     macOS/Linux 安装：curl -fsSL https://ollama.com/install.sh | sh"
        echo "     安装后拉取模型：ollama pull qwen2.5:7b"
        echo "     然后在 .env 填：LLM_BASE_URL=http://127.0.0.1:11434/v1  LLM_MODEL=qwen2.5:7b"
        echo ""
        echo "  或使用低价 API（国内可用）："
        echo "     DeepSeek：https://platform.deepseek.com  →  LLM_BASE_URL=https://api.deepseek.com/v1  LLM_MODEL=deepseek-chat"
        echo "     MiniMax： https://www.minimaxi.com       →  LLM_BASE_URL=https://api.minimaxi.com/v1  LLM_MODEL=MiniMax-M2.7"
    fi
    echo ""
    echo "  填好后重新运行：bash start.sh"
    echo ""

    # macOS：尝试用默认编辑器打开
    if command -v open &>/dev/null && [ "$(uname)" = "Darwin" ]; then
        open .env
    elif command -v xdg-open &>/dev/null; then
        xdg-open .env
    fi
    exit 0
fi

# ── 4. 启动 ───────────────────────────────────────────────────────────────────
echo ""
case "$MODE" in
    web|dashboard)
        PORT=${STOCKWATCH_WEB_PORT:-8765}
        echo "🚀 StockWatch Web 控制台启动中..."
        echo "   浏览器访问：http://localhost:$PORT"
        echo "   停止服务：按 Ctrl+C"
        echo ""
        exec "$PYTHON_VENV" main.py dashboard
        ;;
    once)
        echo "▶  立即运行一次完整分析..."
        echo ""
        exec "$PYTHON_VENV" main.py once
        ;;
    daemon)
        echo "🔄 守护进程模式启动（早9:10 / 午12:30 / 收盘15:15 自动运行）"
        echo "   停止服务：按 Ctrl+C 或 kill \$PID"
        echo ""
        exec "$PYTHON_VENV" main.py daemon
        ;;
    test)
        echo "🔍 运行自检..."
        exec "$PYTHON_VENV" main.py test
        ;;
    *)
        echo "用法：bash start.sh [web|once|daemon|test]"
        echo "  web    → 打开 Web 控制台（默认）"
        echo "  once   → 立即运行一次分析"
        echo "  daemon → 启动定时守护进程"
        echo "  test   → 自检连接"
        exit 1
        ;;
esac
