#!/bin/bash
# StockWatch 一键安装脚本（树莓派 / Linux）

set -e

echo "============================================"
echo "  StockWatch 安装脚本"
echo "============================================"

# ---- 基础依赖 ----
echo "[1/6] 安装系统依赖..."
sudo apt update -qq
sudo apt install -y -qq python3-venv python3-pip sqlite3 cron > /dev/null 2>&1

# ---- 时区 ----
echo "[2/6] 设置时区..."
sudo timedatectl set-timezone Asia/Shanghai
echo "时区: $(timedatectl show --property=Timezone --value)"

# ---- venv ----
WORKDIR="$HOME/stockwatch"
echo "[3/6] 创建 venv: $WORKDIR"
python3 -m venv "$WORKDIR/.venv"
source "$WORKDIR/.venv/bin/activate"

# ---- 依赖安装 ----
echo "[4/6] 安装 Python 依赖..."
pip install --upgrade pip -q
pip install -r "$WORKDIR/requirements.txt" -q

# ---- .env 配置 ----
if [ ! -f "$WORKDIR/.env" ]; then
    cp "$WORKDIR/.env.example" "$WORKDIR/.env"
    chmod 600 "$WORKDIR/.env"
    echo "[5/6] .env 已创建，请编辑填写凭证："
    echo "  nano $WORKDIR/.env"
else
    echo "[5/6] .env 已存在，跳过"
fi

# ---- systemd 服务 ----
echo "[6/6] 安装 systemd 服务..."
SERVICE_FILE="$WORKDIR/scripts/stockwatch.service"
if [ -f "$SERVICE_FILE" ]; then
    sudo cp "$SERVICE_FILE" /etc/systemd/system/stockwatch.service
    sudo systemctl daemon-reload
    sudo systemctl enable stockwatch
    echo "  systemd 服务已 enable"
else
    echo "  警告: $SERVICE_FILE 不存在，跳过服务安装"
fi

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  下一步操作："
echo "  1. 编辑配置: nano $WORKDIR/.env"
echo "  2. 测试连接:  cd $WORKDIR && source .venv/bin/activate && python main.py test"
echo "  3. 手动运行:  cd $WORKDIR && source .venv/bin/activate && python main.py once"
echo "  4. 查看日志:  tail -f ~/.stockwatch/logs/"
echo "  5. 启动服务:  systemctl start stockwatch && systemctl status stockwatch"
echo ""
echo "  飞书权限提示：请在 https://open.feishu.cn/app 开启 im:message 和 im:message:send_as_bot 权限"
echo "============================================"