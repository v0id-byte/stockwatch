# StockWatch — A股智能盯盘系统

> 在树莓派5上运行，每日3次飞书推送，为非技术用户（我母亲）提供买卖建议。

---

## 快速开始

### 1. 配置凭证

```bash
nano ~/stockwatch/.env
```

必填项：
- `MINIMAX_API_KEY` — MiniMax API 密钥
- `FEISHU_APP_ID` / `FEISHU_APP_SECRET` — 飞书自建应用凭证
- `FEISHU_RECEIVE_ID` — 接收人 open_id/user_id/email

### 2. 修改自选股

```bash
# 编辑 .env 中的 WATCHLIST（逗号分隔代码）
WATCHLIST=600519,000858,510300,510500,159915
# 格式：6位股票代码，ETF同理（159915=创业板ETF）
# 保存后重启服务：systemctl restart stockwatch
```

### 3. 运行模式

```bash
cd ~/stockwatch && source .venv/bin/activate

# 自检（AKShare / MiniMax / 飞书）
python main.py test

# 立即运行一次完整流程
python main.py once

# 守护进程模式（按调度自动运行）
systemctl start stockwatch
systemctl status stockwatch
```

---

## 日志

```bash
# 实时日志
tail -f ~/.stockwatch/logs/stockwatch_$(date +%Y%m%d).log

# 7天保留，滚动删除
~/.stockwatch/logs/
```

---

## 数据库

```bash
# 查看 SQLite 数据
sqlite3 ~/.stockwatch/db.sqlite

# 查看最近运行记录
sqlite3 ~/.stockwatch/db.sqlite "SELECT * FROM runs ORDER BY run_ts DESC LIMIT 5;"

# 查看推送记录
sqlite3 ~/.stockwatch/db.sqlite "SELECT run_id, code, name, action, confidence, pushed FROM decisions ORDER BY run_ts DESC LIMIT 20;"
```

---

## 调度说明

每日自动运行时间（Asia/Shanghai）：
- **09:10** 早盘前：隔夜消息 + 当日策略
- **12:30** 午间：上午盘面回顾 + 下午建议
- **15:15** 收盘后：全天复盘 + 次日观察

非交易日（周末/节假日）自动跳过。

---

## v2 升级路径

```bash
cd ~/stockwatch && source .venv/bin/activate

# 1. 拉取新代码后先做增量迁移（可重复执行）
python scripts/migrate_v2.py

# 2. 编辑 .env，按模块逐个开启
nano ~/stockwatch/.env

# 3. 先自检，再重启服务
python main.py test
systemctl restart stockwatch
```

v2 新增开关默认关闭：

```bash
ENABLE_CALIBRATION=false
CALIBRATION_LOOKBACK_DAYS=5
CALIBRATION_MIN_SAMPLES=50

ENABLE_ALPHA158=false

ENABLE_LGBM=false
LGBM_MODEL_PATH=~/.stockwatch/models/lgbm.txt

ENABLE_REGIME=false

ENABLE_SECTOR=false
```

建议开启顺序：`ENABLE_REGIME` → `ENABLE_SECTOR` → `ENABLE_ALPHA158` → `ENABLE_CALIBRATION` → `ENABLE_LGBM`。

### LightGBM 离线训练

```bash
# 在 Mac 上
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt

python scripts/bootstrap_history.py
python scripts/build_training_set.py
python scripts/train_lgbm.py

# 输出 models/lgbm.txt 和 models/lgbm_meta.json 后，拷到树莓派
scp models/lgbm.* pi@<rpi_ip>:~/.stockwatch/models/
```

树莓派推理端只在 `ENABLE_LGBM=true` 时加载模型；模型缺失会记录日志并跳过。

---

## 卸载

```bash
# 停止服务
systemctl stop stockwatch
systemctl disable stockwatch

# 删除文件
rm -rf ~/stockwatch ~/.stockwatch

# 删除 systemd 单元
sudo rm /etc/systemd/system/stockwatch.service
sudo systemctl daemon-reload
```

---

## 飞书权限配置

1. 打开 [飞书开放平台](https://open.feishu.cn/app) → 找到你的应用
2. 权限管理 → 开启：
   - `im:message`（发送消息）
   - `im:message:send_as_bot`（以机器人发送）
3. 版本发布（必须发布才生效）

---

## 问题排查

```bash
# 查看服务状态
systemctl status stockwatch

# 手动跑一次看完整日志
cd ~/stockwatch && source .venv/bin/activate && python main.py once

# AKShare 数据获取失败？先更新：
pip install -U akshare

# 飞书 401/99991663 错误？
# → 检查 FEISHU_APP_ID / FEISHU_APP_SECRET 是否正确
# → 检查飞书开放平台是否已发布版本
```
