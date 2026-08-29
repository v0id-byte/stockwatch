# AGENTS.md — StockWatch

> A 股家庭盯盘提醒机器人（**不是交易工具**）。本文件是 agent 进入项目的入口；
> 架构/API 摘要不在这里 —— 用 §0 的 CodeGraph MCP 工具按需拉。

---

## §0 如何用 CodeGraph 导航

| 你想... | 命令 |
|---|---|
| 理解一个模块的整体流程 | `codegraph_explore "<关键词>"`（如 `"StockRef"`、`"decision engine"`） |
| 找符号定义 | `codegraph_search <name>` |
| 改 X 会影响谁 | `codegraph_impact X --depth 2` |
| 谁调用了 X | `codegraph_callers X` |
| X 调用了什么 | `codegraph_callees X` |
| 看符号完整源码 | `codegraph_node X` |
| 项目结构 / 索引健康度 | `codegraph files` / `codegraph status` |
| 直接拿给 LLM 的上下文包 | `codegraph context "<任务>"` |

**先查再读**：进入项目先 `codegraph status` + `codegraph files`，再按需 explore。**不要**上来就 Read 一堆源文件。

---

## §1 项目布局

```
StockWatch/
├── main.py               # 常驻服务入口（scheduler 启动）
├── cli.py                # 用户命令（cmd_install/dashboard/bot/...）
├── dashboard.py          # Web 控制台（**2897 行长杆**）
├── config.py             # 从 .env 加载的 Config 类
├── analysis/             # 量化分析：factors/technical/fundamental/sentiment/
│                         #   events/regime/sector/propagation/lgbm/calibration/report
├── core/                 # runner(单次扫描主流程) / scheduler / monitor
├── decision/engine.py    # 多信号合成最终动作
├── data/                 # market(MarketData) / news / universe
├── bot/                  # service / runner / research(StockRef 解析) /
│                         #   financial_report / parser
├── push/feishu.py        # FeishuClient 飞书推送
├── utils/                # storage(SQLite 持久化) / llm / health
├── scripts/              # 回填/训练/迁移/烟测一次性脚本
├── tests/                # pytest 套件
├── models/               # 训练好的 LightGBM .txt（git tracked，**别手动改**）
└── docs/                 # 用户文档 + 宣传文案
```

**入口逻辑链**：`cli.py` 命令 → `main.py` 起 scheduler → `core/scheduler.py` 触发
`core/runner.py` → `data/*` 拿数据 → `analysis/*` 算因子/事件 → `decision/engine.py`
合成信号 → `push/feishu.py` 或 `dashboard.py` 推送。

---

## §2 关键 Hub（改前必跑 `codegraph_impact`）

| 文件 | 角色 | fan |
|---|---|---|
| `utils/storage.py` | 全项目数据落地层 | in-fan **19**（最高） |
| `data/market.py` | 行情/AKShare 入口 | in-fan 13 |
| `bot/research.py` | 自然语言→StockRef 解析 | in-fan 9 + out-fan 11（双料 hub） |
| `core/runner.py` | 单次扫描全流程 | out-fan **18**（最高） |
| `push/feishu.py` | 唯一推送通道 | in-fan 7 |

依赖形状：**`storage` 和 `market` 是叶子 hub**（被广泛读），**`runner` 是顶层 hub**
（一次性调 18 个模块）。`bot/research.py` 同时是高频被调和频繁外调——改它影响面最广。

---

## §3 高风险区

### 3.1 长杆文件（按 LOC）

| 文件 | 行 | 风险 |
|---|---|---|
| `dashboard.py` | 2897 | 路由/鉴权/表单/因子上传/自定义因子/持仓 CRUD 全堆一起。**先 `codegraph_explore "dashboard"` 看路由分区** |
| `utils/storage.py` | 832 | 全项目 Schema。改字段 = 全项目迁移 |
| `bot/research.py` | 736 | StockRef 解析 + LLM prompt。改坏 = 机器人答非所问 |
| `core/runner.py` | 504 | 单次扫描全流程 |
| `cli.py` | 465 | systemd/服务安装 |
| `config.py` | 403 | Config schema，改 = .env 兼容性 |

### 3.2 重构候选
- `dashboard.py` 拆分（需要用户明确批准）
- `bot/research.py` 把 LLM prompt 与解析逻辑分开

**默认不重构**。Review-first workflow：用户没明确说"拆 X"就别动。

---

## §4 红旗区

### 4.1 可疑目录
**没有**可疑目录（`Backup_*` / `deprecated_*` / `old_*` / `.bak_*`）。整个项目都是活的。

### 4.2 功能敏感区（不要"顺手优化"）
- `push/feishu.py` 飞书签名/webhook —— 改坏 = 推送静默失败
- `utils/storage.py` SQLite schema —— 改字段 = 老库读不出
- `analysis/lgbm.py` + `models/*.txt` —— 模型 git tracked，改训练脚本后必须重跑 `scripts/train_lgbm.py`
- `.env` 里 `LLM_API_KEY` / `FEISHU_*` —— 永远不要 commit / log / echo

### 4.3 算法常量
`analysis/factors.py` / `technical.py` 窗口长度、阈值、`ALERT_LEVELS` —— **只有用户明确说"调阈值 X"才改**。不要"感觉 5 天窗口太小就改 7 天"。

### 4.4 提交 / 推送
**Review-first**。Agent 默认：
- 改完 **不** `git commit`
- 改完 **不** `git push`
- 跑 `pytest tests/ -x` + `bash scripts/smoke_test.py` 验证后告诉用户，等审

只有用户说"commit 一下"才提交；说"push"才推送。

---

## §5 项目约定

| 项 | 值 |
|---|---|
| 默认分支 | `main` |
| 语言 | Python 3.10+ |
| 部署目标 | 树莓派 5 / macOS / Linux（systemd 或 `start.sh`）；Docker Compose 可选 |
| 数据源 | AKShare（实时行情/公告/新闻），本地 SQLite 持久化 |
| AI 接入 | OpenAI 兼容协议（Ollama / DeepSeek / MiniMax 任选），`.env` 配 `LLM_*` |
| 推送通道 | 飞书 Webhook / 机器人；Web 控制台 http://localhost:8765 |
| 模型产物 | `models/*.txt` LightGBM ranking（git tracked，**不要手动改**） |
| 配置 | `.env`（从 `.env.example` 复制），`config.py:Config` 加载 |
| License | Apache 2.0 |
| 命名 | `snake_case` 文件/函数；类名 PascalCase；私有前缀 `_` |
| **不要做** | 不要做交易/下单；不要替用户做买卖决策；不要接券商交易接口（合规边界） |

---

## §6 常见工作流

### 6.1 修 bug
1. `codegraph_explore "<错误涉及的模块>"` → `codegraph_callers <可疑函数>`
2. 读相关函数的**特定行号**（agent 已知道行号，**别整文件 Read**）
3. 改完 `pytest tests/test_<对应模块>.py -x` + `bash scripts/smoke_test.py`

### 6.2 加新功能
1. `codegraph_explore "<新功能概念>"` + `codegraph_impact <要改的入口>`
2. 在合适模块加函数（§1 知道哪个目录管什么）
3. 写测试（参考 `tests/test_fixes.py` 的 patch 风格）
4. 跑 `pytest tests/ -x`，告诉用户改了哪些文件，等 review

### 6.3 性能 / 调度排查
1. `codegraph_callers <瓶颈函数>` + `codegraph_explore "scheduler"`
2. 读 `core/runner.py` 单次耗时分布
3. 跑 `python cli.py cmd_logs` 看最近运行日志

### 6.4 LightGBM 模型
1. 改因子 → `scripts/build_fundamental_features.py` + `build_sentiment_features.py` + `build_training_set.py`
2. 训练 → `scripts/train_lgbm.py`，产物落 `models/`
3. 评估 → 看 `analysis/calibration.py` 校准曲线，确认 `lgbm_meta.json` 指标没退化
4. **模型 `.txt` 是 git tracked** —— 训练完会一起进 diff，确认无误再让用户 commit

---

## §7 什么时候**不**用 CodeGraph

- 改 typo / 改一行注释 → 直接 Edit
- 改 `.env.example` / `README.md` / `docs/*.md` → 直接 Edit
- 改 `docker-compose.yml` / `Dockerfile` / `requirements*.txt` → 直接 Edit（CodeGraph 不索引细节）
- 看 `.gitignore` / `.dockerignore` → 直接 Read
- 处理用户上传的图片 / 日志附件 → 用对应 MCP 工具
- 跑 cli 命令（`cli.py cmd_*`）→ 直接 `python cli.py cmd_logs` 等

---

## §8 一句话总结

> StockWatch = A 股家庭盯盘提醒机器人（Python 3.10+，**非交易工具**）。
> 三层架构：`data/` 行情 + `analysis/` 因子/事件 + `decision/engine.py` 信号合成，
> 结果送 `push/feishu.py` 或 `dashboard.py`；改任何模块前先 `codegraph_explore`
> 看上下文，改完不自动 commit，等用户 review。