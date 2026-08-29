# StockWatch 量化研究备忘录

日期: 2026-06-17

## 摘要

这轮研究的核心问题是: 在 StockWatch 当前可交易、可回测的 A 股流动股票 universe 里，技术因子、公告事件、PEAD 或负面事件排除层，能不能形成现金账户 long-only 可用的 alpha。

结论: 没有发现可部署的 long-only alpha。技术排序、公告计数、标题级 PEAD、结构化 PEAD 都没有通过扣费后相对等权 universe 的门槛。唯一保留下来的结果是一个低换手的负面公告排除层，它更像轻量风控/排雷工具，而不是收益引擎。

这不是交易建议，也不是买卖信号。StockWatch 仍应定位为家庭盯盘提醒和风险观察工具，不应把这些研究结果包装成可自动下单或可保证跑赢的策略。

## 结题判决

### 1. 能不能部署

技术上可以部署成风控/排除层，但不能部署成 alpha 引擎。

如果上线，它应该挂在被动核心之上，只负责标记和回避一批近期有负面公告风险的股票。它不应该输出“买入这些股票”的列表，也不应该替用户决定仓位。更合适的产品形态是风险标签、排除清单或 paper-monitor。

### 2. 能不能创造收益

作为 alpha，不能。

`combo_broad_negative @ 20d` 的全样本年化增量约 `+2.13%`，但主要来自 2022-2024；到 OOS 2025-2026 只剩约 `+0.23%/年`，低于预设的 `1%/年` alpha gate。随机 same-count control 证明它不是纯噪声，但这不能让 OOS 收益重新成立。

它稳定交付的是一个小的回撤缓冲: 全样本最大回撤改善约 `+1.19pct`，OOS 仍约 `+1.01pct`。这是降低风险，不是创造 alpha。账户真正的收益和回撤主要来自等权/小盘 beta，而不是这个模型。

### 3. 用户怎么用

正确用法:

- 核心仓位来自被动 beta: 宽基、ETF 或透明规则化篮子。
- 模型只做排除: 每约 20 个交易日，把命中 `combo_broad_negative` 的股票从可持有池里剔除。
- 飞书/看板只显示“当前风险标记/建议回避”，不显示“推荐买入”。

用户必须理解:

- 账户大涨大跌主要来自小盘/等权 regime 敞口。
- 排除层只尝试跳过一批风险更高的股票，换取约 `1pct` 量级的回撤垫。
- 它不挑赢家，不承诺跑赢市场，也不应该作为加仓理由。

## 方法纪律

本轮研究逐步收紧了验证标准:

- 所有 long-only 结论优先对比等权 universe，而不是只对比 CSI300。
- CSI300 超额只作为辅助信息，因为等权 universe 自带明显的小盘/等权 beta。
- 对事件类信号使用披露日之后的入场，避免公告日前未来函数。
- 对 PEAD 明确区分当前快照和真实 historical vintage。
- 对 exclusion layer 使用非重叠 20 日 rebalance，避免重叠 forward return 夸大显著性。
- 对排除层预注册 `combo_broad_negative @ 20d`，不再从多个 filter/horizon 中挑最好看的结果。
- 把 alpha gate 和 drawdown gate 分开，不把降回撤误读成 alpha。

## 已关闭的研究线

### 技术排序模型

技术 ranker 的残差 IC 在完整中性化后仍然存在，但 long-only 可交易端失败。模型主要识别残差空间中的差票，而不是选出 raw return 上会跑赢的好票。成本后 top decile/top50 相对等权 universe 为负，因此不能作为家庭现金账户的主动选股器。

结论: 停止作为 long-only alpha 继续调参。除非引入真正新的信息家族或改变目标，否则不应复活这条线。

### 公告计数因子

公告计数类特征本身没有正向 long-only alpha。部分公告类型呈现温和负向特征，尤其是减持、资本运作、风险类公告，但它们更适合作为风险提示，不适合作为买入排序因子。

结论: 不作为 alpha；可作为风险上下文或排除层输入。

### 标题级 PEAD

CNINFO 标题级解析无法稳定拿到业绩超预期的方向和幅度。标题级 signed PEAD 的正向事件对 universe 超额为负，说明标题信息不足以构造可交易 PEAD。

结论: 标题级 PEAD 关闭。

### 结构化 PEAD

结构化 PEAD 使用 AKShare/Eastmoney 的业绩预告/快报字段，补上了方向、幅度、扭亏/转亏 guard 和 next-day entry。结果仍未通过 long-only gate:

- signed-score IC 接近零或偏负。
- 部分正向组合能跑赢 CSI300 价格指数，但跑不赢等权 universe。
- OOS 与逐年拆分显示收益 alpha 不稳定。

结论: 结构化 PEAD research-only，不进入主动排名。

## 排除层结果

预注册候选:

`combo_broad_negative @ 20d`

定义:

`风险公告 OR 增减持公告 OR 资本运作公告 OR PEAD负面20日`

全样本结果:

- 过滤后相对等权 universe: `+0.1408% / 20d`, t=`2.09`
- 被剔除桶相对 universe: `-0.2915% / 20d`, t=`-2.16`
- 年化增量: `+2.13%`
- 最大回撤改善: `+1.19pct`
- 负收益期比例: `52.8% -> 49.1%`

逐年:

| 年份 | 过滤超额 / 20d | 年化增量 | 最大回撤改善 |
| --- | ---: | ---: | ---: |
| 2022 | +0.1129% | +1.33pct | +1.52pct |
| 2023 | +0.3293% | +4.30pct | +0.48pct |
| 2024 | +0.1763% | +2.89pct | +0.61pct |
| 2025 | +0.0275% | +0.81pct | +1.01pct |
| 2026 | -0.0793% | -0.84pct | +0.03pct |

OOS 2025-2026:

- 过滤超额约 `+0.0024% / 20d`
- 年化增量约 `+0.23%`
- 最大回撤改善约 `+1.01pct`

判定: alpha gate fail，drawdown gate weak pass。它不是收益 alpha；可以暂时视为 research-only 的轻量风控/排雷候选。

## 随机与波动率对照

为了确认回撤改善不是单纯剔除 35% 股票造成的，对 `combo_broad_negative @ 20d` 做了两个 same-count control。

随机剔除同样数量股票，200 次:

- 随机年化增量均值约 `+0.03%`
- 随机最大回撤改善均值约 `-0.08pct`
- 实际排除层年化增量处在随机分布约 `99%` 分位
- 实际排除层最大回撤改善处在随机分布约 `95%` 分位

结论: combo_broad 明显强过随便剔同样数量的股票。

按 `STD20` 最高波动剔同样数量股票:

- 最大回撤改善约 `+2.27pct`
- 但年化增量约 `-2.58%`
- 过滤后相对 universe 均值为负

结论: 机械去高波动能更强地降回撤，但会损失收益。combo_broad 的价值不在于最大化降波动，而是在保留收益的同时剔掉一批公告风险更高的股票。不过它没有证明自己是 alpha。

## Size Beta 解释

当前等权 universe 自身是主要收益与风险来源。它相对 CSI300 的表现很大一部分来自小盘/等权 beta，而不是模型 alpha。排除层每 20 日贡献的量级很小，在 universe 年度波动面前只是防御性点缀。

因此，任何“跑赢 CSI300”的表述都必须拆成两部分:

1. 等权/小盘 beta 对 CSI300 的暴露。
2. 排除层相对等权 universe 的增量。

目前第 2 部分不足以称为稳定 alpha。

## 实用定位

建议定位:

- 被动核心: 宽基或透明规则化篮子。
- 风险提示: 减持、资本运作、风险公告、强负面事件以警示形式展示。
- 排除层: 仅作为 research-only 候选或模拟监控，不进入“推荐买入”排序。

不建议:

- 把技术模型或 PEAD 打分作为主动买入排名。
- 只用 CSI300 超额证明 alpha。
- 继续围绕同一批技术/公告/PEAD 变体调参。
- 将排除层包装成收益增强策略。

## 后续最小动作

如果继续落地，只建议做低风险的产品化表达:

- 在 StockWatch 中把负面公告显示为风险标签，而不是买卖指令。
- 做 paper-monitor，持续记录排除层对组合回撤和踩雷率的影响。
- 若未来上线排除层，说明它是“等权/小盘暴露 + 风险排除”的风控叠加，不是 alpha 引擎。

研究上，这一轮可以结题。负结果是有效结果: 它阻止了把家人账户暴露给未经证实的主动选股模型。

---

# 第四轮：训练重启（2026-08-29）

完整协议、九项 P0 修正与执行记录见当日会话；本节只存结论与产物指针。

## 结论

1. **风险模型（回撤预测）通过全部五道预注册门禁，成为首个可部署模型。**
   LightGBM，目标 `min(close[t+1..t+20])/open[t+1]-1`（截面 rank，higher=safer），
   deploy 特征集 = 精确 Qlib Alpha158 + 23 个非重叠鲁棒因子，CSI500 PIT universe，
   purged expanding walk-forward：
   - G1 回撤 IC：retro 窗（2025-01 起）mean 0.272 / ICIR 1.30 / 正比率 89.6%（全样本 0.278/1.42/91.4%）
   - G2 坏尾富集：最差十分位预期回撤 −8.2% vs 全池 −5.4%；真实回撤 ≤−15% 识别 lift 2.47
   - G3 排除层：等权池最大回撤 −16.9%→−15.5%（+1.3pct）且总收益 +4.3pct（与第一轮"牺牲收益换回撤"的排除层不同）
   - G4 同数量随机对照 200 次：实际改善位于 100 分位
   - G5 高波动对照：0.0132 > 0.0096，非变相去波动
   报告：`~/.stockwatch/history/risk_model_v2_report.json`；模型：`models/lgbm_v2_risk.txt`（meta 含方向契约，健康门禁走 drawdown IC 分支）。

2. **收益模型再次 REJECTED**（chronological test IC −0.069；walk-forward 七道门禁 REJECTED），
   与前三轮一致。健康门禁正确拦截，线上不加载。
   报告：`~/.stockwatch/history/retrospective_candidate_report.json`。

3. **新旧风险构念诊断**：可执行化目标与旧强信号日截面 Spearman 0.98、坏尾重叠 0.91——旧先验可沿用。

4. **Replay parity**：生产打分路径（core/model_scoring，复用训练同款下载与特征代码）
   与离线面板 20 个抽样日 × 40 股 × 181 特征共 144,800 项对账零失配。

## 纪律要点（本轮新增且必须延续）

- 2025-01 起的窗口一律称 **retrospective OOS**（已被历史研究观察）；PASS ≠ 未见数据确认。
- **LOCKBOX**（2026-06-12→2026-08-26，45,421 行）物理分文件封存，只暴露存在性四项事实；
  开箱前必须先冻结 go/no-go 判据文档；开箱一次性，只做 go/no-go。
- 真正的验收 = 部署后 ≥3 个月 prospective paper-monitor。
- 生产打分契约：夜间在 CSI500 全池上批打分（与训练同一 reference universe 的截面 rank），
  runner/bot 只查表（`model_scores`），绝不在小 watchlist 批上现算模型特征。
- LLM 公告特征（MiniMax 打分进行中）标记 NOT_DEPLOYABLE_V1，仅供三臂 ablation
  （deploy / +counts / +counts+semantic）exploratory 研究。

## 数据资产（本轮新建）

- `csi500_membership_pit.parquet`：官方锚点+完整调样链逆向回放（含 300114→302132 换码处理），1025 日 × 精确 500 名
- `pit_universe_daily.parquet`：三态 PIT 网格（member/listed/ST/停牌/涨跌停），member 日 UNKNOWN 0.006%
- `stocks/`：857 只历史成员 schema-v2 双价历史（新浪主 + baostock 退市股回退，0 失败）
- `development_panel.parquet` / `lockbox_panel.parquet` / `training_panel_v2.parquet`（immutable sha 见各 report）
