# LOCKBOX 开箱 go/no-go 判据 v1

状态：**冻结候选（起草于 2026-08-29，开箱前必须 commit 定稿）**。
本文档必须在任何人（含任何脚本、agent、统计摘要）接触 lockbox 段数据**之前**
commit 进仓库；开箱后对本文档的任何修改一律无效——结果只做 go/no-go 判定，
不允许看到结果后重新解释、重划窗口、调整阈值或补充"辅助口径"。

## 1. 封锁对象与工件指纹

| 工件 | 值 |
|---|---|
| `lockbox_panel.parquet` sha256 | `6828a07fd86a4f186ab1eb2686f762f5f02e03d510e97553e6779b4b0a16918a` |
| lockbox 日期范围 | 2026-06-12 → 2026-08-28（signal_date） |
| lockbox 行数 | 45,421 |
| `development_panel.parquet` sha256 | `631afc81122f73ecdb764cefa8990d9c58b0ae0aad968a164ab8b6207c58404d` |
| 母面板 immutable sha256 | `b9bbda1549d2e699f92bd0867741e7242f1a719270488fcce7e61d7f1b2ece72` |
| `models/lgbm_v2_risk.txt` sha256 | `c2be0acc0bac38d55062aba404999a8c6af8b501c4e439871600366ff6a77a2e` |
| `models/lgbm_v2_risk_meta.json` sha256 | `832ad2a940f8595f7279d6baf119b7e8f055f17d63ebe99db4ca2e5527ae262d` |

开箱脚本第一步必须逐项校验以上 sha256，任何一项不符即中止（防止工件被换）。

## 2. 一次性原则

- lockbox **只允许打开一次**。开箱时一次性评估当时全部 `validation_status ==
  VALIDATED` 的候选（截至本文冻结日：**仅风险模型 lgbm_v2_risk**；alpha 七道门禁
  REJECTED，GRU / LLM 三臂均为 exploratory，皆无资格）。
- 开箱之后 lockbox 段视为**已消耗**：此后任何新候选不得再以该段作为"未见数据"
  证据，只能走 ≥3 个月 prospective paper-monitor。
- 若 GRU challenger 在开箱前完成且通过 retrospective 七道门禁 + AND 晋级判定，
  可加入同一批次，判据用 §5；否则不加。开箱不因等待任何候选而推迟或提前——
  开箱时机由用户决定，本文件只锁"打开后怎么判"。

## 3. 风险模型判据（唯一在册候选）

**评估口径**（与 retrospective 完全一致，无新自由度）：

- 样本：lockbox 段 `universe_member == True` 且 `forward_drawdown_20d` 非空的行。
  标签需要 t+21 开盘价，故有标签的 signal_date 约止于 2026-07-29，预计 **~33 个
  交易日**——统计功效低，这是方向性 sanity check，不是显著性检验（预先声明，
  防止事后用"不显著"或"样本少"任意方向找补）。
- 打分：与 `evaluate_risk_model_v2.py` 相同——deploy 特征列在**当日 member 全池**
  上 `rank(pct) - 0.5`、NaN→0.0（float32），booster 预测得 `score`，
  `higher_is_safer` 方向不变。

**GO 条件（三条全部满足，AND）：**

1. lockbox 平均日截面 Spearman IC（score vs `forward_drawdown_20d`）**> 0**；
2. IC 为正的交易日占比 **≥ 0.60**（retro 为 0.896；0.60 是按 ~33 天功效放宽的
   方向性阈值，冻结于此，开箱后不得再调）;
3. worst-decile enrichment 方向正确：`E[drawdown | score 最低 10%] <
   E[drawdown | universe]`（即 enrichment < 0）。

**NO-GO**：任一条不满足 → 风险模型 v1 **不部署**、不进入线上决策；结果如实写入
memo 与 report，模型可继续以纯研究身份跑 paper-monitor，但不得接入 runner/bot。

**GO 的含义边界**：GO 仅允许进入部署流程（RPi + 夜间批打分 + 查表展示）。最终
验收仍是部署后 **≥3 个月 prospective paper-monitor**（预注册指标：prospective
平均 IC > 0 且 bad-tail enrichment 方向正确）；GO ≠ 提升模型在推送逻辑中的权重。

**仅记录、不参与判定的次级指标**（报告透明度用）：lockbox ICIR、bad-tail lift
（阈值 −15% 同 retro）、逐日 IC 序列、score 分布相对 development 段的漂移（PSI）、
标签覆盖率、单侧 t 检验 p 值与 block bootstrap 95% CI（只报告，不作门槛）。

## 4. 开箱程序（一步不多，一步不少）

1. 确认本文档已 commit 且工作区干净（`git status` 无未提交改动）。
2. 运行唯一开箱命令（脚本须在开箱前 review + commit，运行时要求显式
   `--unlock-lockbox` 且默认拒绝）：

   ```
   python scripts/evaluate_lockbox_one_shot.py --unlock-lockbox
   ```

3. 脚本输出 `lockbox_one_shot_report.json`（含全部 §3 指标 + go/no-go 布尔 +
   工件 sha256 回执）；报告原样 commit，不做任何编辑。
4. 按报告布尔执行 GO / NO-GO，流程终止。禁止：二次运行、改参重跑、
   对子窗口/子集重新计算、以任何形式将 lockbox 数据引入后续开发决策。

## 5. GRU challenger 判据（仅当 §2 条件满足时生效）

- GO（AND）：lockbox 平均日截面 Spearman IC（score vs `target_rank_20d` 底层收益）
  **> 0** 且 lockbox 段 topk_dropout(50,5)、TradingCosts(3,5,5) 组合净超额
  （vs 等权可交易 universe）**> 0**。
- NO-GO：任一不满足 → GRU 归档为 exploratory 记录，不部署。

## 6. 判定后动作对照表

| 结果 | 动作 |
|---|---|
| 风险模型 GO | 按计划 Stage 8 部署 RPi（scp 模型 + `.env` 追加 `ENABLE_RISK_MODEL=true` + 夜间批打分任务 + 重启三服务），随即启动 prospective paper-monitor |
| 风险模型 NO-GO | 不部署；memo 记录；本轮交付为"基础设施 + 生产打分契约"，无在线模型 |
| 任何候选 | 结果与全部次级指标进 `docs/quant_research_memo.md`，无论方向 |
