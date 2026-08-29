# 公告 LLM 评分规范 v1（ann_score_v1）

状态：**草案（DRAFT）**。在跑任何全量评分、看到任何收益结果之前，本文档必须定稿并冻结；
冻结后任何修改 = 升级 `PROMPT_VERSION` 并重新走 QA 流程。缓存主键含 `prompt_version`，
历史公告永久使用首次评分，供应商换模型不重打。

## 1. 模型与调用

| 项 | 值 |
|---|---|
| 模型 | `MiniMax-M3`（Anthropic 兼容 API） |
| 端点 | `MINIMAX_BASE_URL` 环境变量（不写死在代码/文档） |
| 鉴权 | `MINIMAX_API_KEY` 环境变量 / `~/.stockwatch/research.env`（gitignored；绝不进仓库） |
| temperature | 0 |
| max_tokens | 试跑后定（目标 ≤300，思考型模型需实测） |
| PROMPT_VERSION | `ann_score_v1` |

## 2. 语料范围

- **Tier-1（训练特征来源）**：`db.sqlite` `announcements` 表标题，代码 ∈ CSI500 PIT
  universe 历史成员，`published_at ≥ 2022-01-01`。
- **Tier-2（仅 QA/校准，v1 不进训练特征）**：`announcement_event_library.sqlite`
  中 `status='done'` 的 8,867 份全文（分层抽样所得，存在可得性偏差）。

### 预过滤（防成本爆炸；本身是 selection bias 源，必须留痕）

标题命中以下任一词族才送 LLM（词表冻结于此，运行时逐字使用）：

```
减持 增持 回购 业绩预告 业绩快报 预增 预减 扭亏 首亏 续亏 预盈 预亏
问询 关注函 监管函 警示函 立案 处罚 诉讼 仲裁 担保 冻结 质押 解押
限售 解禁 重大合同 中标 资产重组 收购 出售 定增 配股 可转债
退市 风险警示 ST 摘帽 停牌 复牌 商誉 减值 违约 破产 清算 占用
辞职 变更 更正 致歉
```

特征层必须保留三列使"零"可区分：`announcement_count`（全量公告数）、
`prefilter_selected_count`（命中词表数）、`llm_scored_count`（实际打分数）。

## 3. 输出 schema（受控词表）

```json
{
  "event_type": "减持|增持|回购|业绩预告|业绩快报|定期报告|问询监管|处罚立案|诉讼仲裁|担保质押|解禁限售|重大合同|资产运作|退市风险|停复牌|更正致歉|人事变动|其他",
  "direction": -2 | -1 | 0 | 1 | 2,
  "severity": 0 | 1 | 2 | 3,
  "horizon": "short|medium|long",
  "is_substantive": true | false,
  "confidence": 0.0-1.0
}
```

- `direction`：对该公司股东价值的方向判断。-2 重大利空 … +2 重大利好；纯程序性公告 = 0。
- `severity`：影响量级（0 无实质 … 3 重大）。
- `is_substantive`：是否包含实质性新信息（例行/程序性 = false）。
- `confidence` v1 **不参与**特征加权（只落库供 QA）。

## 4. Prompt（冻结体）

system：
```
你是A股公告分析员。仅根据给出的公告标题（或正文摘录）判断该公告对上市公司股东价值的
方向与量级。只输出一个JSON对象，不要输出任何其他文字。不知道或无法判断时
direction=0、severity=0、is_substantive=false。字段与取值必须严格符合给定枚举。
```

user（模板）：
```
公告标题：{title}
{可选：正文摘录（Tier-2）：{body_6000字符以内}}
输出JSON字段：event_type, direction(-2..2), severity(0..3),
horizon(short|medium|long), is_substantive(bool), confidence(0..1)
```

- prompt 中**不含**发布时间、股价、代码之外的行情信息（防泄漏；available_at 由数据管线控制）。

## 5. available_at 契约

- 每条公告标注 `publication_time_quality ∈ {EXACT_TIMESTAMP, DATE_ONLY, INFERRED}`。
- EXACT_TIMESTAMP（Asia/Shanghai）：`published_at ≤ 当日 18:00 → available_trade_date = 当日`，
  否则下一交易日。cutoff 函数离线/线上共用。
- DATE_ONLY / INFERRED：`available_trade_date = next_trade_date(publish_date)`（保守）。

## 6. 日频聚合特征（build_llm_event_features.py，全部只用 available_at ≤ trade_date 的事件）

| 特征 | 定义 |
|---|---|
| `llm_neg_sev_sum_5d` / `_20d` | Σ min(direction,0)·severity，5/20 交易日窗 |
| `llm_pos_sev_sum_5d` / `_20d` | Σ max(direction,0)·severity |
| `llm_neg_sev_decay_20d` / `llm_pos_sev_decay_20d` | 同上，指数衰减半衰期 10 交易日 |
| `llm_worst_direction_20d` | 窗内最小 direction |
| `llm_substantive_count_20d` | is_substantive 计数 |
| `llm_family_减持_20d` / `_问询处罚_20d` / `_预亏_20d` | 负面家族事件计数（预注册这三族） |
| `announcement_count_20d` / `prefilter_selected_count_20d` / `llm_scored_count_20d` | 见 §2 |
| `llm_any_event_20d` | 窗内有无已打分事件 |

无事件 = 全 0（`llm_any_event_20d=0` 提供区分）。

## 7. QA 验收（150 条分层人工样本，按类别分别报告）

| 指标 | 门槛 |
|---|---|
| `is_substantive` 二元一致 | ≥85% |
| `direction` 符号 {-1,0,+1} 一致 | ≥85% |
| high-negative flag（direction≤-1 且 severity≥2）召回 | ≥80% |
| `event_type` family 一致 | ≥80% |
| `severity` MAE | ≤0.5 |
| 20 条同题重跑一致性 | direction 符号一致 ≥85%、severity MAE ≤0.5（实测 MiniMax temperature=0 仍有相邻档漂移；下游可复现性由"每条只打一次分、缓存永久"保证，见 §8） |

任何一项不达标 → 修 prompt → **升级 PROMPT_VERSION** 重打（旧分保留在缓存中不覆盖）。

## 8. 缓存与留痕

sqlite `~/.stockwatch/history/announcement_llm_scores.sqlite`，主键
`(prompt_version, model_id, content_sha256)`；另存 `scored_at, raw_response_sha256,
parsed_schema_version, api_model_identifier, usage_input_tokens, usage_output_tokens,
retry_count, publication_time_quality`。
