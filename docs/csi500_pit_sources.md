# 中证500历史 PIT 成分与权重来源

## 结论

2026-07-16 的联网烟测只在 `/tmp` 验证了当前500只成分、单一月末500只权重快照和官网搜索候选；公告详情接口返回403，且默认历史目录没有写入 manifest、报告或 parquet。因此这不是已经落地的历史 PIT 数据集，下面描述的是脚本的数据契约和取得完整原始公告链后才能达到的能力。

- **历史成分：可恢复，但必须保存原始公告链。** 中证指数官网公开当前完整500只样本，并公开历次定期/临时调样公告、明确生效日及调入调出名单。`capture_csi500_official.py` 原样保存接口 JSON、附件和 SHA-256；`build_csi500_pit_membership.py` 从当前官方锚点逆推每次调样，并要求每个版本恰好500只。
- **历史每日权重：免费公开入口证据不足，必须 fail-closed。** 官网公开文件和 AKShare 官方封装都只给当前成分及一个当前/月末权重快照，没有历史日期参数。代码只生成 `csi500_current_weight_snapshot.parquet`，并写入 `snapshot_only=true`、`usable_for_historical_backtest=false`；绝不生成或回填 `csi500_weights_pit.parquet`。
- **不能直接把 active-only 成分文件冒充完整交易标记。** `csi500_membership_pit.parquet` 每日只含500条 `is_member=true`。提供 `--scope <trade_date,code表>` 时，才额外生成逐行显式 true/false 的 `csi500_membership_grid_pit.parquet`。ST、上市、停牌、涨跌停仍须由其他 PIT 数据源补齐。

## 使用

```bash
.venv/bin/python scripts/capture_csi500_official.py \
  --start 2022-01-01 --end 2026-07-16

.venv/bin/python scripts/build_csi500_pit_membership.py \
  --calendar ~/.stockwatch/history/training_set.parquet \
  --scope ~/.stockwatch/history/training_set.parquet
```

输出保留：

- `trade_date, code, index_code, is_member`
- `published_at, available_at, effective_from`
- `source_url, source_hash, source_chain_hash, extractor_version`

公告只有日期没有时刻时，`available_at` 保守设为次日00:00（Asia/Shanghai）；成员变更仅在公告写明的“收市后生效”日期之后的首个输入交易日启用。交易日必须由调用方提供，脚本不拿工作日猜交易所日历。

## 证据与边界

1. **[R1, Tier A] 中证500编制方案。** 中证指数有限公司，2022，说明半年调样时间以及以调整市值计算指数；这意味着仅知道调入调出名单不足以恢复每日权重。[官方 PDF](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000904_Index_Methodology_cn.pdf)
2. **[R2, Tier A] 中证指数股票指数计算与维护细则。** 中证指数有限公司，2023，说明公司事件、股本和除数会引起日常维护，并指出公司事件数据服务文件；因此不能把定期调样日权重静态铺到整个历史区间。[官方 PDF](https://oss-ch.csindex.com.cn/notice/20230908165124-%E3%80%8A%E4%B8%AD%E8%AF%81%E6%8C%87%E6%95%B0%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8%E8%82%A1%E7%A5%A8%E6%8C%87%E6%95%B0%E8%AE%A1%E7%AE%97%E4%B8%8E%E7%BB%B4%E6%8A%A4%E7%BB%86%E5%88%99%E3%80%8B.pdf)
3. **[R3, Tier A] 当前完整成分文件。** 中证指数官网静态文件，路径固定为当前快照，文件内含快照日期及500只成分。[官方 XLS](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/000905cons.xls)
4. **[R4, Tier A] 当前收盘权重文件。** 中证指数官网静态文件，文件内是单一快照日的500只权重；没有历史日期请求参数。[官方 XLS](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/closeweight/000905closeweight.xls)
5. **[R5, Tier A] 历史调样附件样例。** 2024年11月公告附件明确给出中证500调出/调入名单，公告正文另给收市后生效日。[官方 PDF](https://oss-ch.csindex.com.cn/notice/20241129172348-%E9%99%84%E4%BB%B6%EF%BC%9A%E9%83%A8%E5%88%86%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC%E8%B0%83%E6%95%B4%E5%90%8D%E5%8D%95.pdf)
6. **[R6, Tier A] 历史调样附件样例。** 2026年5月附件继续提供中证500完整调出/调入对；用于验证2022—2026公告格式延续。[官方 PDF](https://oss-ch.csindex.com.cn/notice/20260529155822-%E9%99%84%E4%BB%B6%EF%BC%9A%E9%83%A8%E5%88%86%E6%8C%87%E6%95%B0%E6%A0%B7%E6%9C%AC%E8%B0%83%E6%95%B4%E5%90%8D%E5%8D%95.pdf)
7. **[R7, Tier A] AKShare 官方指数文档。** `index_stock_cons_csindex` 与 `index_stock_cons_weight_csindex` 的参数只有指数代码，权重单位为百分数，没有历史日期参数；它只是官方当前文件的便捷封装。[AKShare 文档](https://akshare.akfamily.xyz/data/index/index.html)

“免费公开历史权重不可得”是对上述公开入口的审计结论，不代表中证指数付费数据服务或授权供应商不存在历史权重。若以后取得有授权的逐日样本/调整股本/公司事件文件，应作为新来源独立捕获并重新验收，不能静默替换本数据集。
