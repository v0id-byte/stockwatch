# StockWatch 宣传文案

## 仓库标题

中文：

```text
StockWatch — 帮家人少盯盘的 A 股提醒机器人
```

English:

```text
StockWatch — an A-share alert bot that helps your family stop staring at stock screens
```

## 仓库简介

中文短版：

```text
一个跑在树莓派/NAS/服务器上的 A 股家庭盯盘助手：自动看自选股、持仓风险、盯价、公告新闻和盘面异动，只在值得看的时候通过 Web 控制台或飞书提醒。
```

中文完整：

```text
StockWatch 不是荐股软件，也不是自动交易工具。它更像一个家庭盯盘助手：把自选股、买入价和盯价告诉它，它会定时分析行情、公告、新闻、资金流、技术面和持仓风险；如果跌破止损、接近目标价、出现重大消息或盘面异动，再通过 Web 控制台或飞书提醒你去看。目标是减少无意义盯盘，而不是替代人的投资判断。
```

English:

```text
A personal A-share watchlist and alert bot for family use. It monitors prices, announcements, news, fund flow, technical signals and tracked positions, then surfaces them in the local web console or sends Feishu/Lark alerts only when something deserves attention.
```

## 小红书文案

标题备选：

```text
我妈每天盯盘太久，我给她做了个 A 股提醒机器人
```

```text
不想让家人一直坐着盯股票，我做了个飞书盯盘助手
```

正文：

```text
我做这个项目的原因很简单：我妈炒股时会一直坐在那里盯盘，几分钟看一次价格，看到下跌会焦虑，看到反弹又怕错过。

所以我做了一个叫 StockWatch 的小工具，目标不是“预测股票”，更不是“自动交易”，而是帮家人少盯盘。

它可以做这些事：

1. 每天早盘、午盘、收盘自动看一遍自选股
2. 买入后自动跟踪止损、目标价和模型转弱
3. 跌到自己设置的盯价时提醒
4. 有重大公告、新闻或异动时提醒
5. 没有必须看的风险时，收盘后发一句“今天不用一直盯盘”
6. 可以像问 AI 一样问：贵州茅台最近怎么样？现在行情怎么样？

我最想解决的不是“怎么一夜暴富”，而是这个问题：

家人能不能不用一直坐在屏幕前，被分时图牵着情绪走？

所以 StockWatch 的定位是：

它不替你买卖，只提醒你什么时候值得看一眼。

现在项目已经开源，Web 控制台里可以直接问行情、控制盯盘，也可以继续接飞书；适合会一点 Python、树莓派/NAS/服务器部署的人折腾。仅供学习和家庭辅助提醒，不构成投资建议。
```

标签建议：

```text
#A股 #股票工具 #飞书机器人 #Python项目 #树莓派 #开源项目 #量化研究 #家庭工具 #AI助手
```

## 技术社区文案

标题备选：

```text
用 AKShare + LLM + 飞书做了一个跑在树莓派上的 A 股家庭盯盘助手
```

```text
开源一个 A 股盯盘机器人：自选股、公告新闻、持仓风险和飞书提醒
```

正文：

```text
我最近开源了一个个人项目：StockWatch，一个面向 A 股个人/家庭场景的盯盘机器人。

它的起点不是自动交易，而是一个很朴素的需求：家人每天盯盘太久，我想把“持续看屏幕”变成“事件触发提醒”。

技术栈：

- Python
- SQLite 本地存储
- AKShare + 腾讯财经公开行情接口
- 巨潮资讯/东方财富公告新闻等公开数据源
- OpenAI-compatible / Anthropic / 本地模型 LLM 接口
- 本地 Web 控制台
- 飞书/Lark 自建应用机器人
- 可选 Alpha158 风格因子和 LightGBM 排序模型
- systemd 或 Docker Compose 部署

当前能力：

- 早盘/午盘/收盘定时分析自选股
- Web 控制台或飞书自然语言问答，例如“600519 最近一周走势如何”
- 大盘问答，例如“现在行情怎么样”
- 买入后持仓跟踪，触发止损、接近目标价、模型转弱时提醒
- 盯价提醒，触价时顺带看盘口卖压
- 重大公告/新闻提醒
- 本地 Web 控制台，可直接提问、设置盯盘、配置模型/渠道/开关
- 自定义因子上传，本地保存为社区贡献包
- 基于历史决策和 K 线的信号复盘报告
- 安心模式：无重大风险时，收盘后发“今天不用盯盘”的总结
- 提醒等级过滤：用户可以只看 critical/warning，过滤普通 info
- 模型提供商可自定义：云端模型看质量，本地模型省成本

项目边界：

- 不连接券商账户
- 不自动下单
- 不承诺收益
- 不把 LLM 输出当作投资建议

我更希望它是一个“家庭辅助提醒系统”，而不是又一个许诺收益的炒股工具。欢迎对 A 股数据、飞书机器人、树莓派部署、Dashboard 或提醒策略感兴趣的朋友提 issue/PR。
```

适合发布位置：

```text
V2EX / GitHub Discussions / 掘金 / 即刻 / X / Hacker News 中文圈 / 少数派 Matrix
```
