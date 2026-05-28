"""LLM 情绪分析"""
import concurrent.futures
from datetime import datetime
import math
import re

from loguru import logger

from utils.llm import get_llm_client
from data.news import NewsData
from config import get_config
from utils.storage import Storage


_SYSTEM_PROMPT = """你是一个A股情绪分析师。请对以下新闻/公告文本的情绪打分。
每条文本输出一个分数，范围 -1（极负面）到 +1（极正面），0为中性。
只输出一个浮点数分数，不要输出其他内容。
输出格式：每行一个分数。"""


def _batch_score_news(news_items: list[dict], client) -> list[float]:
    """对一批新闻一次性调用 LLM 打分"""
    if not news_items:
        return []

    texts = []
    for item in news_items:
        combined = f"来源：{item['source']} 时间：{item['ts']}\n标题：{item['title']}"
        if item.get("content"):
            combined += f"\n内容：{item['content'][:200]}"
        texts.append(combined)

    prompt_text = "\n---\n".join([f"[{i+1}] {t}" for i, t in enumerate(texts)])
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text[:3000]},
    ]

    raw = client.chat(messages)
    scores = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # 取最后一个非数字字符后的内容
        try:
            # 尝试直接解析
            score = float(line)
        except ValueError:
            # 去掉行号等前缀后解析
            nums = re.findall(r"-?(?:\d+(?:\.\d+)?|\.\d+)", line)
            if nums:
                score = float(nums[-1])
            else:
                score = 0.0
        scores.append(max(-1.0, min(1.0, score)))
    return scores


def analyze_sentiment(code: str, name: str, storage: Storage | None = None) -> float:
    """
    获取近7日新闻 + LLM 打分，返回加权情绪值
    score ∈ [-1, +1]
    """
    cfg = get_config()
    client = get_llm_client()

    news = NewsData.get_news(code, days=7)
    if not news:
        return 0.0
    if storage:
        storage.upsert_news(code, news)

    # 批量打分（最多15条，减少 token 消耗）
    batch = news[:15]
    try:
        scores = _batch_score_news(batch, client)
    except Exception as e:
        logger.warning(f"情绪分析失败 {code}: {e}")
        return 0.0

    # 时间衰减加权：越近的权重越大
    total_score = 0.0
    total_weight = 0.0
    now_ts = datetime.now().timestamp()

    for item, score in zip(batch, scores):
        try:
            dt = datetime.fromisoformat(item["ts"])
            age_days = (now_ts - dt.timestamp()) / 86400
            weight = math.exp(-0.15 * age_days)
            total_score += score * weight
            total_weight += weight
        except Exception:
            total_score += score
            total_weight += 1.0
        if storage:
            storage.update_news_sentiment(code, item["title"], item["ts"], score)

    sentiment = total_score / total_weight if total_weight > 0 else 0.0
    logger.info(f"情绪分析 {code}({name}): score={sentiment:.3f} (基于{len(scores)}条新闻)")
    return round(sentiment, 4)


def batch_sentiment(codes_names: list[tuple[str, str]], storage: Storage | None = None) -> dict[str, float]:
    """并发分析多只股票情绪（并发≤3）"""
    results = {}
    semaphore = __import__("threading").Semaphore(3)

    def _work(code, name):
        with semaphore:
            results[code] = analyze_sentiment(code, name, storage=storage)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_work, c, n) for c, n in codes_names]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.warning(f"并发情绪分析异常: {e}")

    return results
