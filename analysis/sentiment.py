"""LLM 情绪分析"""
import concurrent.futures
from datetime import datetime
import math
import re

from loguru import logger

from utils.llm import get_llm_client
from data.news import NewsData
from utils.storage import Storage


_SYSTEM_PROMPT = """你是一个A股情绪分析师。请对以下新闻/公告文本的情绪打分。
每条文本输出一个分数，范围 -1（极负面）到 +1（极正面），0为中性。
只输出一个浮点数分数，不要输出其他内容。
输出格式：每行一个分数。"""
SENTIMENT_SCORE_MODEL_VERSION = "a-share-news-llm-v1"


def _score_label(score: float) -> str:
    if score >= 0.3:
        return "明显正面"
    if score >= 0.08:
        return "偏正面"
    if score <= -0.3:
        return "明显负面"
    if score <= -0.08:
        return "偏负面"
    return "中性"


def _compact_title(title: str, limit: int = 38) -> str:
    text = " ".join(str(title or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _format_news_context(scored_items: list[dict], news_count: int, sentiment: float) -> str:
    if news_count <= 0:
        return "消息面摘要: 近7日未获取到个股新闻/公告，按中性处理"
    if not scored_items:
        return f"消息面摘要: 近7日获取到{news_count}条新闻，但打分失败，按中性处理"

    ranked = sorted(scored_items, key=lambda item: abs(item["score"]), reverse=True)
    lines = [
        f"消息面摘要: 近7日新闻{news_count}条，已打分{len(scored_items)}条，整体{_score_label(sentiment)}"
    ]
    samples = []
    for item in ranked[:3]:
        direction = "正面" if item["score"] > 0.08 else "负面" if item["score"] < -0.08 else "中性"
        source = item.get("source") or "未知来源"
        title = _compact_title(item.get("title", ""))
        samples.append(f"{direction}{item['score']:+.2f} {source}: {title}")
    if samples:
        lines.append("代表新闻: " + "；".join(samples))
    return "\n".join(lines)


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


def score_news_items(news_items: list[dict]) -> list[float]:
    """Score supplied news items with the same LLM sentiment prompt."""
    if not news_items:
        return []
    return _batch_score_news(news_items, get_llm_client())


def analyze_sentiment_detail(code: str, name: str, storage: Storage | None = None) -> dict:
    """
    获取近7日新闻 + LLM 打分，返回加权情绪值和可展示摘要
    score ∈ [-1, +1]
    """
    client = get_llm_client()

    news = NewsData.get_news(code, days=7)
    if not news:
        return {
            "score": 0.0,
            "news_count": 0,
            "scored_count": 0,
            "context": _format_news_context([], 0, 0.0),
            "items": [],
        }
    if storage:
        storage.upsert_news(code, news)

    # 批量打分（最多15条，减少 token 消耗）
    batch = news[:15]
    try:
        scores = _batch_score_news(batch, client)
    except Exception as e:
        logger.warning(f"情绪分析失败 {code}: {e}")
        return {
            "score": 0.0,
            "news_count": len(news),
            "scored_count": 0,
            "context": _format_news_context([], len(news), 0.0),
            "items": [],
        }
    if len(scores) < len(batch):
        scores.extend([0.0] * (len(batch) - len(scores)))

    # 时间衰减加权：越近的权重越大
    total_score = 0.0
    total_weight = 0.0
    now_ts = datetime.now().timestamp()
    scored_items = []

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
        scored_items.append({
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "ts": item.get("ts", ""),
            "score": score,
        })
        if storage:
            storage.update_news_sentiment(
                code, item["title"], item["ts"], score,
                model_version=SENTIMENT_SCORE_MODEL_VERSION,
            )

    sentiment = total_score / total_weight if total_weight > 0 else 0.0
    logger.info(f"情绪分析 {code}({name}): score={sentiment:.3f} (基于{len(scores)}条新闻)")
    sentiment = round(sentiment, 4)
    return {
        "score": sentiment,
        "news_count": len(news),
        "scored_count": len(scored_items),
        "context": _format_news_context(scored_items, len(news), sentiment),
        "items": scored_items,
    }


def analyze_sentiment(code: str, name: str, storage: Storage | None = None) -> float:
    """
    获取近7日新闻 + LLM 打分，返回加权情绪值
    score ∈ [-1, +1]
    """
    return analyze_sentiment_detail(code, name, storage=storage)["score"]


def batch_sentiment_details(codes_names: list[tuple[str, str]], storage: Storage | None = None) -> dict[str, dict]:
    """并发分析多只股票情绪详情（并发≤3）"""
    results = {}
    semaphore = __import__("threading").Semaphore(3)

    def _work(code, name):
        with semaphore:
            results[code] = analyze_sentiment_detail(code, name, storage=storage)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_work, c, n) for c, n in codes_names]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.warning(f"并发情绪分析异常: {e}")

    return results


def batch_sentiment(codes_names: list[tuple[str, str]], storage: Storage | None = None) -> dict[str, float]:
    """并发分析多只股票情绪（并发≤3）"""
    details = batch_sentiment_details(codes_names, storage=storage)
    return {code: item.get("score", 0.0) for code, item in details.items()}
