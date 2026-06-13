"""市场数据获取（腾讯财经主源，AKShare 备用源）"""
import re
import json
import requests
from loguru import logger

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://gu.qq.com/'
}


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _to_float(value) -> float:
    try:
        return float(str(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return 0.0


class MarketData:
    @staticmethod
    def get_realtime_quote(codes: list[str]) -> dict[str, dict]:
        if not codes:
            return {}
        result = MarketData._get_realtime_quote_tencent(codes)
        if not result:
            logger.warning("腾讯行情接口无响应，切换 AKShare 备用源")
            result = MarketData._get_realtime_quote_akshare(codes)
        return result

    @staticmethod
    def _get_realtime_quote_tencent(codes: list[str]) -> dict[str, dict]:
        code_str = ','.join([
            f"sh{c}" if c.startswith(('6', '5')) else f"sz{c}"
            for c in codes
        ])
        try:
            url = f"https://qt.gtimg.cn/q={code_str}"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = 'gb18030'
            lines = resp.text.strip().split('\n')
            result = {}
            for line in lines:
                m = re.search(r'v_(\w+)="([^"]+)"', line)
                if not m:
                    continue
                raw_code = m.group(1)
                if len(raw_code) < 4:
                    continue
                code = raw_code[2:]  # strip sh/sz prefix
                fields = m.group(2).split('~')
                if len(fields) < 40:
                    continue
                try:
                    buy_volume_5 = sum(_to_float(fields[i]) for i in [10, 12, 14, 16, 18] if len(fields) > i)
                    sell_volume_5 = sum(_to_float(fields[i]) for i in [20, 22, 24, 26, 28] if len(fields) > i)
                    result[code] = {
                        "name": fields[1].strip(),
                        "open": float(fields[5]) if fields[5] else 0,
                        "high": float(fields[33]) if fields[33] else 0,
                        "low": float(fields[34]) if fields[34] else 0,
                        "close": _to_float(fields[3]),
                        "volume": float(fields[37]) if fields[37] else 0,
                        "pct_change": float(fields[32]) if fields[32] else 0,
                        "quote_time": fields[30] if len(fields) > 30 else "",
                        "outer_volume": _to_float(fields[7]) if len(fields) > 7 else 0,
                        "inner_volume": _to_float(fields[8]) if len(fields) > 8 else 0,
                        "buy_volume_5": buy_volume_5,
                        "sell_volume_5": sell_volume_5,
                    }
                except (ValueError, IndexError):
                    continue
            if len(result) < len(codes):
                missing = [c for c in codes if c not in result]
                logger.debug(f"腾讯行情缺失 {len(missing)} 只（解析失败或无数据）: {missing[:10]}")
            return result
        except Exception as e:
            logger.warning(f"腾讯实时报价失败: {e}")
            return {}

    @staticmethod
    def _get_realtime_quote_akshare(codes: list[str]) -> dict[str, dict]:
        """AKShare 备用实时报价（缺少五档盘口数据，已标注）。"""
        try:
            import akshare as ak
            df = ak.stock_zh_a_spot_em()
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            target = set(codes)
            result = {}
            for _, row in df.iterrows():
                code = str(row.get("代码", "")).zfill(6)
                if code not in target:
                    continue
                result[code] = {
                    "name": str(row.get("名称", "")),
                    "open": _to_float(row.get("今开")),
                    "high": _to_float(row.get("最高")),
                    "low": _to_float(row.get("最低")),
                    "close": _to_float(row.get("最新价")),
                    "volume": _to_float(row.get("成交量")),
                    "pct_change": _to_float(row.get("涨跌幅")),
                    "quote_time": "",
                    # 五档盘口 AKShare 不提供，填 0 以保持字段兼容
                    "outer_volume": 0,
                    "inner_volume": 0,
                    "buy_volume_5": 0,
                    "sell_volume_5": 0,
                }
            logger.info(f"AKShare 备用报价返回 {len(result)} 只")
            return result
        except Exception as e:
            logger.warning(f"AKShare 实时报价备用源失败: {e}")
            return {}

    @staticmethod
    def get_daily_kline(code: str, period: str = "daily", adjust: str = "qfq", limit: int = 320) -> list[dict]:
        """日线 K 线（腾讯主源，AKShare 备用）。"""
        result = MarketData._get_daily_kline_tencent(code, limit)
        if not result:
            logger.warning(f"腾讯K线无响应 {code}，切换 AKShare 备用源")
            result = MarketData._get_daily_kline_akshare(code, limit)
        return result

    @staticmethod
    def _get_daily_kline_tencent(code: str, limit: int = 320) -> list[dict]:
        try:
            prefix = "sh" if code.startswith(('6', '5')) else "sz"
            url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                   f"?_var=kline_dayfqzf&param={prefix}{code},day,,,{limit},qfq&r=0.1")
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = 'utf-8'
            text = resp.text.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start < 0 or end <= start:
                return []
            data = json.loads(text[start:end])
            stock_data = data.get('data', {}).get(f"{prefix}{code}", {})
            day_data = stock_data.get('qfqday') or stock_data.get('day') or []
            records = []
            for item in day_data:
                if len(item) < 6:
                    continue
                try:
                    records.append({
                        "trade_date": item[0],
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "volume": float(item[5]),
                        "amount": 0,
                    })
                except (ValueError, IndexError):
                    continue
            return records
        except Exception as e:
            logger.warning(f"腾讯K线获取失败 {code}: {e}")
            return []

    @staticmethod
    def _get_daily_kline_akshare(code: str, limit: int = 320) -> list[dict]:
        """AKShare 备用 K 线（前复权）。"""
        try:
            import akshare as ak
            from datetime import date, timedelta
            end = date.today().isoformat().replace("-", "")
            start = (date.today() - timedelta(days=limit * 2)).isoformat().replace("-", "")
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily", adjust="qfq",
                start_date=start, end_date=end
            )
            if df is None or len(df) == 0:
                return []
            records = []
            for _, row in df.tail(limit).iterrows():
                try:
                    records.append({
                        "trade_date": str(row["日期"])[:10],
                        "open": float(row.get("开盘") or 0),
                        "close": float(row.get("收盘") or 0),
                        "high": float(row.get("最高") or 0),
                        "low": float(row.get("最低") or 0),
                        "volume": float(row.get("成交量") or 0),
                        "amount": float(row.get("成交额") or 0),
                    })
                except (ValueError, KeyError):
                    continue
            logger.info(f"AKShare 备用K线返回 {len(records)} 条 {code}")
            return records
        except Exception as e:
            logger.warning(f"AKShare K线备用源失败 {code}: {e}")
            return []

    @staticmethod
    def get_index_kline(index_code: str = "sh000001", limit: int = 800) -> list[dict]:
        """腾讯财经指数日线，index_code 形如 sh000001 / sh000300。"""
        try:
            url = (f"https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
                   f"?_var=kline_day&param={index_code},day,,,{limit}")
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = 'utf-8'
            text = resp.text.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start < 0 or end <= start:
                return []
            data = json.loads(text[start:end])
            day_data = data.get("data", {}).get(index_code, {}).get("day", [])
            records = []
            for item in day_data:
                if len(item) < 6:
                    continue
                try:
                    records.append({
                        "trade_date": item[0],
                        "open": float(item[1]),
                        "close": float(item[2]),
                        "high": float(item[3]),
                        "low": float(item[4]),
                        "volume": float(item[5]),
                        "amount": 0,
                    })
                except (ValueError, IndexError):
                    continue
            return records
        except Exception as e:
            logger.warning(f"指数K线获取失败 {index_code}: {e}")
            return []

    @staticmethod
    def get_index_pct(index_code: str = "sh000001") -> float:
        """获取指数实时涨跌幅，默认上证指数。"""
        try:
            url = f"https://qt.gtimg.cn/q={index_code}"
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.encoding = 'utf-8'
            match = re.search(r'="([^"]+)"', resp.text)
            if not match:
                return 0.0
            fields = match.group(1).split('~')
            if len(fields) > 32:
                return _to_float(fields[32])
        except Exception as e:
            logger.warning(f"指数行情获取失败 {index_code}: {e}")
        return 0.0

    @staticmethod
    def get_north_money() -> list[dict]:
        """北向资金净买入 Top10。"""
        try:
            import akshare as ak
            df = ak.stock_hsgt_hold_stock_em(market="北向", indicator="今日排行")
            if len(df) == 0:
                return []
            net_col = next((c for c in df.columns if "增持估计-市值" in str(c)), "")
            if net_col:
                df = df.sort_values(net_col, ascending=False)
            top = df.head(10)
            return [
                {"code": _normalize_code(row.get("代码", "")),
                 "name": str(row.get("名称", "")),
                 "net_buy": _to_float(row.get(net_col, 0) if net_col else 0)}
                for _, row in top.iterrows()
                if _normalize_code(row.get("代码", ""))
            ]
        except Exception as e:
            logger.warning(f"北向资金获取失败: {e}")
            return []
