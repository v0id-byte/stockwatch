"""市场数据获取（腾讯财经接口）"""
import re, json
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
        # 腾讯接口：sh + 6位数代码（ETF/股票均用 sh/sz 前缀）
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
                raw_code = m.group(1)  # e.g. v_sh600519
                if len(raw_code) < 4:
                    continue
                code = raw_code[2:]  # strip sh/sz
                fields = m.group(2).split('~')
                if len(fields) < 40:
                    continue
                try:
                    result[code] = {
                        "name": fields[1].strip(),
                        "open": float(fields[5]) if fields[5] else 0,
                        "high": float(fields[33]) if fields[33] else 0,
                        "low": float(fields[34]) if fields[34] else 0,
                        "close": float(fields[3]),
                        "volume": float(fields[37]) if fields[37] else 0,
                        "pct_change": float(fields[32]) if fields[32] else 0,
                    }
                except (ValueError, IndexError):
                    continue
            return result
        except Exception as e:
            logger.warning(f"实时报价失败: {e}")
            return {}

    @staticmethod
    def get_daily_kline(code: str, period: str = "daily", adjust: str = "qfq", limit: int = 320) -> list[dict]:
        """腾讯财经日线前复权"""
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
            logger.warning(f"K线获取失败 {code}: {e}")
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
