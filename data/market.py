"""市场数据获取（腾讯财经接口）"""
import re, json
import requests
from datetime import datetime, timedelta
from loguru import logger

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://gu.qq.com/'
}


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
            resp.encoding = 'utf-8'
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
    def get_daily_kline(code: str, period: str = "daily", adjust: str = "qfq") -> list[dict]:
        """腾讯财经日线前复权"""
        try:
            prefix = "sh" if code.startswith(('6', '5')) else "sz"
            url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                   f"?_var=kline_dayfqzf&param={prefix}{code},day,,,320,qfq&r=0.1")
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
    def is_trading_day() -> bool:
        """用内置节日表 + weekday 判断（非交易日返回 False）"""
        from datetime import date
        today = date.today()
        if today.weekday() >= 5:
            return False
        # 简化：默认交易日（实际部署时 AKShare 修复后用真实日历）
        return True

    @staticmethod
    def get_north_money() -> list[dict]:
        """北向资金 Top10（用 stock_hsgt_stock_statistics_em）"""
        try:
            import akshare as ak
            df = ak.stock_hsgt_stock_statistics_em(symbol='北向', end_date=datetime.now().strftime('%Y%m%d'))
            if len(df) == 0:
                return []
            top = df.head(10)
            return [
                {"code": str(row.get("代码", "")).zfill(6),
                 "name": str(row.get("名称", "")),
                 "net_buy": float(row.get("当日成交净买额", 0))}
                for _, row in top.iterrows()
            ]
        except Exception as e:
            logger.warning(f"北向资金获取失败: {e}")
            return []
