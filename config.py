"""配置加载 + 环境校验"""
import os
import json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


class Config:
    """单例配置，所有配置项从 .env 读取"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, env_path: str | None = None) -> "Config":
        if self._loaded:
            return self
        if env_path:
            load_dotenv(env_path)
        else:
            load_dotenv()
        self._loaded = True
        self._validate()
        return self

    def _validate(self):
        required = [
            "MINIMAX_API_KEY",
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_RECEIVE_ID",
        ]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"缺少必填配置项: {', '.join(missing)}")

    # === LLM ===
    @property
    def minimax_api_key(self) -> str:
        return os.getenv("MINIMAX_API_KEY", "")

    @property
    def minimax_base_url(self) -> str:
        return os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1").rstrip("/")

    @property
    def minimax_model(self) -> str:
        return os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

    # === 飞书 ===
    @property
    def feishu_app_id(self) -> str:
        return os.getenv("FEISHU_APP_ID", "")

    @property
    def feishu_app_secret(self) -> str:
        return os.getenv("FEISHU_APP_SECRET", "")

    @property
    def feishu_receive_id(self) -> str:
        return os.getenv("FEISHU_RECEIVE_ID", "")

    @property
    def feishu_receive_id_type(self) -> str:
        return os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id")

    # === 数据源 ===
    @property
    def tushare_token(self) -> str:
        return os.getenv("TUSHARE_TOKEN", "")

    # === 业务参数 ===
    @property
    def watchlist(self) -> list[str]:
        raw = os.getenv("WATCHLIST", "600519,000858,510300,510500,159915")
        return [s.strip() for s in raw.split(",") if s.strip()]

    @property
    def max_stocks_per_run(self) -> int:
        return int(os.getenv("MAX_STOCKS_PER_RUN", "50"))

    @property
    def min_confidence_to_push(self) -> float:
        return float(os.getenv("MIN_CONFIDENCE_TO_PUSH", "0.6"))

    @property
    def stop_loss_fallback_pct(self) -> float:
        return float(os.getenv("STOP_LOSS_FALLBACK_PCT", "0.07"))

    # === 运行 ===
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    # === 路径 ===
    @property
    def home_dir(self) -> Path:
        p = Path.home() / ".stockwatch"
        p.mkdir(exist_ok=True)
        return p

    @property
    def db_path(self) -> Path:
        return self.home_dir / "db.sqlite"

    @property
    def log_dir(self) -> Path:
        p = self.home_dir / "logs"
        p.mkdir(exist_ok=True)
        return p

    @property
    def feishu_token_cache_path(self) -> Path:
        return self.home_dir / ".feishu_token.json"

    def __repr__(self):
        return "<Config loaded=True>"


def get_config() -> Config:
    cfg = Config()
    cfg.load()
    return cfg