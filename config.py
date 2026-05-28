"""配置加载 + 环境校验"""
import os
import json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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

    @property
    def feishu_receive_id_2(self) -> str:
        return os.getenv("FEISHU_RECEIVE_ID_2", "")

    @property
    def feishu_verification_token(self) -> str:
        return os.getenv("FEISHU_VERIFICATION_TOKEN", "")

    @property
    def feishu_encrypt_key(self) -> str:
        return os.getenv("FEISHU_ENCRYPT_KEY", "")

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

    # === v2 feature flags（默认关闭） ===
    @property
    def enable_calibration(self) -> bool:
        return _env_bool("ENABLE_CALIBRATION", False)

    @property
    def calibration_lookback_days(self) -> int:
        return int(os.getenv("CALIBRATION_LOOKBACK_DAYS", "5"))

    @property
    def calibration_min_samples(self) -> int:
        return int(os.getenv("CALIBRATION_MIN_SAMPLES", "50"))

    @property
    def enable_alpha158(self) -> bool:
        return _env_bool("ENABLE_ALPHA158", False)

    @property
    def enable_lgbm(self) -> bool:
        return _env_bool("ENABLE_LGBM", False)

    @property
    def lgbm_model_path(self) -> Path:
        raw = os.getenv("LGBM_MODEL_PATH", "~/.stockwatch/models/lgbm.txt")
        return Path(os.path.expanduser(raw))

    @property
    def enable_regime(self) -> bool:
        return _env_bool("ENABLE_REGIME", False)

    @property
    def enable_sector(self) -> bool:
        return _env_bool("ENABLE_SECTOR", False)

    @property
    def any_v2_enabled(self) -> bool:
        return any([
            self.enable_calibration,
            self.enable_alpha158,
            self.enable_lgbm,
            self.enable_regime,
            self.enable_sector,
        ])

    @property
    def any_v2_context_enabled(self) -> bool:
        return any([
            self.enable_alpha158,
            self.enable_lgbm,
            self.enable_regime,
            self.enable_sector,
        ])

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
