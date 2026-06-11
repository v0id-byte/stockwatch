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
        if _env_bool("STOCKWATCH_SKIP_REQUIRED_CONFIG", False):
            return
        required = [
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_RECEIVE_ID",
        ]
        if not self.llm_api_key and not self.llm_allows_empty_key:
            required.append("LLM_API_KEY")
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"缺少必填配置项: {', '.join(missing)}")

    # === LLM ===
    @property
    def llm_provider(self) -> str:
        raw = os.getenv("LLM_PROVIDER", "").strip().lower()
        if raw:
            return raw
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "openai"

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "anthropic":
            return os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
        return os.getenv("LLM_API_KEY") or os.getenv("MINIMAX_API_KEY", "")

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == "anthropic":
            raw = (
                os.getenv("LLM_BASE_URL")
                or os.getenv("ANTHROPIC_BASE_URL")
                or "https://api.anthropic.com"
            )
        else:
            raw = (
                os.getenv("LLM_BASE_URL")
                or os.getenv("MINIMAX_BASE_URL")
                or "https://api.minimaxi.com/v1"
            )
        return raw.rstrip("/")

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "anthropic":
            return (
                os.getenv("LLM_MODEL")
                or os.getenv("ANTHROPIC_MODEL")
                or "claude-3-5-sonnet-latest"
            )
        return os.getenv("LLM_MODEL") or os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

    @property
    def llm_allows_empty_key(self) -> bool:
        host = self.llm_base_url.lower()
        local_hosts = ("http://localhost", "http://127.0.0.1", "http://0.0.0.0")
        openai_like = {"openai", "openai-compatible", "custom", "local", "minimax"}
        return self.llm_provider in openai_like and host.startswith(local_hosts)

    @property
    def llm_api_key_or_placeholder(self) -> str:
        return self.llm_api_key or "not-needed"

    @property
    def minimax_api_key(self) -> str:
        return self.llm_api_key

    @property
    def minimax_base_url(self) -> str:
        return self.llm_base_url

    @property
    def minimax_model(self) -> str:
        return self.llm_model

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

    @property
    def enable_reassurance_mode(self) -> bool:
        return _env_bool("ENABLE_REASSURANCE_MODE", False)

    @property
    def enable_family_brief(self) -> bool:
        return _env_bool("ENABLE_FAMILY_BRIEF", False)

    @property
    def enable_after_close_summary(self) -> bool:
        return _env_bool("ENABLE_AFTER_CLOSE_SUMMARY", self.enable_reassurance_mode)

    @property
    def alert_levels(self) -> set[str]:
        raw = os.getenv("ALERT_LEVELS", "critical,warning,info").strip().lower()
        if raw in {"", "all"}:
            return {"critical", "warning", "info"}
        aliases = {
            "red": "critical",
            "orange": "warning",
            "yellow": "warning",
            "blue": "info",
            "green": "info",
        }
        levels = set()
        for item in raw.split(","):
            value = aliases.get(item.strip(), item.strip())
            if value in {"critical", "warning", "info"}:
                levels.add(value)
        return levels

    def alert_level_enabled(self, level: str) -> bool:
        return str(level or "info").lower() in self.alert_levels

    @property
    def ai_response_style(self) -> str:
        value = os.getenv("AI_RESPONSE_STYLE", "balanced").strip().lower()
        if value in {"concise", "balanced", "detailed", "expert"}:
            return value
        return "balanced"

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
