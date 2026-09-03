"""配置加载 + 环境校验"""
import os
import json
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from core.settings import settings_path, stockwatch_home


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
        configured_env_path = Path(env_path).expanduser() if env_path else settings_path(Path(__file__).resolve().parent)
        if configured_env_path.exists():
            load_dotenv(configured_env_path, override=True)
        self._loaded = True
        self._validate()
        return self

    def _validate(self):
        if _env_bool("STOCKWATCH_SKIP_REQUIRED_CONFIG", False):
            return
        required = []
        if self.notify_channel == "feishu":
            required.extend([
                "FEISHU_APP_ID",
                "FEISHU_APP_SECRET",
                "FEISHU_RECEIVE_ID",
            ])
        # 只有当用户「显式」要求开启 AI 却没给凭证时才报错。
        # 默认 ENABLE_AI=auto：没配 LLM 就自动进入规则模式（止损/盯价/公告照常跑），
        # 不再因为缺 LLM_API_KEY 而启动失败——这是家用「装一次、只看网页」的关键。
        if self.ai_enabled and not self.llm_api_key and not self.llm_allows_empty_key:
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
    def ai_enabled(self) -> bool:
        """是否启用 AI（LLM）分析。
        ENABLE_AI = on/off/auto（默认 auto）。
        - on：强制启用（缺凭证会在 _validate 报错，提醒用户去配）。
        - off / none / rule：规则模式——完全不调用 LLM，只跑止损/盯价/公告/持仓提醒。
        - auto：配了 LLM_API_KEY，或 base_url 指向本地服务（Ollama 等），才算启用；
                否则自动降级为规则模式。家用「装一次只看网页」默认走这里。"""
        raw = os.getenv("ENABLE_AI", "auto").strip().lower()
        if raw in {"1", "true", "yes", "y", "on"}:
            return True
        if raw in {"0", "false", "no", "n", "off", "none", "rule"}:
            return False
        return bool(self.llm_api_key) or self.llm_allows_empty_key

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
    def feishu_receive_ids(self) -> list[str]:
        """多用户推送：FEISHU_RECEIVE_ID 可填多个接收人，用逗号/分号/空格分隔，
        例如 FEISHU_RECEIVE_ID=ou_aaa,ou_bbb,ou_ccc。所有接收人共用同一个
        FEISHU_RECEIVE_ID_TYPE（默认 open_id），所以要么都是 open_id、要么都是
        user_id/email。旧的 FEISHU_RECEIVE_ID_2 仍然兼容，会自动并入列表。"""
        import re

        ids = [x for x in re.split(r"[,;\s]+", os.getenv("FEISHU_RECEIVE_ID", "")) if x]
        second = os.getenv("FEISHU_RECEIVE_ID_2", "").strip()
        if second:
            ids.append(second)
        seen, out = set(), []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    @property
    def feishu_verification_token(self) -> str:
        return os.getenv("FEISHU_VERIFICATION_TOKEN", "")

    @property
    def feishu_encrypt_key(self) -> str:
        return os.getenv("FEISHU_ENCRYPT_KEY", "")

    @property
    def notify_channel(self) -> str:
        # 默认 web：与 .env.example / README 的「先用 Web 控制台、无需配飞书」一致，
        # 避免缺失该项时默认走飞书、又因缺飞书凭证而启动校验失败。
        value = os.getenv("NOTIFY_CHANNEL", "web").strip().lower()
        if value in {"feishu", "web"}:
            return value
        return "web"

    @property
    def agent_enable_bot(self) -> bool:
        return _env_bool("STOCKWATCH_AGENT_ENABLE_BOT", False)

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
    def must_see_mode(self) -> bool:
        """必看模式：一键把提醒收紧到"宁可漏、不要烦"。
        打开后只推 critical（止损/强风险/重大负面），并抬高置信度门槛。
        相当于同时设 ALERT_LEVELS=critical + 提高 MIN_CONFIDENCE_TO_PUSH，
        但用户只需拨一个开关。"""
        return _env_bool("MUST_SEE_MODE", False)

    MUST_SEE_CONFIDENCE_FLOOR = 0.72

    @property
    def min_confidence_to_push(self) -> float:
        base = float(os.getenv("MIN_CONFIDENCE_TO_PUSH", "0.6"))
        return max(base, self.MUST_SEE_CONFIDENCE_FLOOR) if self.must_see_mode else base

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
        if self.must_see_mode:
            return {"critical"}
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
        raw = os.getenv("LGBM_MODEL_PATH", "").strip()
        return Path(os.path.expanduser(raw)) if raw else self.home_dir / "models" / "lgbm.txt"

    @property
    def enable_risk_model(self) -> bool:
        return _env_bool("ENABLE_RISK_MODEL", False)

    @property
    def risk_model_path(self) -> Path:
        raw = os.getenv("RISK_MODEL_PATH", "").strip()
        return Path(os.path.expanduser(raw)) if raw else self.home_dir / "models" / "lgbm_v2_risk.txt"

    @property
    def model_scoring_universe(self) -> str:
        """Reference universe for nightly batch scoring; must match training."""
        return os.getenv("MODEL_SCORING_UNIVERSE", "csi500").strip().lower()

    @property
    def market_regime(self) -> str:
        """auto = 按 CSI300 站上/跌破年线自动判定；bull/bear = 用户手动锁定。"""
        value = os.getenv("MARKET_REGIME", "auto").strip().lower()
        return value if value in {"auto", "bull", "bear"} else "auto"

    def resolve_lgbm_model_path(self, regime: str) -> Path:
        """Bear regime uses the bear-specialized model (validated OOS lift); every
        other regime uses the universal model. Falls back to universal if the
        bear model file is missing."""
        base = self.lgbm_model_path
        if regime == "bear":
            bear = base.with_name(base.stem + "_bear" + base.suffix)
            if bear.exists():
                return bear
        return base

    @property
    def enable_regime(self) -> bool:
        return _env_bool("ENABLE_REGIME", False)

    @property
    def enable_sector(self) -> bool:
        return _env_bool("ENABLE_SECTOR", False)

    @property
    def enable_events(self) -> bool:
        # 结构化事件提醒（解禁/业绩预告/增减持/回购）作为风险/情境上下文，非涨跌预测
        return _env_bool("ENABLE_EVENTS", False)

    @property
    def enable_propagation(self) -> bool:
        return _env_bool("ENABLE_PROPAGATION", False)

    @property
    def propagation_max_candidates(self) -> int:
        return int(os.getenv("PROPAGATION_MAX_CANDIDATES", "10"))

    @property
    def propagation_leader_return_threshold(self) -> float:
        return float(os.getenv("PROPAGATION_LEADER_RETURN_THRESHOLD", "0.04"))

    @property
    def propagation_min_corr(self) -> float:
        return float(os.getenv("PROPAGATION_MIN_CORR", "0.15"))

    @property
    def propagation_history_dir(self) -> Path:
        raw = os.getenv("PROPAGATION_HISTORY_DIR", "").strip()
        return Path(os.path.expanduser(raw)) if raw else self.home_dir / "history" / "stocks"

    @property
    def any_v2_enabled(self) -> bool:
        return any([
            self.enable_calibration,
            self.enable_alpha158,
            self.enable_lgbm,
            self.enable_regime,
            self.enable_sector,
            self.enable_propagation,
        ])

    @property
    def any_v2_context_enabled(self) -> bool:
        return any([
            self.enable_alpha158,
            self.enable_lgbm,
            self.enable_regime,
            self.enable_sector,
            self.enable_propagation,
        ])

    # === 运行 ===
    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    # === 路径 ===
    @property
    def home_dir(self) -> Path:
        return stockwatch_home()

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
