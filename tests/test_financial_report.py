"""Tests for the 财报解析 module and the related parser / source-link changes."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")


def _bot_module():
    """Lazy import so monkeypatching akshare / llm does not leak between tests."""
    import bot.financial_report as fr
    return fr


class TestParseUserReply:
    def test_numeric_sequence(self):
        fr = _bot_module()
        result = fr.parse_user_reply("1 2 3 4")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_letter_sequence(self):
        fr = _bot_module()
        result = fr.parse_user_reply("a b c d")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_letter_uppercase(self):
        fr = _bot_module()
        result = fr.parse_user_reply("A B C D")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_comma_separator(self):
        fr = _bot_module()
        result = fr.parse_user_reply("1,2,3,4")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_chinese_full_width(self):
        fr = _bot_module()
        result = fr.parse_user_reply("1，2，3，4")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_named_prefix(self):
        fr = _bot_module()
        result = fr.parse_user_reply("内容1 角度2 形式3 建议4")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_partial_uses_default_later(self):
        fr = _bot_module()
        result = fr.parse_user_reply("1 2")
        assert result == {"content": 1, "focus": 2}
        # resolve_choices fills in defaults for missing keys
        resolved = fr.resolve_choices(result)
        for dim in fr.DIMENSIONS:
            assert dim.key in resolved

    def test_default_keyword_returns_empty_dict(self):
        fr = _bot_module()
        assert fr.parse_user_reply("默认") == {}
        assert fr.parse_user_reply("default") == {}

    def test_unparseable_returns_none(self):
        fr = _bot_module()
        assert fr.parse_user_reply("600519 最近怎么样") is None
        assert fr.parse_user_reply("帮我看看") is None
        assert fr.parse_user_reply("") is None

    def test_out_of_range_rejects_whole_reply(self):
        fr = _bot_module()
        # Any out-of-range token should reject the whole reply so the user
        # is forced to fix it (cleaner UX than silently shifting positions).
        assert fr.parse_user_reply("99 2 3 4") is None
        assert fr.parse_user_reply("1 2 3 99") is None
        # Letter 'z' = 26, content only has 7 options
        assert fr.parse_user_reply("z 2 3 4") is None

    def test_mixed_named_and_positional(self):
        """`内容1 2 3 4` should fill content=1 (named), then focus=2, format=3,
        suggest=4 (positional fill-in) — not silently drop the tail."""
        fr = _bot_module()
        result = fr.parse_user_reply("内容1 2 3 4")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_named_only_overrides_specific_dims(self):
        fr = _bot_module()
        # 只有两个命名，其余维度保持空 → 调用方用默认
        result = fr.parse_user_reply("内容6 建议3")
        assert result == {"content": 6, "suggest": 3}

    def test_whitespace_only(self):
        fr = _bot_module()
        assert fr.parse_user_reply("   ") is None
        assert fr.parse_user_reply("\u3000\t\n") is None

    def test_unknown_token_silently_skipped(self):
        """无法解析为数字/字母/命名前缀的 token 静默跳过，不影响其他有效选择。

        整条文本「所有 token 都解析不出」时才返回 None（见
        ``test_unparseable_returns_none``）。
        """
        fr = _bot_module()
        # "质量1" 不是已知前缀也不是单个数字/字母 → 跳过
        assert fr.parse_user_reply("质量1 2 3 4") == {"content": 2, "focus": 3, "format": 4}

    def test_letter_is_choice_value_not_dim_alias(self):
        """单字母 c/f/p/s 永远解释为选项序号（c=3, s=19, ...），不是维度别名。

        维度前缀仅支持中文：内容/财报/角度/看/形式/呈现/建议。
        """
        fr = _bot_module()
        # c=3 → content=3, "2" 走 focus=2, s=19 (suggest 越界) → 整条拒绝
        assert fr.parse_user_reply("c 2 s 4") is None
        # 但只用 a-d 字母就都有效：a=1, b=2, c=3, d=4
        assert fr.parse_user_reply("a b c d") == {"content": 1, "focus": 2, "format": 3, "suggest": 4}

    def test_full_width_digits(self):
        """NFKC 应该把 ０-９ 归一化成 0-9。"""
        fr = _bot_module()
        result = fr.parse_user_reply("１ ２ ３ ４")
        assert result == {"content": 1, "focus": 2, "format": 3, "suggest": 4}


class TestResolveChoices:
    def test_resolve_with_none_uses_all_defaults(self):
        fr = _bot_module()
        resolved = fr.resolve_choices(None)
        for dim, idx in resolved.values():
            assert idx == dim.default

    def test_resolve_with_partial_uses_defaults_for_missing(self):
        fr = _bot_module()
        resolved = fr.resolve_choices({"content": 6})
        assert resolved["content"][1] == 6
        for key, (dim, idx) in resolved.items():
            if key != "content":
                assert idx == dim.default

    def test_resolve_clamps_out_of_range(self):
        fr = _bot_module()
        resolved = fr.resolve_choices({"content": 9999})
        assert resolved["content"][1] == resolved["content"][0].default


class TestOptionMenu:
    def test_menu_contains_all_dimensions(self):
        fr = _bot_module()
        menu = fr.render_option_menu()
        for dim in fr.DIMENSIONS:
            assert f"【{dim.title}】" in menu
        for dim in fr.DIMENSIONS:
            for label, _hint in dim.options:
                assert label in menu
        assert "默认" in menu

    def test_menu_count_matches_dim_options(self):
        fr = _bot_module()
        menu = fr.render_option_menu()
        for dim in fr.DIMENSIONS:
            for i in range(1, len(dim.options) + 1):
                # 形如 "  1. 业绩快报..."
                assert f"{i}. " in menu


class TestFormatSourcesURL:
    def test_announcement_url_is_rendered(self):
        from bot.research import _format_sources
        items = [
            {
                "label": "公告1",
                "date": "2026-04-17",
                "source": "巨潮资讯公告",
                "title": "贵州茅台2025年年度报告",
                "url": "http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600519&announcementId=1225114741",
            },
            {
                "label": "公告2",
                "date": "2026-04-17",
                "source": "巨潮资讯公告",
                "title": "贵州茅台2025年年度报告（英文版）",
                # no URL: should be skipped
            },
        ]
        out = _format_sources(items, "公告")
        assert "详情：http://www.cninfo.com.cn/" in out
        # second item has no URL: no 详情 line after its title
        lines = out.splitlines()
        second_block_start = next(i for i, line in enumerate(lines) if "公告2" in line)
        assert not any("详情：" in line for line in lines[second_block_start:])

    def test_https_url_kept(self):
        from bot.research import _format_sources
        items = [{"label": "研报1", "date": "2026-01-01", "source": "东方财富研报", "title": "Test", "url": "https://example.com/r"}]
        out = _format_sources(items, "研报")
        assert "详情：https://example.com/r" in out

    def test_javascript_url_ignored(self):
        from bot.research import _format_sources
        items = [{"label": "公告1", "date": "2026-01-01", "source": "X", "title": "T", "url": "javascript:alert(1)"}]
        out = _format_sources(items, "公告")
        assert "javascript:" not in out

    def test_empty_items(self):
        from bot.research import _format_sources
        assert _format_sources([], "公告") == "公告: 暂无"


class TestFinancialReportFallback:
    """When LLM is unavailable, _format_report_fallback should still produce a
    readable summary that includes URLs and the disclaimer."""

    def _make_report(self):
        from bot.financial_report import ReportData
        from bot.research import StockRef

        report = ReportData(stock=StockRef(code="600519", name="贵州茅台"))
        report.announcements = [
            {
                "type": "年报",
                "title": "贵州茅台2025年年度报告",
                "date": "2026-04-17",
                "url": "https://www.cninfo.com.cn/x.pdf",
            },
        ]
        report.balance_sheet = [{
            "period": "20251231",
            "items": {
                "资产-总资产": 2_500_000_000_000.0,
                "资产-总资产同比": 12.5,
                "负债-总负债": 200_000_000_000.0,
                "负债-总负债同比": -3.0,
                "资产负债率": 8.0,
                "资产-货币资金": 100_000_000_000.0,
                "资产-存货": 30_000_000_000.0,
            },
            "ann_date": "2026-04-17",
        }]
        report.income_statement = [{
            "period": "20251231",
            "items": {
                "营业总收入": 150_000_000_000.0,
                "营业总收入同比": 15.0,
                "净利润": 60_000_000_000.0,
                "净利润同比": 14.0,
                "营业利润": 80_000_000_000.0,
                "营业总支出-销售费用": 5_000_000_000.0,
                "营业总支出-管理费用": 8_000_000_000.0,
                "营业总支出-财务费用": -1_000_000_000.0,
            },
            "ann_date": "2026-04-17",
        }]
        report.key_indicators = [
            {"报告期": "2025年报", "每股收益": 47.46, "净资产收益率": 35.0, "销售毛利率": 91.0, "资产负债率": 8.0, "营收同比": 15.0, "归母净利同比": 14.0},
            {"报告期": "2024年报", "每股收益": 41.21, "净资产收益率": 36.5, "销售毛利率": 92.0, "资产负债率": 9.0, "营收同比": 13.0, "归母净利同比": 13.0},
        ]
        report.warnings = ["测试告警字段"]
        return report

    def test_fallback_has_disclaimer_and_url(self):
        from bot.financial_report import _format_report_fallback
        report = self._make_report()
        text = _format_report_fallback(report)
        assert "免责声明" in text
        assert "https://www.cninfo.com.cn/x.pdf" in text
        assert "贵州茅台" in text
        # 资产负债表关键字段
        assert "总资产" in text
        assert "营业利润" in text
        # 报告期文字
        assert "2025年报" in text

    def test_fallback_when_empty(self):
        from bot.financial_report import _RISK_DISCLAIMER
        from bot.financial_report import _format_report_fallback
        from bot.research import StockRef
        from bot.financial_report import ReportData

        empty = ReportData(stock=StockRef(code="999999", name="未取到"))
        text = _format_report_fallback(empty)
        # 当所有数据为空时仍应包含免责声明和股票名
        assert "未取到" in text
        assert "免责声明" in text


class TestParserFinancialReport:
    def test_financial_report_with_code(self):
        from bot.parser import parse_command
        cmd = parse_command("财报解析 600519")
        assert cmd.action == "financial_report_menu"
        assert cmd.code == "600519"

    def test_financial_report_without_code(self):
        from bot.parser import parse_command
        cmd = parse_command("财报")
        assert cmd.action == "financial_report_menu"
        assert cmd.code == ""

    def test_financial_report_aliases(self):
        from bot.parser import parse_command
        for text in ("财务报告 000001", "年报解析 000001", "看财报 000001"):
            cmd = parse_command(text)
            assert cmd.action == "financial_report_menu", text
            assert cmd.code == "000001", text

    def test_cancel_pending_flow(self):
        from bot.parser import parse_command
        for text in ("取消财报", "退出财报", "重置菜单", "取消菜单"):
            assert parse_command(text).action == "cancel_pending_flow", text

    def test_choice_reply_does_not_trigger_financial_menu(self):
        """A bare '1 2 3 4' should not be re-parsed as a financial-report trigger
        — the multi-turn flow is owned by the runner, not the parser."""
        from bot.parser import parse_command
        cmd = parse_command("1 2 3 4")
        assert cmd.action != "financial_report_menu"

    def test_help_lines_includes_financial_report(self):
        from bot.parser import help_lines
        text = "\n".join(help_lines())
        assert "财报解析" in text


class TestRunnerFlowHelpers:
    def test_set_peek_pop_flow(self):
        import bot.runner as runner
        from bot.research import StockRef
        runner._set_financial_flow("u1", "c1", StockRef(code="600519", name="贵州茅台"))
        flow = runner._claim_financial_flow("u1", "c1")
        assert flow is not None
        assert flow.code == "600519"
        # 已 pop，再 claim 应该 None
        assert runner._claim_financial_flow("u1", "c1") is None

    def test_ttl_expiry(self):
        import bot.runner as runner
        from bot.research import StockRef
        runner._set_financial_flow("u2", "c2", StockRef(code="000001", name="X"))
        key = "u2:c2"
        assert key in runner._FINANCIAL_FLOW_STATE
        # 手动把 created_at 拉到过期
        runner._FINANCIAL_FLOW_STATE[key].created_at -= 99999
        # claim 应该返回 None（因为已过期），并把 key 清掉
        assert runner._claim_financial_flow("u2", "c2") is None
        assert key not in runner._FINANCIAL_FLOW_STATE

    def test_atomic_claim_under_concurrency(self):
        """两个线程同时 claim 同一 user 的 pending flow，应当只拿到一个。"""
        import bot.runner as runner
        import threading
        from bot.research import StockRef

        runner._set_financial_flow("u3", "c3", StockRef(code="600519", name="X"))
        results: list = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()
            flow = runner._claim_financial_flow("u3", "c3")
            results.append(flow)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start(); t2.start()
        t1.join(); t2.join()

        wins = [r for r in results if r is not None]
        losses = [r for r in results if r is None]
        assert len(wins) == 1
        assert len(losses) == 1
        assert wins[0].code == "600519"

    def test_discard_flow(self):
        import bot.runner as runner
        from bot.research import StockRef
        runner._set_financial_flow("u4", "c4", StockRef(code="000001", name="X"))
        runner._discard_financial_flow("u4", "c4")
        assert runner._claim_financial_flow("u4", "c4") is None


class TestFilterDfByCode:
    def test_none_input(self):
        fr = _bot_module()
        assert fr._filter_df_by_code(None, "600519") is None

    def test_non_dataframe_input(self):
        fr = _bot_module()
        assert fr._filter_df_by_code({"foo": "bar"}, "600519") is None
        assert fr._filter_df_by_code([], "600519") is None
        assert fr._filter_df_by_code("not a df", "600519") is None

    def test_missing_code_column_returns_none(self):
        """Bug guard: previously returned df.head(0) which crashed callers
        that did ``row_df.iloc[0]``."""
        import pandas as pd
        fr = _bot_module()
        df = pd.DataFrame({"name": ["贵州茅台"], "sector": ["白酒"]})
        assert fr._filter_df_by_code(df, "600519") is None

    def test_empty_dataframe(self):
        import pandas as pd
        fr = _bot_module()
        df = pd.DataFrame({"股票代码": [], "净利润": []})
        assert fr._filter_df_by_code(df, "600519") is None

    def test_matches_code(self):
        import pandas as pd
        fr = _bot_module()
        df = pd.DataFrame({
            "股票代码": ["600519", "000001"],
            "净利润": [600e8, 100e8],
        })
        result = fr._filter_df_by_code(df, "600519")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["净利润"] == 600e8


class TestLLMFailureFallback:
    def test_answer_financial_report_falls_back_on_llm_error(self, monkeypatch):
        import bot.financial_report as fr
        from bot.research import StockRef
        from bot.financial_report import ReportData

        report = ReportData(stock=StockRef(code="600519", name="贵州茅台"))
        report.announcements = [{"type": "年报", "title": "T", "date": "2026-04-17", "url": "https://x"}]
        monkeypatch.setattr(fr, "fetch_report_data", lambda stock: report)

        class _BoomClient:
            def chat(self, *args, **kwargs):
                raise RuntimeError("LLM down")
            @staticmethod
            def _strip_think(t):
                return t

        monkeypatch.setattr(fr, "get_llm_client", lambda: _BoomClient())

        text, returned_report, choices = fr.answer_financial_report(StockRef(code="600519", name="贵州茅台"))
        assert "免责声明" in text
        assert "https://x" in text
        assert returned_report is report
        # 选择项应当被 resolve（默认 + 默认 + 默认 + 默认）
        assert all(idx == dim.default for dim, idx in choices.values())