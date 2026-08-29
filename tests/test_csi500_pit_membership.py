import json

import pandas as pd
import pytest

from scripts.build_csi500_pit_membership import (
    PITSourceError,
    RebalanceDelta,
    _daily_membership,
    _explicit_membership_grid,
    _parse_html_delta,
    _parse_pdf_delta_text,
    _parse_xlsx_delta,
    _reconstruct_versions,
)
from scripts.capture_csi500_official import _conservative_available_at


def _delta(notice_id, effective, removed, added):
    return RebalanceDelta(
        notice_id=notice_id,
        published_at="2024-05-31",
        available_at="2024-06-01T00:00:00+08:00",
        effective_after_close=pd.Timestamp(effective),
        removed=tuple(removed),
        added=tuple(added),
        source_url=f"https://official/{notice_id}",
        source_hash=(str(notice_id) * 64)[:64],
    )


def test_date_only_announcement_is_not_available_same_day():
    assert _conservative_available_at("2024-05-31") == "2024-06-01T00:00:00+08:00"


def test_parse_html_delta_uses_only_csi500_table():
    html = """
    <p>沪深300指数样本调整名单：</p>
    <table><tr><td>000001</td><td>A</td><td>000002</td><td>B</td></tr></table>
    <p>中证500指数样本调整名单：</p>
    <table>
      <tr><td>调出名单</td><td>调入名单</td></tr>
      <tr><td>证券代码</td><td>证券名称</td><td>证券代码</td><td>证券名称</td></tr>
      <tr><td>000031</td><td>大悦城</td><td>000034</td><td>神州数码</td></tr>
      <tr><td>600066</td><td>宇通客车</td><td>600160</td><td>巨化股份</td></tr>
    </table>
    """

    removed, added = _parse_html_delta(html)

    assert removed == ["000031", "600066"]
    assert added == ["000034", "600160"]


def test_parse_pdf_text_stops_at_next_index_section():
    text = """
    中证 500 指数样本调整名单：
                调出名单                     调入名单
      证券代码 证券名称            证券代码 证券名称
       000012 南玻 A              000539 粤电力 A
       600171 上海贝岭             688120 华海清科
    中证 1000 指数样本调整名单：
       000012 不应读取              000539 不应读取
    """

    removed, added = _parse_pdf_delta_text(text)

    assert removed == ["000012", "600171"]
    assert added == ["000539", "688120"]


def test_parse_official_xlsx_filters_index_code(tmp_path):
    path = tmp_path / "adjustments.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({
            "指数代码": ["000300", "000905"],
            "指数简称": ["沪深300", "中证500"],
            "证券代码": [1, 31],
            "证券简称": ["A", "大悦城"],
        }).to_excel(writer, sheet_name="调出", index=False)
        pd.DataFrame({
            "指数代码": ["000300", "000905"],
            "指数简称": ["沪深300", "中证500"],
            "证券代码": [2, 34],
            "证券简称": ["B", "神州数码"],
        }).to_excel(writer, sheet_name="调入", index=False)

    removed, added = _parse_xlsx_delta(path)

    assert removed == ["000031"]
    assert added == ["000034"]


def test_reverse_delta_chain_builds_exact_daily_membership():
    first = _delta(1, "2024-06-14", ["000001"], ["000004"])
    second = _delta(2, "2024-12-13", ["000002"], ["000005"])
    anchor = {"000003", "000004", "000005"}

    versions = _reconstruct_versions(anchor, [first, second], expected_members=3)
    calendar = pd.DatetimeIndex([
        "2024-06-14", "2024-06-17", "2024-12-13", "2024-12-16", "2024-12-17"
    ])
    daily = _daily_membership(
        calendar,
        versions,
        pd.Timestamp("2024-12-17"),
        "a" * 64,
        expected_members=3,
    )

    june = set(daily.loc[daily["trade_date"] == pd.Timestamp("2024-06-17"), "code"])
    december = set(daily.loc[daily["trade_date"] == pd.Timestamp("2024-12-16"), "code"])
    assert june == {"000002", "000003", "000004"}
    assert december == anchor
    assert daily.groupby("trade_date")["code"].nunique().eq(3).all()
    assert set(daily["index_code"]) == {"000905"}
    assert daily["is_member"].all()


def test_reverse_chain_fails_closed_when_an_adjustment_is_missing():
    impossible = _delta(7, "2024-06-14", ["000001"], ["000099"])

    with pytest.raises(PITSourceError, match="cannot reverse from anchor"):
        _reconstruct_versions({"000001", "000002", "000003"}, [impossible], 3)


def test_membership_rows_never_contain_historical_weight():
    delta = _delta(1, "2024-06-14", ["000001"], ["000004"])
    versions = _reconstruct_versions({"000002", "000003", "000004"}, [delta], 3)
    daily = _daily_membership(
        pd.DatetimeIndex(["2024-06-17"]),
        versions,
        pd.Timestamp("2024-06-17"),
        "a" * 64,
        3,
    )

    assert "benchmark_weight" not in daily.columns
    assert json.loads(json.dumps(daily.iloc[0]["source_hash"]))


def test_scope_grid_emits_explicit_false_without_dropping_rows():
    delta = _delta(1, "2024-06-14", ["000001"], ["000004"])
    versions = _reconstruct_versions({"000002", "000003", "000004"}, [delta], 3)
    active = _daily_membership(
        pd.DatetimeIndex(["2024-06-17"]), versions, pd.Timestamp("2024-06-17"), "a" * 64, 3
    )
    scope = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-06-17", "2024-06-17"]),
        "code": ["000004", "000099"],
    })

    grid = _explicit_membership_grid(scope, active)

    assert grid.set_index("code")["is_member"].to_dict() == {
        "000004": True,
        "000099": False,
    }
    assert len(grid) == len(scope)
