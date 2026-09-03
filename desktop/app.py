"""PySide6 tray UI and zero-key first-run setup."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from desktop.client import AgentClient, AgentUnavailable


def _agent_command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--background-agent"]
    return [sys.executable, "-m", "desktop.app", "--background-agent"]


def _start_agent_if_needed(client: AgentClient) -> None:
    try:
        client.status()
        return
    except AgentUnavailable:
        pass
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(
        _agent_command(),
        cwd=str(Path.home() if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        start_new_session=os.name != "nt",
    )
    for _ in range(60):
        time.sleep(0.25)
        try:
            client.status()
            return
        except AgentUnavailable:
            continue
    raise AgentUnavailable("StockWatch Agent did not become ready")


def _run_background_agent() -> int:
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    from core.agent import run_agent

    run_agent()
    return 0


def run_desktop() -> int:
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
        QLabel, QLineEdit, QMenu, QMessageBox, QStyle, QSystemTrayIcon,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("StockWatch")
    app.setQuitOnLastWindowClosed(False)
    client = AgentClient()
    try:
        _start_agent_if_needed(client)
    except AgentUnavailable as exc:
        QMessageBox.critical(None, "StockWatch", f"后台服务启动失败：{exc}")
        return 1

    tray = QSystemTrayIcon(app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon))
    tray.setToolTip("StockWatch 家庭盯盘")
    menu = QMenu()

    def open_console():
        webbrowser.open("http://127.0.0.1:8765")

    def request_run(action: str):
        try:
            client.run(action)
            tray.showMessage("StockWatch", "任务已交给后台运行")
        except AgentUnavailable as exc:
            tray.showMessage("StockWatch", f"后台不可用：{exc}")

    menu.addAction("打开高级控制台", open_console)
    menu.addAction("立即检查", lambda: request_run("full"))
    menu.addSeparator()
    quit_action = QAction("退出桌面界面")
    quit_action.triggered.connect(app.quit)
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: open_console() if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
    tray.show()

    try:
        configured = bool(client.status().get("configured"))
    except AgentUnavailable:
        configured = True
    if not configured:
        dialog = QDialog()
        dialog.setWindowTitle("欢迎使用 StockWatch")
        layout = QFormLayout(dialog)
        layout.addRow(QLabel("无需 API Key 也能先使用规则提醒。"))
        watchlist = QLineEdit("600519,000858,510300")
        ai_enabled = QCheckBox("我稍后配置 AI 解读")
        risk_enabled = QCheckBox("启用内置风险评分（若安装包包含模型）")
        startup = QCheckBox("登录电脑后自动运行")
        startup.setChecked(True)
        layout.addRow("自选股代码", watchlist)
        layout.addRow(ai_enabled)
        layout.addRow(risk_enabled)
        layout.addRow(startup)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            client.save_settings({
                "WATCHLIST": watchlist.text(),
                "ENABLE_AI": "auto" if ai_enabled.isChecked() else "off",
                "ENABLE_RISK_MODEL": "true" if risk_enabled.isChecked() else "false",
                "NOTIFY_CHANNEL": "web",
            })
            if startup.isChecked():
                from desktop.autostart import set_enabled

                ok, error = set_enabled(True)
                if not ok:
                    tray.showMessage("StockWatch", f"自动启动尚未启用：{error}")

    last_alert_id = 0
    alerts_initialized = False

    def poll_status():
        nonlocal last_alert_id, alerts_initialized
        try:
            status = client.status()
            alerts = status.get("recent_alerts") or []
            if not alerts_initialized:
                if alerts:
                    last_alert_id = max(int(a.get("id") or 0) for a in alerts)
                alerts_initialized = True
                return
            fresh = [a for a in alerts if int(a.get("id") or 0) > last_alert_id]
            for alert in reversed(fresh):
                tray.showMessage(str(alert.get("title") or "StockWatch 提醒"), str(alert.get("body") or ""))
            if alerts:
                last_alert_id = max(last_alert_id, *(int(a.get("id") or 0) for a in alerts))
        except AgentUnavailable:
            tray.setToolTip("StockWatch · 后台暂不可用")

    poller = QTimer()
    poller.timeout.connect(poll_status)
    poller.start(60_000)
    poll_status()
    return app.exec()


def main() -> int:
    executable_role = Path(sys.executable).stem.lower()
    if "--background-agent" in sys.argv or executable_role == "stockwatchagent":
        return _run_background_agent()
    return run_desktop()


if __name__ == "__main__":
    raise SystemExit(main())
