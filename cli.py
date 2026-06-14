"""Unified ``stockwatch`` command-line entry.

Lets users type things like::

    stockwatch                  # 总览：服务状态 + Web 控制台地址
    stockwatch dashboard        # 确保 Web 控制台在跑，打印访问方式
    stockwatch start            # 同 dashboard（开 Web 控制台）
    stockwatch stop             # 停 Web 控制台
    stockwatch restart          # 重启 Web 控制台
    stockwatch status           # 所有服务详细状态
    stockwatch logs [-f]        # 看 Web 控制台日志
    stockwatch open             # 在浏览器中打开
    stockwatch daemon start     # 管理盯盘守护进程（同样 stop/restart/status/logs）
    stockwatch bot start        # 管理飞书机器人服务
    stockwatch test             # 自检 AKShare / LLM / 飞书
    stockwatch once             # 立即跑一次完整分析
    stockwatch demo 600519      # 终端问答体验
    stockwatch install          # 把 stockwatch 链接到 ~/.local/bin
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DASHBOARD_PORT = int(os.environ.get("STOCKWATCH_WEB_PORT", "8765"))

SERVICES: dict[str, str] = {
    "dashboard": "stockwatch-dashboard.service",
    "daemon": "stockwatch.service",
    "bot": "stockwatch-bot.service",
}
SERVICE_DESC: dict[str, str] = {
    "dashboard": f"Web 控制台（端口 {DASHBOARD_PORT}）",
    "daemon": "盯盘守护进程（按调度自动分析）",
    "bot": "飞书交互式查询机器人",
}
LOG_FILES: dict[str, str] = {
    "dashboard": "dashboard.log",
    "daemon": "daemon.log",
    "bot": "bot.log",
}

HELP_TEXT = """\
stockwatch — StockWatch 统一命令

常用:
  stockwatch                     总览：服务状态 + Web 控制台地址
  stockwatch dashboard           确保 Web 控制台在跑，打印访问方式
  stockwatch start               同 dashboard（一键开 Web 控制台）
  stockwatch stop                停 Web 控制台
  stockwatch restart             重启 Web 控制台
  stockwatch status              所有服务详细状态
  stockwatch logs [-f]           看 Web 控制台日志（-f 持续跟踪）
  stockwatch open                在浏览器中打开 Web 控制台

服务管理:
  stockwatch daemon  <start|stop|restart|status|logs [-f]>   盯盘守护进程
  stockwatch bot     <start|stop|restart|status|logs [-f]>   飞书机器人

其它:
  stockwatch once                立即跑一次完整分析
  stockwatch test                自检 AKShare / LLM / 飞书
  stockwatch demo [问题]         终端问答体验（无需飞书）
  stockwatch report [参数]       生成信号复盘报告
  stockwatch backtest [参数]     量化信号回测
  stockwatch monitor             跑一次盘中异动监控
  stockwatch install             把 stockwatch 链接到 ~/.local/bin
  stockwatch help                显示本帮助
"""


# ─── system probing ─────────────────────────────────────────────────────────────

def _have_systemd() -> bool:
    return shutil.which("systemctl") is not None and Path(
        "/etc/systemd/system/stockwatch-dashboard.service"
    ).exists()


def _service_installed(name: str) -> bool:
    unit = SERVICES[name]
    return Path(f"/etc/systemd/system/{unit}").exists()


def _service_state(name: str) -> str:
    """Return ``active`` / ``inactive`` / ``failed`` / ``missing`` / ``unknown``."""
    if not _service_installed(name):
        return "missing"
    try:
        out = subprocess.check_output(
            ["systemctl", "is-active", SERVICES[name]],
            stderr=subprocess.DEVNULL, text=True, timeout=3,
        ).strip()
    except subprocess.CalledProcessError as e:
        return (e.output or "").strip() or "inactive"
    except Exception:
        return "unknown"
    return out or "unknown"


def _sudo_prefix() -> list[str]:
    return [] if os.geteuid() == 0 else ["sudo", "-n"]


def _run_systemctl(action: str, name: str) -> int:
    if not _have_systemd():
        print("（未检测到 systemd，无法用服务方式管理；请改用 bash start.sh）", file=sys.stderr)
        return 1
    if not _service_installed(name):
        print(f"服务 {SERVICES[name]} 未安装。", file=sys.stderr)
        print(
            f"安装：sudo cp {PROJECT_DIR}/scripts/{SERVICES[name].replace('.service','')}.service "
            f"/etc/systemd/system/  &&  sudo systemctl daemon-reload  &&  "
            f"sudo systemctl enable --now {SERVICES[name]}",
            file=sys.stderr,
        )
        return 1
    cmd = _sudo_prefix() + ["systemctl", action, SERVICES[name]]
    return subprocess.call(cmd)


# ─── URL helpers ────────────────────────────────────────────────────────────────

def _detect_lan_ip() -> str | None:
    if not shutil.which("ip"):
        return None
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            addr = parts[3].split("/")[0]
            if not addr.startswith("127."):
                return addr
    return None


def _print_dashboard_url(prefix: str = "") -> None:
    port = DASHBOARD_PORT
    if prefix:
        print(prefix)
    print(f"  本机     http://127.0.0.1:{port}")
    lan = _detect_lan_ip()
    if lan:
        print(f"  局域网   http://{lan}:{port}")
    if _have_systemd():
        print()
        print("浏览器在另一台机器时：")
        if lan:
            print(f"  · 同一局域网直接打开 http://{lan}:{port}")
        print("  · 或 SSH 端口转发：")
        print("        ssh -L 8765:127.0.0.1:8765 rpi@mc.void1211.com -p 1211")
        print("      然后浏览器打开 http://127.0.0.1:8765")


def _print_auth_hint() -> None:
    env = PROJECT_DIR / ".env"
    if not env.exists():
        return
    try:
        for line in env.read_text().splitlines():
            if line.startswith("WEB_AUTH_TOKEN=") and line.split("=", 1)[1].strip():
                print("  · 登录令牌（WEB_AUTH_TOKEN）已设置，首次访问需要输入。")
                return
    except OSError:
        pass


# ─── command handlers ──────────────────────────────────────────────────────────

def cmd_overview(_args) -> int:
    """Default action (no args): show service state + dashboard URL."""
    print("StockWatch")
    print("=" * 50)
    have_sd = _have_systemd()
    if have_sd:
        for name in ("dashboard", "daemon", "bot"):
            state = _service_state(name)
            marker = {
                "active": "●", "inactive": "○", "failed": "✗",
                "missing": "·", "unknown": "?",
            }.get(state, "?")
            print(f"  {marker} {name:9s} {state:8s}  {SERVICE_DESC[name]}")
    else:
        print("  （未检测到 systemd 服务；以下为本地直跑方式）")
    print("=" * 50)
    if not have_sd or _service_state("dashboard") == "active":
        print()
        print("Web 控制台地址：")
        _print_dashboard_url()
        _print_auth_hint()
    else:
        print()
        print("Web 控制台未运行。开起来：stockwatch start")
    if have_sd:
        print()
        print("更多：stockwatch help")
    return 0


def cmd_dashboard(_args) -> int:
    """Ensure dashboard is up and print how to access it."""
    if not _have_systemd():
        # Local dev: run in foreground (Ctrl+C exits).
        print(f"在前台启动 Web 控制台（Ctrl+C 退出）...")
        print(f"浏览器打开：http://127.0.0.1:{DASHBOARD_PORT}")
        os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
        return subprocess.call([sys.executable, str(PROJECT_DIR / "main.py"), "dashboard"])

    if not _service_installed("dashboard"):
        return _run_systemctl("status", "dashboard")  # will print install hint

    state = _service_state("dashboard")
    if state != "active":
        print(f"Web 控制台未在运行（{state}），正在启动...")
        rc = _run_systemctl("start", "dashboard")
        if rc != 0:
            return rc
        # systemd starts async — wait briefly for the port.
        for _ in range(20):
            time.sleep(0.25)
            if _service_state("dashboard") == "active":
                break
    print("Web 控制台已运行：")
    _print_dashboard_url()
    _print_auth_hint()
    return 0


def cmd_stop(_args) -> int:
    if not _have_systemd():
        print("本地直跑模式：请到运行 stockwatch start 的窗口按 Ctrl+C 即可。")
        return 0
    return _run_systemctl("stop", "dashboard")


def cmd_restart(_args) -> int:
    if not _have_systemd():
        print("本地直跑模式：请先 Ctrl+C 终止旧实例，再 stockwatch start。")
        return 1
    rc = _run_systemctl("restart", "dashboard")
    if rc == 0:
        print()
        _print_dashboard_url("Web 控制台已重启：")
        _print_auth_hint()
    return rc


def cmd_status(_args) -> int:
    if not _have_systemd():
        return cmd_overview(_args)
    rc = 0
    for name in ("dashboard", "daemon", "bot"):
        if not _service_installed(name):
            continue
        print(f"━━━ {name} ({SERVICES[name]}) ━━━")
        sub_rc = subprocess.call(
            ["systemctl", "status", SERVICES[name], "--no-pager", "-n", "5"]
        )
        rc = rc or sub_rc
        print()
    return rc


def cmd_logs(args, service: str = "dashboard") -> int:
    log_file = Path.home() / ".stockwatch" / "logs" / LOG_FILES[service]
    if log_file.exists():
        cmd = ["tail", "-n", str(args.tail)]
        if args.follow:
            cmd.append("-f")
        cmd.append(str(log_file))
        return subprocess.call(cmd)
    # Fall back to journalctl when running under systemd without a log file yet.
    if _have_systemd() and _service_installed(service):
        cmd = ["journalctl", "-u", SERVICES[service], "-n", str(args.tail)]
        if args.follow:
            cmd.append("-f")
        return subprocess.call(_sudo_prefix() + cmd)
    print(f"日志不存在：{log_file}", file=sys.stderr)
    return 1


def cmd_open(_args) -> int:
    url = f"http://127.0.0.1:{DASHBOARD_PORT}"
    opener: list[str] | None = None
    if sys.platform == "darwin" and shutil.which("open"):
        opener = ["open", url]
    elif sys.platform.startswith("linux") and shutil.which("xdg-open") and os.environ.get("DISPLAY"):
        opener = ["xdg-open", url]
    if opener:
        print(f"在浏览器中打开 {url} ...")
        return subprocess.call(opener)
    print("（当前环境无可用浏览器打开器，请手动访问下面地址）")
    _print_dashboard_url()
    _print_auth_hint()
    return 0


def _cmd_service_group(args, service: str) -> int:
    """Generic start/stop/restart/status/logs for a named service."""
    action = args.action
    if action == "logs":
        return cmd_logs(args, service)
    if action == "status":
        if not _service_installed(service):
            print(f"○ {service} ({SERVICES[service]})  未安装")
            return 0
        return subprocess.call(
            ["systemctl", "status", SERVICES[service], "--no-pager", "-n", "10"]
        )
    return _run_systemctl(action, service)


def cmd_daemon(args) -> int:
    return _cmd_service_group(args, "daemon")


def cmd_bot(args) -> int:
    return _cmd_service_group(args, "bot")


def _delegate(mode: str, argv: list[str]) -> int:
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    return subprocess.call([sys.executable, str(PROJECT_DIR / "main.py"), mode, *argv])


# ─── install ───────────────────────────────────────────────────────────────────

def _ensure_path_in_bashrc() -> None:
    target = 'export PATH="$HOME/.local/bin:$PATH"'
    for rc_name in (".bashrc", ".zshrc"):
        rc = Path.home() / rc_name
        if not rc.exists():
            continue
        text = rc.read_text()
        if ".local/bin" in text:
            continue
        print(f"提示：~/.local/bin 不在 PATH 中，已写入 {rc}：")
        print(f"  {target}")
        print(f"新终端将自动生效；当前 shell 请执行：source {rc}")
        with rc.open("a") as f:
            f.write(f"\n# Added by: stockwatch install\n{target}\n")


def cmd_install(_args) -> int:
    src = PROJECT_DIR / "stockwatch"
    if not src.exists():
        print(f"错误：找不到 wrapper {src}", file=sys.stderr)
        return 1
    if not os.access(src, os.X_OK):
        src.chmod(0o755)
    dst_dir = Path.home() / ".local" / "bin"
    dst = dst_dir / "stockwatch"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            current = dst.resolve()
        except OSError:
            current = None
        if current == src.resolve():
            print(f"已安装：{dst} -> {src}")
        else:
            print(f"已存在：{dst} -> {current}（不是当前项目）")
            ans = input("替换吗？[y/N] ").strip().lower()
            if ans != "y":
                return 0
            dst.unlink()
            dst.symlink_to(src)
            print(f"已替换：{dst} -> {src}")
    else:
        dst.symlink_to(src)
        print(f"已创建：{dst} -> {src}")
    _ensure_path_in_bashrc()
    print()
    print("现在可以在任意目录运行：stockwatch dashboard")
    return 0


# ─── argparse plumbing ─────────────────────────────────────────────────────────

PASSTHROUGH_TO_MAIN = {"once", "test", "monitor", "demo", "report", "backtest"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stockwatch", description="StockWatch 统一命令",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=HELP_TEXT,
        add_help=False,
    )
    p.add_argument("-h", "--help", action="store_true", help="显示帮助")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("dashboard", help="开 Web 控制台并打印访问方式")
    sub.add_parser("start", help="同 dashboard")
    sub.add_parser("stop", help="停 Web 控制台")
    sub.add_parser("restart", help="重启 Web 控制台")
    sub.add_parser("status", help="所有服务详细状态")
    sub.add_parser("open", help="在浏览器中打开 Web 控制台")

    lp = sub.add_parser("logs", help="看 Web 控制台日志")
    lp.add_argument("-f", "--follow", action="store_true", help="持续跟踪")
    lp.add_argument("-n", "--tail", type=int, default=200, help="末尾行数（默认 200）")

    for group_name, help_text in (("daemon", "管理盯盘守护进程"), ("bot", "管理飞书机器人")):
        gp = sub.add_parser(group_name, help=help_text)
        gp.add_argument(
            "action",
            choices=["start", "stop", "restart", "status", "logs"],
            help="子动作",
        )
        gp.add_argument("-f", "--follow", action="store_true", help="logs: 持续跟踪")
        gp.add_argument("-n", "--tail", type=int, default=200, help="logs: 末尾行数")

    sub.add_parser("install", help="把 stockwatch 链接到 ~/.local/bin")
    sub.add_parser("help", help="显示帮助")
    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return cmd_overview(None)
    if argv[0] in ("-h", "--help", "help"):
        print(HELP_TEXT)
        return 0
    if argv[0] in PASSTHROUGH_TO_MAIN:
        return _delegate(argv[0], argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.help:
        print(HELP_TEXT)
        return 0

    handler = {
        "dashboard": cmd_dashboard,
        "start":     cmd_dashboard,
        "stop":      cmd_stop,
        "restart":   cmd_restart,
        "status":    cmd_status,
        "open":      cmd_open,
        "logs":      lambda a: cmd_logs(a, "dashboard"),
        "daemon":    cmd_daemon,
        "bot":       cmd_bot,
        "install":   cmd_install,
    }.get(args.cmd)
    if handler is None:
        return cmd_overview(args)
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
