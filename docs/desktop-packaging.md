# Desktop packaging foundation

This document covers the M0/M1 preview pipeline. Preview artifacts are not yet
code-signed and are intended for maintainer testing, not general distribution.

## Runtime shape

- `python main.py agent` runs the single background Agent: Shanghai-time
  scheduler, local dashboard/API, database access, and optional bot thread.
- `python -m desktop.app` runs the PySide6 tray client. Closing the tray client
  does not stop the Agent.
- Settings and runtime data live under `STOCKWATCH_HOME` (default
  `~/.stockwatch`). Existing source installs with a project `.env` remain
  compatible.
- The local API binds to `127.0.0.1` by default. Remote access still requires
  `WEB_AUTH_TOKEN` and is not part of M1.

The desktop UI is deliberately small: first-run setup, status notifications,
manual checks, and a link to the existing browser dashboard. It does not run
the analysis loop itself.

## Build locally

Create a clean environment and install `requirements-desktop.txt`, then run:

```bash
python -m PyInstaller --noconfirm --clean packaging/stockwatch-desktop.spec
```

PyInstaller must run on the target operating system and architecture. The CI
workflow therefore publishes separate macOS arm64, macOS x64, and Windows x64
artifacts rather than a universal binary.

## Inference model bundle

Training scripts and scikit-learn are excluded from the desktop build. To add
the inference assets, stage them before PyInstaller:

```bash
python packaging/prepare_models.py /path/to/model-directory --require-risk
```

The script copies supported model/runtime metadata into the ignored
`packaging/runtime-models` directory and writes SHA-256 values to a bundle
manifest. At first launch the Agent copies missing bundled files into
`$STOCKWATCH_HOME/models`; it never overwrites a user-managed model.

For CI release tags, configure the repository secret
`STOCKWATCH_MODEL_BUNDLE_URL` with a short-lived or otherwise access-controlled
URL to a ZIP containing the approved inference assets. A tagged build fails if
the risk model is absent. Pull-request preview builds remain rules-only when no
bundle URL is configured.

## Distribution boundary

The workflow currently stops at unsigned preview ZIPs. M2 must add these steps
without changing the application layout:

1. macOS: embed the LaunchAgent plist, sign nested Mach-O files from the inside
   out with hardened runtime, sign the app, build/sign the DMG, notarize, and
   staple.
2. Windows: sign the executable and installer with the project certificate.
3. Replace the file secret backend with Keychain/Credential Manager and add a
   signed update feed.

Do not publish an unsigned preview as a family-facing release.

The macOS packaging step copies the same frozen executable to
`Contents/MacOS/StockWatchAgent`. The embedded LaunchAgent uses `BundleProgram`
to run that dedicated name without unsupported launch arguments; the entry
point chooses the background role from the executable name.
