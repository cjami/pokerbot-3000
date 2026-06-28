import pytest

from pokerbot_3000 import __version__, cli


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["--version"])

    assert raised.value.code == 0
    assert f"pokerbot-3000 {__version__}" in capsys.readouterr().out


def test_cli_launches_uvicorn_without_browser(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        captured["app"] = app
        captured.update(kwargs)

    def fake_open(_url: str) -> None:
        msg = "Browser should not open when --no-browser is used."
        raise AssertionError(msg)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli.webbrowser, "open", fake_open)

    cli.main(["--no-browser", "--server-name", "127.0.0.1", "--server-port", "9000"])

    assert captured == {
        "app": "pokerbot_3000.app.server:create_app",
        "factory": True,
        "host": "127.0.0.1",
        "port": 9000,
    }


def test_cli_opens_browser_by_default(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(_app: str, **_kwargs: object) -> None:
        captured["server_started"] = True

    def fake_open(url: str) -> bool:
        captured["url"] = url
        return True

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli.webbrowser, "open", fake_open)

    cli.main(["--server-port", "9001"])

    assert captured == {
        "server_started": True,
        "url": "http://127.0.0.1:9001/",
    }
