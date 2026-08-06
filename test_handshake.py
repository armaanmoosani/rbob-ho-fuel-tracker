import json
from unittest.mock import MagicMock, patch

import handshake


def test_environment_credentials_take_precedence(monkeypatch):
    monkeypatch.setenv("SCHWAB_APP_KEY", "key")
    monkeypatch.setenv("SCHWAB_APP_SECRET", "secret")
    monkeypatch.setenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
    with patch("handshake.load_keychain_credentials") as load_keychain:
        credentials = handshake.get_credentials()
    assert credentials["app_key"] == "key"
    load_keychain.assert_not_called()


def test_keychain_credentials_are_loaded_without_prompt():
    expected = {
        "app_key": "key",
        "app_secret": "secret",
        "redirect_uri": "https://127.0.0.1",
    }
    with patch("handshake.environment_credentials", return_value=None), patch(
        "handshake.load_keychain_credentials", return_value=expected
    ), patch("handshake.prompt_for_credentials") as prompt:
        assert handshake.get_credentials() == expected
    prompt.assert_not_called()


def test_keychain_payload_requires_all_fields():
    result = MagicMock(returncode=0, stdout=json.dumps({"app_key": "key"}))
    with patch("handshake.subprocess.run", return_value=result):
        assert handshake.load_keychain_credentials() is None


def test_parse_authorization_code_rejects_mismatched_state():
    try:
        handshake.parse_authorization_code("https://127.0.0.1/?code=test&state=other", "expected")
    except ValueError as error:
        assert "state mismatch" in str(error).lower()
    else:
        raise AssertionError("Expected a state mismatch error")


def test_exchange_code_uses_saved_redirect_uri():
    response = MagicMock()
    response.json.return_value = {"refresh_token": "refresh"}
    credentials = {
        "app_key": "key",
        "app_secret": "secret",
        "redirect_uri": "https://127.0.0.1",
    }
    with patch("handshake.requests.post", return_value=response) as post:
        assert handshake.exchange_code(credentials, "code") == {"refresh_token": "refresh"}
    assert post.call_args.kwargs["data"]["redirect_uri"] == credentials["redirect_uri"]


def test_github_repository_uses_origin_url_when_environment_is_absent(monkeypatch):
    monkeypatch.delenv("GH_REPO", raising=False)
    result = MagicMock(returncode=0, stdout="git@github.com:armaanmoosani/rbob-ho-fuel-tracker.git\n")
    with patch("handshake.subprocess.run", return_value=result):
        assert handshake.github_repository() == "armaanmoosani/rbob-ho-fuel-tracker"


def test_refresh_token_is_supplied_to_gh_over_stdin(monkeypatch):
    monkeypatch.setenv("GH_REPO", "armaanmoosani/rbob-ho-fuel-tracker")
    result = MagicMock(returncode=0, stderr="")
    with patch("handshake.subprocess.run", return_value=result) as run:
        assert handshake.update_github_refresh_token("refresh-token") == (True, None)
    args, kwargs = run.call_args
    assert args[0] == [
        "gh", "secret", "set", "SCHWAB_REFRESH_TOKEN", "--repo",
        "armaanmoosani/rbob-ho-fuel-tracker",
    ]
    assert kwargs["input"] == "refresh-token"
    assert "--body" not in args[0]


def test_handshake_does_not_print_token_after_automatic_secret_update(capsys):
    credentials = {
        "app_key": "key",
        "app_secret": "secret",
        "redirect_uri": "https://127.0.0.1",
    }
    with patch("handshake.get_credentials", return_value=credentials), patch(
        "handshake.secrets.token_urlsafe", return_value="state"
    ), patch("builtins.input", return_value="https://127.0.0.1/?code=code&state=state"), patch(
        "handshake.exchange_code", return_value={"refresh_token": "do-not-print"}
    ), patch("handshake.update_github_refresh_token", return_value=(True, None)):
        handshake.run_handshake()
    output = capsys.readouterr().out
    assert "SCHWAB_REFRESH_TOKEN was updated" in output
    assert "do-not-print" not in output
