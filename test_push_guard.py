"""Guards against a script publishing to the production branch by accident.

``backtest.main()`` and ``ingest_prices.main()`` both ended with an
unconditional ``git commit && git push``.  Simply running either script to
inspect its output -- locally, or from any harness -- published to ``main``.
Publishing now requires GitHub Actions or an explicit opt-in.
"""

import subprocess
import sys
from unittest.mock import patch

import backtest
import ingest_prices


class TestPushAuthorisation:
    def test_denied_by_default(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("BACKTEST_ALLOW_PUSH", raising=False)
        monkeypatch.delenv("INGEST_ALLOW_PUSH", raising=False)
        monkeypatch.setattr(sys, "argv", ["backtest.py"])
        assert backtest._push_is_authorised() is False
        assert ingest_prices.push_is_authorised() is False

    def test_allowed_in_github_actions(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setattr(sys, "argv", ["backtest.py"])
        assert backtest._push_is_authorised() is True
        assert ingest_prices.push_is_authorised() is True

    def test_allowed_by_explicit_flag(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr(sys, "argv", ["backtest.py", "--commit"])
        assert backtest._push_is_authorised() is True
        assert ingest_prices.push_is_authorised() is True

    def test_allowed_by_environment_opt_in(self, monkeypatch):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.setattr(sys, "argv", ["backtest.py"])
        monkeypatch.setenv("BACKTEST_ALLOW_PUSH", "1")
        assert backtest._push_is_authorised() is True
        monkeypatch.setenv("INGEST_ALLOW_PUSH", "1")
        assert ingest_prices.push_is_authorised() is True


class TestNoGitWithoutAuthorisation:
    """The decisive test: no git subprocess may run when unauthorised."""

    def test_backtest_runs_no_git_command(self, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("BACKTEST_ALLOW_PUSH", raising=False)
        monkeypatch.setattr(sys, "argv", ["backtest.py"])
        with patch.object(subprocess, "run") as run:
            backtest.git_commit_push("test message")
        run.assert_not_called()
        assert "not committing" in capsys.readouterr().out

    def test_ingest_runs_no_git_command(self, monkeypatch, capsys):
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("INGEST_ALLOW_PUSH", raising=False)
        monkeypatch.setattr(sys, "argv", ["ingest_prices.py"])
        with patch.object(subprocess, "run") as run:
            ingest_prices.git_commit_push("test message")
        run.assert_not_called()
        assert "not committing" in capsys.readouterr().out

    def test_backtest_does_push_when_authorised(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        with patch.object(backtest.subprocess, "run") as run:
            run.return_value = type("R", (), {"returncode": 1, "args": []})()
            backtest.git_commit_push("test message")
        commands = [c.args[0] for c in run.call_args_list if c.args]
        assert any(cmd[:2] == ["git", "push"] for cmd in commands), (
            "CI must still publish; the guard may not block GitHub Actions")
