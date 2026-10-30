"""Downloads trust the operating system's certificates by default (``--no-system-certs`` / ``WISSI_VIZ_SYSTEM_CERTS=0``
keep Python's own list)."""

from __future__ import annotations

import ssl
import subprocess
import sys

import pytest

from wissi_viz import certs, cli


@pytest.fixture
def calls(monkeypatch):
    seen = []
    monkeypatch.setattr(certs.truststore, 'inject_into_ssl', lambda: seen.append(1))
    monkeypatch.setattr(certs, '_on', False)
    monkeypatch.delenv(certs.ENV, raising=False)
    return seen


def test_on_by_default_once_per_process(calls):
    assert certs.take_flag(['serve', '/data']) == ['serve', '/data']
    assert calls == [1] and certs.enabled()
    certs.take_flag([])
    assert calls == [1]


def test_flags_are_taken_out_anywhere(calls):
    assert certs.take_flag(['--demo', '--system-certs', 'quick']) == ['--demo', 'quick']  # the old flag: still fine
    assert calls == [1]


def test_opt_out(calls):
    assert certs.take_flag(['--demo', '--no-system-certs', 'quick']) == ['--demo', 'quick']
    assert calls == [] and not certs.enabled()


@pytest.mark.parametrize('value, on', [('1', True), ('true', True), ('', True), ('0', False), ('off', False),
                                       ('NO', False)])  # fmt: skip
def test_environment_variable(calls, monkeypatch, value, on):
    monkeypatch.setenv(certs.ENV, value)
    certs.take_flag([])
    assert bool(calls) is on


def test_hint_only_for_certificate_failures(calls):
    err = '<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer>'
    assert '--no-system-certs' in certs.hint(err) and certs.ENV in certs.hint(err)
    assert certs.hint('HTTP Error 404: Not Found') == ''
    certs.use_system_certs()
    assert 'ask IT' in certs.hint(err)  # already on: the computer does not trust it either


def test_cli_strips_the_flags_before_parsing(calls, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(['--system-certs', 'info', '--help'])
    assert calls == [1]
    assert 'usage: wissi-viz info' in capsys.readouterr().out


def test_help_lists_the_opt_out():
    out = subprocess.run(
        [sys.executable, '-m', 'wissi_viz.cli', '--help'], capture_output=True, text=True, encoding='utf-8', check=True
    ).stdout
    assert '--no-system-certs' in out


def test_really_uses_the_system_store():
    """In a fresh interpreter: after the switch, new SSL contexts are truststore's (OS verification)."""
    code = (
        'import ssl, wissi_viz.certs as c; c.use_system_certs(); print(type(ssl.create_default_context()).__module__)'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.strip().startswith('truststore')
    assert type(ssl.create_default_context()).__module__ == 'ssl'  # this process is untouched
