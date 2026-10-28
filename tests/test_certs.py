"""--system-certs / WISSI_VIZ_SYSTEM_CERTS: downloads trust the operating system's certificates (corporate networks)."""

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


def test_flag_is_taken_out_anywhere_and_switches_it_on(calls):
    assert certs.take_flag(['--demo', '--system-certs', 'quick']) == ['--demo', 'quick']
    assert calls == [1] and certs.enabled()
    certs.take_flag(['--system-certs'])  # once per process
    assert calls == [1]


def test_off_without_the_flag_or_variable(calls):
    assert certs.take_flag(['serve', '/data']) == ['serve', '/data']
    assert calls == [] and not certs.enabled()


@pytest.mark.parametrize('value, on', [('1', True), ('true', True), ('YES', True), ('0', False), ('', False)])
def test_environment_variable(calls, monkeypatch, value, on):
    monkeypatch.setenv(certs.ENV, value)
    certs.take_flag([])
    assert bool(calls) is on


def test_hint_only_for_certificate_failures(calls):
    err = '<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer>'
    assert '--system-certs' in certs.hint(err) and certs.ENV in certs.hint(err)
    assert certs.hint('HTTP Error 404: Not Found') == ''
    certs.use_system_certs()
    assert certs.hint(err) == ''  # already on: the flag would not help


def test_cli_strips_the_flag_before_parsing(calls, tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(['--system-certs', 'info', '--help'])
    assert calls == [1]
    assert 'usage: wissi-viz info' in capsys.readouterr().out


def test_help_lists_the_flag():
    out = subprocess.run(
        [sys.executable, '-m', 'wissi_viz.cli', '--help'], capture_output=True, text=True, encoding='utf-8', check=True
    ).stdout
    assert '--system-certs' in out


def test_really_uses_the_system_store():
    """In a fresh interpreter: after the switch, new SSL contexts are truststore's (OS verification)."""
    code = (
        'import ssl, wissi_viz.certs as c; c.use_system_certs(); print(type(ssl.create_default_context()).__module__)'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout
    assert out.strip().startswith('truststore')
    assert type(ssl.create_default_context()).__module__ == 'ssl'  # this process is untouched
