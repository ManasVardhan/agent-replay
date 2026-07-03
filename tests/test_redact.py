"""Tests for the redaction module and redact CLI command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from agent_replay.cli import cli
from agent_replay.redact import (
    BUILTIN_PATTERNS,
    _compile_patterns,
    redact_text,
    redact_trace,
)
from agent_replay.trace import EventType, Trace


def make_trace_with_secrets() -> Trace:
    trace = Trace(name="secret-trace", metadata={"owner": "alice@example.com"})
    span = trace.add_span("agent", metadata={"api_key": "sk-" + "a1B2c3d4" * 4})
    span.add_event(
        EventType.LLM_REQUEST,
        {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "key is sk-ant-" + "x9Y8z7w6" * 4},
                {"role": "user", "content": "email bob@corp.io please"},
            ],
        },
    )
    span.add_event(
        EventType.TOOL_CALL,
        {
            "tool": "http",
            "args": {"headers": {"Authorization": "Bearer abcdef1234567890TOKEN"}},
        },
    )
    span.add_event(EventType.LOG, {"message": "aws AKIAABCDEFGHIJKLMNOP used"})
    span.close()
    trace.close()
    return trace


class TestRedactText:
    def test_openai_key(self):
        patterns = _compile_patterns()
        text, counts = redact_text("token sk-" + "Ab12Cd34" * 4 + " end", patterns)
        assert "sk-" not in text
        assert counts == {"openai_key": 1}

    def test_anthropic_key_labeled_correctly(self):
        patterns = _compile_patterns()
        text, counts = redact_text("sk-ant-" + "Ab12Cd34" * 4, patterns)
        assert counts == {"anthropic_key": 1}
        assert "[REDACTED:anthropic_key]" in text

    def test_aws_access_key(self):
        patterns = _compile_patterns()
        text, counts = redact_text("AKIAABCDEFGHIJKLMNOP", patterns)
        assert counts == {"aws_access_key": 1}

    def test_github_token(self):
        patterns = _compile_patterns()
        text, counts = redact_text("ghp_" + "a1B2c3D4e" * 4, patterns)
        assert counts == {"github_token": 1}

    def test_bearer_token(self):
        patterns = _compile_patterns()
        text, counts = redact_text("Authorization: Bearer abcd1234efgh5678", patterns)
        assert counts == {"bearer_token": 1}

    def test_email(self):
        patterns = _compile_patterns()
        text, counts = redact_text("contact alice@example.com now", patterns)
        assert counts == {"email": 1}
        assert "alice@example.com" not in text

    def test_multiple_matches_counted(self):
        patterns = _compile_patterns()
        _, counts = redact_text("a@b.co and c@d.io", patterns)
        assert counts == {"email": 2}

    def test_clean_text_untouched(self):
        patterns = _compile_patterns()
        text, counts = redact_text("nothing secret here", patterns)
        assert text == "nothing secret here"
        assert counts == {}

    def test_custom_placeholder(self):
        patterns = _compile_patterns()
        text, _ = redact_text("a@b.co", patterns, placeholder="***")
        assert text == "***"

    def test_extra_pattern_wins_over_builtin(self):
        patterns = _compile_patterns({"email": r"NOMATCH12345"})
        text, counts = redact_text("a@b.co", patterns)
        assert text == "a@b.co"
        assert counts == {}


class TestRedactTrace:
    def test_original_not_modified(self):
        trace = make_trace_with_secrets()
        before = json.dumps(trace.to_dict())
        redact_trace(trace)
        assert json.dumps(trace.to_dict()) == before

    def test_all_secrets_removed(self):
        trace = make_trace_with_secrets()
        redacted, counts = redact_trace(trace)
        dumped = json.dumps(redacted.to_dict())
        assert "sk-" not in dumped
        assert "AKIA" not in dumped
        assert "@example.com" not in dumped
        assert "@corp.io" not in dumped
        assert "Bearer abcdef" not in dumped
        assert sum(counts.values()) >= 5

    def test_counts_by_label(self):
        trace = make_trace_with_secrets()
        _, counts = redact_trace(trace)
        assert counts["openai_key"] == 1
        assert counts["anthropic_key"] == 1
        assert counts["aws_access_key"] == 1
        assert counts["bearer_token"] == 1
        assert counts["email"] == 2

    def test_structure_preserved(self):
        trace = make_trace_with_secrets()
        redacted, _ = redact_trace(trace)
        assert redacted.trace_id == trace.trace_id
        assert redacted.name == trace.name
        assert len(redacted.spans) == len(trace.spans)
        assert redacted.spans[0].span_id == trace.spans[0].span_id
        assert [e.event_id for e in redacted.spans[0].events] == [
            e.event_id for e in trace.spans[0].events
        ]

    def test_nested_lists_and_dicts(self):
        trace = Trace(name="nested")
        span = trace.add_span("s")
        span.add_event(
            EventType.LOG,
            {"deep": [{"inner": ["a@b.co", {"more": "c@d.io"}]}, "plain"]},
        )
        redacted, counts = redact_trace(trace)
        assert counts == {"email": 2}
        data = redacted.spans[0].events[0].data
        assert data["deep"][0]["inner"][0] == "[REDACTED:email]"
        assert data["deep"][0]["inner"][1]["more"] == "[REDACTED:email]"
        assert data["deep"][1] == "plain"

    def test_non_string_values_untouched(self):
        trace = Trace(name="types")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {"n": 42, "f": 1.5, "b": True, "none": None})
        redacted, counts = redact_trace(trace)
        assert counts == {}
        assert redacted.spans[0].events[0].data == {
            "n": 42,
            "f": 1.5,
            "b": True,
            "none": None,
        }

    def test_extra_patterns(self):
        trace = Trace(name="custom")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {"message": "internal id ACME-12345"})
        redacted, counts = redact_trace(trace, extra_patterns={"acme_id": r"ACME-\d+"})
        assert counts == {"acme_id": 1}
        assert "[REDACTED:acme_id]" in redacted.spans[0].events[0].data["message"]

    def test_roundtrip_save_load(self, tmp_path):
        trace = make_trace_with_secrets()
        redacted, _ = redact_trace(trace)
        path = tmp_path / "clean.jsonl"
        redacted.save(path)
        loaded = Trace.load(path)
        assert "sk-" not in json.dumps(loaded.to_dict())
        assert loaded.event_count == trace.event_count


class TestRedactCLI:
    def test_redact_writes_output(self, tmp_path):
        trace = make_trace_with_secrets()
        src = tmp_path / "trace.jsonl"
        trace.save(src)
        out = tmp_path / "clean.jsonl"

        runner = CliRunner()
        result = runner.invoke(cli, ["redact", str(src), "-o", str(out)])
        assert result.exit_code == 0
        assert "Redacted" in result.output
        assert out.exists()
        assert "sk-" not in out.read_text()

    def test_default_output_name(self, tmp_path):
        trace = make_trace_with_secrets()
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["redact", str(src)])
        assert result.exit_code == 0
        assert (tmp_path / "trace.redacted.jsonl").exists()

    def test_clean_trace_message(self, tmp_path):
        trace = Trace(name="clean")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {"message": "hello"})
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["redact", str(src)])
        assert result.exit_code == 0
        assert "No sensitive data found" in result.output

    def test_custom_pattern(self, tmp_path):
        trace = Trace(name="custom")
        span = trace.add_span("s")
        span.add_event(EventType.LOG, {"message": "user SSN 123-45-6789"})
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["redact", str(src), "-p", r"ssn=\d{3}-\d{2}-\d{4}"]
        )
        assert result.exit_code == 0
        assert "ssn" in result.output
        out = (tmp_path / "trace.redacted.jsonl").read_text()
        assert "123-45-6789" not in out

    def test_bad_pattern_spec_rejected(self, tmp_path):
        trace = Trace(name="t")
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["redact", str(src), "-p", "no-equals-sign"])
        assert result.exit_code != 0
        assert "LABEL=REGEX" in result.output

    def test_invalid_regex_rejected(self, tmp_path):
        trace = Trace(name="t")
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["redact", str(src), "-p", "bad=[unclosed"])
        assert result.exit_code != 0
        assert "invalid regex" in result.output

    def test_custom_placeholder(self, tmp_path):
        trace = make_trace_with_secrets()
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["redact", str(src), "--placeholder", "<HIDDEN>"]
        )
        assert result.exit_code == 0
        out = (tmp_path / "trace.redacted.jsonl").read_text()
        assert "<HIDDEN>" in out

    def test_redacted_file_loadable(self, tmp_path):
        trace = make_trace_with_secrets()
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        runner.invoke(cli, ["redact", str(src)])
        loaded = Trace.load(tmp_path / "trace.redacted.jsonl")
        assert loaded.event_count == trace.event_count


class TestBuiltinPatterns:
    def test_all_builtin_patterns_compile(self):
        import re

        for name, pat in BUILTIN_PATTERNS.items():
            re.compile(pat)

    def test_public_api_exported(self):
        import agent_replay

        assert hasattr(agent_replay, "redact_trace")
        assert hasattr(agent_replay, "BUILTIN_PATTERNS")


class TestDurationZeroBug:
    def test_info_zero_duration_not_na(self, tmp_path):
        trace = Trace(name="instant", start_time=100.0, end_time=100.0)
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(src)])
        assert result.exit_code == 0
        assert "0.000s" in result.output
        assert "N/A" not in result.output

    def test_stats_zero_duration_not_na(self, tmp_path):
        trace = Trace(name="instant", start_time=100.0, end_time=100.0)
        src = tmp_path / "trace.jsonl"
        trace.save(src)

        runner = CliRunner()
        result = runner.invoke(cli, ["stats", str(src)])
        assert result.exit_code == 0
        assert "0.000s" in result.output
        assert "N/A" not in result.output
