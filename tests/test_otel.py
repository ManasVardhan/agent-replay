"""Tests for the OpenTelemetry OTLP/JSON exporter."""

import json
from pathlib import Path

from click.testing import CliRunner

from agent_replay import __version__
from agent_replay.cli import cli
from agent_replay.otel import _any_value, _nanos, _normalize_id, export_otlp, to_otlp
from agent_replay.recorder import Recorder
from agent_replay.trace import EventType, Span, Trace


def _make_trace() -> Trace:
    with Recorder("otel-test") as rec:
        with rec.span("step-one"):
            rec.llm_request(model="gpt-4")
            rec.llm_response(content="done", tokens=10)
        with rec.span("step-two"):
            rec.tool_call("search", {"q": "otel"})
    return rec.trace


class TestNormalizeId:
    def test_pads_short_hex_trace_id(self):
        assert _normalize_id("abc123", 32) == "abc123".zfill(32)

    def test_pads_short_hex_span_id(self):
        result = _normalize_id("deadbeef1234", 16)
        assert result == "0000deadbeef1234"
        assert len(result) == 16

    def test_truncates_long_hex(self):
        raw = "a" * 40
        assert _normalize_id(raw, 32) == "a" * 32

    def test_uppercase_hex_is_lowercased(self):
        assert _normalize_id("DEADBEEF", 16) == "deadbeef".zfill(16)

    def test_non_hex_is_hashed_deterministically(self):
        a = _normalize_id("not-hex!", 32)
        b = _normalize_id("not-hex!", 32)
        assert a == b
        assert len(a) == 32
        assert set(a) <= set("0123456789abcdef")

    def test_empty_string_is_hashed(self):
        result = _normalize_id("", 16)
        assert len(result) == 16

    def test_different_inputs_differ(self):
        assert _normalize_id("alpha", 32) != _normalize_id("beta", 32)


class TestNanos:
    def test_whole_seconds(self):
        assert _nanos(1.0) == "1000000000"

    def test_fractional_seconds(self):
        assert _nanos(1700000000.5) == "1700000000500000000"

    def test_zero(self):
        assert _nanos(0.0) == "0"


class TestAnyValue:
    def test_string(self):
        assert _any_value("hi") == {"stringValue": "hi"}

    def test_bool_before_int(self):
        assert _any_value(True) == {"boolValue": True}

    def test_int_is_string_encoded(self):
        assert _any_value(42) == {"intValue": "42"}

    def test_float(self):
        assert _any_value(1.5) == {"doubleValue": 1.5}

    def test_nested_dict_becomes_json_string(self):
        result = _any_value({"a": 1})
        assert json.loads(result["stringValue"]) == {"a": 1}

    def test_list_becomes_json_string(self):
        result = _any_value([1, 2])
        assert json.loads(result["stringValue"]) == [1, 2]

    def test_none_becomes_json_null_string(self):
        assert _any_value(None) == {"stringValue": "null"}


class TestToOtlp:
    def test_top_level_structure(self):
        doc = to_otlp(_make_trace())
        assert len(doc["resourceSpans"]) == 1
        rs = doc["resourceSpans"][0]
        assert len(rs["scopeSpans"]) == 1
        scope = rs["scopeSpans"][0]["scope"]
        assert scope == {"name": "agent-replay", "version": __version__}

    def test_service_name_resource_attribute(self):
        doc = to_otlp(_make_trace())
        attrs = doc["resourceSpans"][0]["resource"]["attributes"]
        by_key = {a["key"]: a["value"] for a in attrs}
        assert by_key["service.name"] == {"stringValue": "otel-test"}

    def test_original_trace_id_preserved_as_attribute(self):
        trace = _make_trace()
        doc = to_otlp(trace)
        attrs = doc["resourceSpans"][0]["resource"]["attributes"]
        by_key = {a["key"]: a["value"] for a in attrs}
        assert by_key["agent_replay.trace_id"] == {"stringValue": trace.trace_id}

    def test_trace_metadata_becomes_resource_attributes(self):
        trace = Trace(name="meta", metadata={"env": "prod", "run": 7})
        doc = to_otlp(trace)
        attrs = doc["resourceSpans"][0]["resource"]["attributes"]
        by_key = {a["key"]: a["value"] for a in attrs}
        assert by_key["env"] == {"stringValue": "prod"}
        assert by_key["run"] == {"intValue": "7"}

    def test_span_ids_are_otlp_lengths(self):
        doc = to_otlp(_make_trace())
        spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 2
        for span in spans:
            assert len(span["traceId"]) == 32
            assert len(span["spanId"]) == 16

    def test_span_times_are_nano_strings(self):
        trace = Trace(name="t")
        span = trace.add_span("s", start_time=10.0)
        span.end_time = 12.5
        doc = to_otlp(trace)
        otlp_span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otlp_span["startTimeUnixNano"] == "10000000000"
        assert otlp_span["endTimeUnixNano"] == "12500000000"

    def test_open_span_end_falls_back_to_start(self):
        trace = Trace(name="t")
        trace.add_span("open", start_time=5.0)
        doc = to_otlp(trace)
        otlp_span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otlp_span["endTimeUnixNano"] == otlp_span["startTimeUnixNano"]

    def test_parent_span_id_mapped(self):
        trace = Trace(name="t")
        parent = trace.add_span("parent")
        child = trace.add_span("child", parent_id=parent.span_id)
        doc = to_otlp(trace)
        spans = {s["name"]: s for s in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]}
        assert "parentSpanId" not in spans["parent"]
        assert spans["child"]["parentSpanId"] == spans["parent"]["spanId"]
        assert child.parent_id == parent.span_id

    def test_events_mapped_with_attributes(self):
        doc = to_otlp(_make_trace())
        spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
        first = spans[0]
        names = [e["name"] for e in first["events"]]
        assert names == ["llm_request", "llm_response"]
        req_attrs = {a["key"]: a["value"] for a in first["events"][0]["attributes"]}
        assert req_attrs["model"] == {"stringValue": "gpt-4"}
        assert "agent_replay.event_id" in req_attrs

    def test_event_times_are_nano_strings(self):
        doc = to_otlp(_make_trace())
        span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        for event in span["events"]:
            assert event["timeUnixNano"].isdigit()

    def test_error_event_sets_error_status(self):
        trace = Trace(name="t")
        span = trace.add_span("failing")
        span.add_event(EventType.ERROR, {"message": "boom"})
        doc = to_otlp(trace)
        otlp_span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        assert otlp_span["status"] == {"code": "STATUS_CODE_ERROR"}

    def test_clean_span_status_unset(self):
        doc = to_otlp(_make_trace())
        for span in doc["resourceSpans"][0]["scopeSpans"][0]["spans"]:
            assert span["status"] == {"code": "STATUS_CODE_UNSET"}

    def test_span_metadata_becomes_attributes(self):
        trace = Trace(name="t")
        span = Span(name="s", metadata={"agent": "planner", "step": 3})
        trace.spans.append(span)
        doc = to_otlp(trace)
        otlp_span = doc["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
        by_key = {a["key"]: a["value"] for a in otlp_span["attributes"]}
        assert by_key["agent"] == {"stringValue": "planner"}
        assert by_key["step"] == {"intValue": "3"}

    def test_empty_trace_is_valid(self):
        doc = to_otlp(Trace(name="empty"))
        assert doc["resourceSpans"][0]["scopeSpans"][0]["spans"] == []

    def test_document_is_json_serializable(self):
        doc = to_otlp(_make_trace())
        roundtrip = json.loads(json.dumps(doc))
        assert roundtrip == doc


class TestExportOtlp:
    def test_writes_valid_json_file(self, tmp_path: Path):
        trace = _make_trace()
        out = export_otlp(trace, tmp_path / "trace.otlp.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "resourceSpans" in data

    def test_accepts_string_path(self, tmp_path: Path):
        out = export_otlp(_make_trace(), str(tmp_path / "t.json"))
        assert isinstance(out, Path)
        assert out.exists()


class TestCliExportOtlp:
    def _save_trace(self, tmp_path: Path) -> Path:
        trace_file = tmp_path / "run.jsonl"
        _make_trace().save(trace_file)
        return trace_file

    def test_export_otlp_default_output(self, tmp_path: Path):
        trace_file = self._save_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["export", str(trace_file), "--format", "otlp"])
        assert result.exit_code == 0
        expected = tmp_path / "run.otlp.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["resourceSpans"][0]["scopeSpans"][0]["scope"]["name"] == "agent-replay"

    def test_export_otlp_custom_output(self, tmp_path: Path):
        trace_file = self._save_trace(tmp_path)
        out_file = tmp_path / "custom.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["export", str(trace_file), "--format", "otlp", "-o", str(out_file)]
        )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        spans = data["resourceSpans"][0]["scopeSpans"][0]["spans"]
        assert len(spans) == 2

    def test_export_rejects_unknown_format(self, tmp_path: Path):
        trace_file = self._save_trace(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["export", str(trace_file), "--format", "yaml"])
        assert result.exit_code != 0
