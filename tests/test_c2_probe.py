"""
c2_probe.py のテスト（実の C2 には触れない）

C2 の代わりに、loopback の http.server（偽の C2）を立てる。問い合わせの本文と、応答の本文には目印の文字列を使い、
画面と書き出したファイルに出ないことを確かめる。要求の形は、本物の C2Retriever / RecentC2Retriever が同じ偽サーバへ
送るものと一致することで固定する。
"""

from __future__ import annotations

import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_lounge import c2_probe, preflight, probe, retriever

_SENTINEL = "SENTINEL-c2probe-6a31"
_REPLY_SENTINEL = "SENTINEL-c2probe-reply-04d9"


class FakeC2:
    """偽の C2。受けた要求（道と引数）を記録する。"""

    def __init__(self, *, health: int = 200) -> None:
        self.requests: list[tuple[str, dict[str, list[str]]]] = []
        self.health = health
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server の規約
                parts = urlsplit(self.path)
                if parts.path == "/health":
                    self._reply(fake.health, {"status": "ok"})
                    return
                fake.requests.append((parts.path, parse_qs(parts.query)))
                self._reply(200, {"results": [{"event_id": "e1", "excerpt": _REPLY_SENTINEL, "score": 0.5,
                                               "type": "utterance"}]})

            def _reply(self, status: int, body: dict) -> None:
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def retrieval_requests(self) -> list[tuple[str, dict[str, list[str]]]]:
        return list(self.requests)


@pytest.fixture
def fake_c2():
    servers: list[FakeC2] = []

    def make(**kwargs) -> FakeC2:
        servers.append(FakeC2(**kwargs))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


def _write_queries(tmp_path, n=3):
    rows = [{"id": f"q{i + 1:02d}", "query": f"{_SENTINEL}-{i}", "shape": "short" if i % 2 == 0 else "long",
             "likely": "hit" if i % 2 == 0 else "miss"} for i in range(n)]
    path = tmp_path / "c2_queries.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def c2_env(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "prepare_env", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(probe, "_set_execution_state", lambda *, prevent: False)
    return tmp_path


def _latency(tmp_path, *extra, label="c2t"):
    return c2_probe.main(["latency", "--profile", "dev", "--label", label, "--queries", str(_write_queries(tmp_path)),
                          "--out-dir", str(tmp_path / "out"), "--no-host-compare", *extra])


class TestRequestShape:
    def test_matches_what_the_real_retrievers_send(self, fake_c2):
        server = fake_c2()
        retriever.C2Retriever(server.url, timeout_s=5).retrieve(f"{_SENTINEL}-x", top_k=15)
        retriever.RecentC2Retriever(server.url, stream_id=None, top_k=10, timeout_s=5).retrieve("x", top_k=15)
        (real_retrieve, real_recent) = server.requests

        path, params = c2_probe.request_for("retrieve", f"{_SENTINEL}-x", 15)
        assert real_retrieve == (path, {k: [str(v)] for k, v in params.items()})
        path, params = c2_probe.request_for("recent", "ignored", 15)
        assert real_recent == (path, {k: [str(v)] for k, v in params.items()})

    def test_recent_uses_the_larger_of_k_and_the_default(self, monkeypatch):
        assert c2_probe.request_for("recent", "x", 15)[1] == {"k": 15}
        monkeypatch.setenv("L2_C2_RECENT_TOP_K", "40")
        assert c2_probe.request_for("recent", "x", 15)[1] == {"k": 40}

    def test_unknown_kind(self):
        with pytest.raises(ValueError):
            c2_probe.request_for("write", "x", 1)

    def test_k_follows_the_composite_retriever(self, monkeypatch):
        assert c2_probe.sub_top_k() == 15  # max(5 * 3, 10)
        monkeypatch.setenv("L2_RAG_TOP_K", "2")
        assert c2_probe.sub_top_k() == 10

    def test_production_source_still_builds_requests_this_way(self):
        from pathlib import Path

        source = Path(retriever.__file__).read_text(encoding="utf-8")
        assert 'url = f"{self._base_url}/retrieve"' in source
        assert 'params: dict[str, str | int] = {"q": query, "k": top_k}' in source
        assert 'url = f"{self._base_url}/retrieve/recent"' in source
        assert '"k": max(top_k, self._default_top_k),' in source
        assert "with httpx.Client(timeout=self._timeout_s) as client:" in source
        graph_source = Path(retriever.__file__).with_name("graph.py").read_text(encoding="utf-8")
        assert 'c2_budget_s = float(os.environ.get("L2_C2_RETRIEVER_BUDGET_MS", "600")) / 1000' in graph_source
        memory = Path(retriever.__file__).with_name("mcp_servers").joinpath("retrieve_memory.py").read_text(encoding="utf-8")
        assert "sub_top_k = max(top_k * 3, 10)" in Path(retriever.__file__).read_text(encoding="utf-8")
        assert 'rag_top_k = int(os.environ.get("L2_RAG_TOP_K", str(top_k)))' in memory


class TestTargets:
    def test_defaults_and_readonly(self, monkeypatch):
        assert [(t.role, t.url) for t in c2_probe.targets_from_env()] == [("primary", "http://127.0.0.1:8100")]
        monkeypatch.setenv("L2_C2_URL", "http://127.0.0.1:8101/")
        monkeypatch.setenv("L2_C2_URL_READONLY", " http://127.0.0.1:8100 ")
        targets = c2_probe.targets_from_env()
        assert [t.label for t in targets] == ["dev(primary)", "prod(readonly)"]
        assert [t.shown for t in targets] == ["127.0.0.1:8101", "127.0.0.1:8100"]

    def test_shown_never_has_credentials_or_path(self):
        target = c2_probe.Target("primary", "http://user:secret@example.test:9000/prefix")
        assert "secret" not in target.shown and "user" not in target.shown and target.shown == "remote:9000"

    def test_with_host_swaps_only_the_name(self):
        assert c2_probe.Target("primary", "http://127.0.0.1:8100").with_host("localhost").url == "http://localhost:8100"

    def test_retriever_count_follows_the_graph(self, monkeypatch):
        assert c2_probe.c2_retriever_count() == 2  # C2Retriever + Recent（既定で有効）
        monkeypatch.setenv("L2_C2_URL_READONLY", "http://127.0.0.1:8100")
        assert c2_probe.c2_retriever_count() == 3
        monkeypatch.setenv("L2_USE_C2_RECENT", "false")
        assert c2_probe.c2_retriever_count() == 2


class TestQueries:
    def test_load(self, tmp_path):
        assert [q.id for q in c2_probe.load_queries(_write_queries(tmp_path))] == ["q01", "q02", "q03"]

    def test_bad_rows_report_only_line_numbers(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps({"id": "q01", "query": _SENTINEL, "shape": "tall", "likely": "hit"}) + "\n"
                        + "{" + _SENTINEL + "\n", encoding="utf-8")
        with pytest.raises(ValueError) as caught:
            c2_probe.load_queries(path)
        assert "1, 2 行目" in str(caught.value) and _SENTINEL not in str(caught.value)

    def test_missing_and_empty(self, tmp_path):
        with pytest.raises(ValueError, match="がありません"):
            c2_probe.load_queries(tmp_path / "none.jsonl")
        (tmp_path / "e.jsonl").write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="空です"):
            c2_probe.load_queries(tmp_path / "e.jsonl")


class TestSummary:
    def _s(self, ms, ok=True, pass_no=2):
        return c2_probe.Sample("g", "retrieve", "q", pass_no, ms, ok, None if ok else "ReadTimeout", 1 if ok else None)

    def test_percentiles_use_responses_and_thresholds_count_failures(self):
        samples = [self._s(100), self._s(250), self._s(700), self._s(10000, ok=False)]
        summary = c2_probe.summarize(samples)
        assert (summary["n"], summary["failed"], summary["max"]) == (4, 1, 700)
        assert summary["over"]["200"] == 0.75 and summary["over"]["600"] == 0.5 and summary["over"]["1500"] == 0.25
        assert summary["errors"] == {"ReadTimeout": 1}

    def test_phase_is_cold_only_on_the_first_pass(self):
        assert self._s(1, pass_no=1).phase == "cold" and self._s(1, pass_no=2).phase == "warm"

    @pytest.mark.parametrize("values, hint", [([120.0] * 100, 200), ([100.0] * 100, 100), ([650.0, 100.0], 700), ([], None)])
    def test_budget_hint_rounds_p99_up_to_100(self, values, hint):
        assert c2_probe.budget_hint(values)[0] == hint


class TestLatencyCommand:
    def test_sends_reads_only_and_records_numbers(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)

        assert _latency(c2_env, "--passes", "2") == 0

        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err and _REPLY_SENTINEL not in captured.out + captured.err
        # 3 問い合わせ × 2 周 × (retrieve + recent)
        assert sorted({path for path, _ in server.requests}) == ["/retrieve", "/retrieve/recent"]
        assert len(server.requests) == 12
        assert all(params["k"] == ["15"] for _path, params in server.requests)
        out = c2_env / "out"
        lines = [json.loads(line) for line in (out / "c2_latency_c2t.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 12 and all(line["ok"] and line["results"] == 1 for line in lines)
        assert {line["query_id"] for line in lines} == {"q01", "q02", "q03"}
        meta = json.loads((out / "c2_latency_c2t.meta.json").read_text(encoding="utf-8"))
        assert meta["queries"]["n"] == 3 and meta["embedding_calls_estimate"] == 6 and meta["passes"] == 2
        for text in ((out / "c2_latency_c2t.jsonl").read_text(encoding="utf-8"), json.dumps(meta)):
            assert _SENTINEL not in text and _REPLY_SENTINEL not in text and server.url not in text
        assert "cold" in captured.out and "warm" in captured.out and "予算の目安" in captured.out
        assert "超えた割合" in captured.out and "600:" in captured.out

    def test_unhealthy_target_stops_before_sending_anything(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2(health=503)
        monkeypatch.setenv("L2_C2_URL", server.url)
        assert _latency(c2_env) == c2_probe.EXIT_MISSING_SERVICE == 3
        out = capsys.readouterr().out
        assert "HTTP-503" in out and f"127.0.0.1:{server.port}" in out and "起動しません" in out
        assert server.requests == [] and not (c2_env / "out" / "c2_latency_c2t.jsonl").exists()

    def test_unreachable_target_stops_with_only_the_error_type(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        port = server.port
        server.close()
        monkeypatch.setenv("L2_C2_URL", f"http://user:secret@127.0.0.1:{port}")
        assert _latency(c2_env) == 3
        out = capsys.readouterr().out
        assert "secret" not in out and "Connect" in out  # 閉じたポートは、OS により ConnectError か ConnectTimeout

    def test_one_unhealthy_readonly_target_also_stops_everything(self, c2_env, fake_c2, monkeypatch):
        primary, readonly = fake_c2(), fake_c2(health=503)
        monkeypatch.setenv("L2_C2_URL", primary.url)
        monkeypatch.setenv("L2_C2_URL_READONLY", readonly.url)
        assert _latency(c2_env) == 3 and primary.requests == []

    def test_readonly_target_gets_retrieve_only(self, c2_env, fake_c2, monkeypatch):
        primary, readonly = fake_c2(), fake_c2()
        monkeypatch.setenv("L2_C2_URL", primary.url)
        monkeypatch.setenv("L2_C2_URL_READONLY", readonly.url)
        assert _latency(c2_env, "--passes", "1") == 0
        assert {path for path, _ in readonly.requests} == {"/retrieve"} and len(readonly.requests) == 3
        assert {path for path, _ in primary.requests} == {"/retrieve", "/retrieve/recent"}

    def test_targets_option_selects_roles(self, c2_env, fake_c2, monkeypatch):
        primary, readonly = fake_c2(), fake_c2()
        monkeypatch.setenv("L2_C2_URL", primary.url)
        monkeypatch.setenv("L2_C2_URL_READONLY", readonly.url)
        assert _latency(c2_env, "--passes", "1", "--targets", "readonly") == 0
        assert primary.requests == [] and len(readonly.requests) == 3

    def test_no_such_target_is_a_usage_error(self, c2_env, fake_c2, monkeypatch):
        monkeypatch.setenv("L2_C2_URL", fake_c2().url)
        assert _latency(c2_env, "--targets", "readonly") == c2_probe.EXIT_USAGE

    def test_slow_responses_are_cut_by_the_timeout_and_counted(self, c2_env, monkeypatch, capsys):
        server = FakeC2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        original = c2_probe.get_once

        def slow(target, kind, query, k, timeout_sec, group, pass_no):
            if kind == "retrieve":
                import httpx
                return c2_probe.Sample(group, kind, query.id, pass_no, timeout_sec * 1000, False,
                                       httpx.ReadTimeout.__name__, None)
            return original(target, kind, query, k, timeout_sec, group, pass_no)

        monkeypatch.setattr(c2_probe, "get_once", slow)
        try:
            assert _latency(c2_env, "--passes", "1") == 0
        finally:
            server.close()
        assert "失敗=3" in capsys.readouterr().out

    def test_existing_label_and_missing_queries_are_refused(self, c2_env, fake_c2, monkeypatch, capsys):
        monkeypatch.setenv("L2_C2_URL", fake_c2().url)
        assert _latency(c2_env, "--passes", "1") == 0
        assert _latency(c2_env, "--passes", "1") == c2_probe.EXIT_REFUSED
        code = c2_probe.main(["latency", "--profile", "dev", "--label", "z", "--out-dir", str(c2_env / "o"),
                              "--queries", str(c2_env / "none.jsonl")])
        assert code == c2_probe.EXIT_REFUSED and "断りました" in capsys.readouterr().out

    def test_bad_arguments(self):
        base = ["latency", "--profile", "dev", "--label", "x"]
        assert c2_probe.main([*base, "--passes", "0"]) == c2_probe.EXIT_USAGE
        assert c2_probe.main(["latency", "--profile", "dev", "--label", "a b"]) == c2_probe.EXIT_USAGE

    def test_host_compare_measures_only_loopback_targets_once_per_name(self, monkeypatch):
        calls = []
        monkeypatch.setattr(c2_probe, "get_once", lambda target, kind, query, k, timeout, group, pass_no:
                            calls.append((target.host, group, pass_no)) or c2_probe.Sample(group, kind, query.id, pass_no,
                                                                                         1.0, True, None, 0))
        queries = [c2_probe.Query("q01", "x", "short", "hit")]
        c2_probe._compare_hosts([c2_probe.Target("primary", "http://127.0.0.1:8100"),
                                 c2_probe.Target("readonly", "http://example.test:8100")],
                                queries, 15, 1.0, lambda sample: None)
        assert calls == [("127.0.0.1", "prod(primary)@127.0.0.1", 1), ("localhost", "prod(primary)@localhost", 1)]


# ─── timeouts ────────────────────────────────────────────────────

def _log_line(message: str, name: str = "lab_lounge.retriever") -> str:
    return f"2026-06-01 12:00:00,123 WARNING  {name}: {message}\n"


def _runs(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    marker = _log_line("Agent LLM step 1 完了 [character=mimi] latency_ms=10", "lab_lounge.graph")

    def write(name, lines):
        (runs / name).write_text(marker + "".join(lines), encoding="utf-8")

    search = _log_line(f"記憶検索実行: '{_SENTINEL}' (top_k=5, stream_id=s)", "lab_lounge.mcp_servers.retrieve_memory")
    write("run_loop_20261020_120000.log", [
        search, search, search, search,
        _log_line("C2Retriever タイムアウト (timeout_s=0.6)"), _log_line("C2Retriever タイムアウト (timeout_s=0.6)"),
        _log_line("RecentC2Retriever タイムアウト (timeout_s=0.6)"),
        _log_line(f"C2Retriever HTTP エラー: {_SENTINEL}"),
        "続きの行（行頭の形なし）C2Retriever タイムアウト (timeout_s=9.9)\n",
    ])
    write("run_loop_20261030_235959.log", [search, _log_line("C2Retriever タイムアウト (timeout_s=0.6)")])
    write("run_loop_20261102_120000.log", [search, search, _log_line("C2Retriever タイムアウト (timeout_s=1.5)")])
    # 使わない: pytest の残り（名前の規則）
    (runs / "run_loop_20261031_100000.log").write_text(
        marker + _log_line("C2Retriever タイムアウト (timeout_s=0.6)") * 50, encoding="utf-8")
    return runs


class TestTimeouts:
    def test_counts_by_month_and_around_the_cutoff(self, tmp_path):
        paths, _, _ = probe.select_run_logs(_runs(tmp_path), date(2026, 5, 8))
        stats = c2_probe.timeouts_stats(paths)
        assert stats["month:2026-10"] == {"tool_calls": 5, "timeout_c2": 3, "timeout_c2_0.6s": 2 + 1,
                                          "timeout_recent": 1, "timeout_recent_0.6s": 1, "http_error": 1}
        assert stats["month:2026-11"] == {"tool_calls": 2, "timeout_c2": 1, "timeout_c2_1.5s": 1}
        assert stats["side:before"]["tool_calls"] == 5 and stats["side:before"]["timeout_c2"] == 3
        assert stats["side:after"] == {"tool_calls": 2, "timeout_c2": 1, "timeout_c2_1.5s": 1}

    def test_command_prints_counts_and_the_note_and_writes_json(self, tmp_path, capsys):
        runs = _runs(tmp_path)
        out_dir = tmp_path / "out"
        assert c2_probe.main(["timeouts", "--label", "t", "--logs-dir", str(runs), "--out-dir", str(out_dir)]) == 0
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err
        assert "2026-10-31 より前" in captured.out and "2026-10-31 以降" in captured.out
        assert "分母" in captured.out and "0.6s=" in captured.out
        saved = json.loads((out_dir / "c2_timeouts_t.json").read_text(encoding="utf-8"))
        assert _SENTINEL not in json.dumps(saved) and saved["cutoff_day"] == "2026-10-31" and "分母" in saved["note"]
        assert saved["files"]["used"] == 3

    def test_no_logs_is_a_failure(self, tmp_path):
        (tmp_path / "runs").mkdir()
        assert c2_probe.main(["timeouts", "--logs-dir", str(tmp_path / "runs"), "--out-dir", str(tmp_path)]) == 1
        assert c2_probe.main(["timeouts", "--logs-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path)]) == 1

    def test_the_retriever_log_formats_still_exist(self):
        from pathlib import Path

        source = Path(retriever.__file__).read_text(encoding="utf-8")
        assert 'logger.warning("C2Retriever タイムアウト (timeout_s=%s)", self._timeout_s)' in source
        assert 'logger.warning("RecentC2Retriever タイムアウト (timeout_s=%s)", self._timeout_s)' in source
        memory = Path(retriever.__file__).with_name("mcp_servers").joinpath("retrieve_memory.py").read_text(encoding="utf-8")
        assert '"記憶検索実行: %r (top_k=%d, stream_id=%s)"' in memory


class TestQueryInputErrors:
    def test_a_file_that_is_not_utf8_is_reported_by_name_only(self, tmp_path):
        path = tmp_path / "c2_queries.jsonl"
        path.write_bytes(_SENTINEL.encode() + bytes([0xFF, 0xFE]))
        with pytest.raises(probe.MeasureInputError) as caught:
            c2_probe.load_queries(path)
        assert "c2_queries.jsonl" in str(caught.value) and _SENTINEL not in str(caught.value)

    def test_separators_inside_a_query_do_not_split_a_row(self, tmp_path):
        row = {"id": "q01", "query": "深海" + chr(0x2028) + "魚", "shape": "short", "likely": "hit"}
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + chr(10), encoding="utf-8")
        assert c2_probe.load_queries(path)[0].text == row["query"]

    def test_bad_queries_are_refused_without_printing_other_errors(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(preflight, "prepare_env", lambda *_a, **_k: [])

        def boom(_path=None):
            raise ValueError(_SENTINEL)

        monkeypatch.setattr(c2_probe, "load_queries", boom)
        with pytest.raises(ValueError):
            c2_probe.main(["latency", "--profile", "dev", "--label", "z", "--out-dir", str(tmp_path)])
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err


class TestTimeoutsIncludeRunsThatNeverAnswered:
    def test_a_run_without_any_agent_answer_is_counted_and_noted(self, tmp_path, capsys):
        runs = _runs(tmp_path)
        # Agent が 1 度も答えなかった回（最も悪い回）。タイムアウトの行だけがある。
        (runs / "run_loop_20261025_090000.log").write_text(
            _log_line("C2Retriever タイムアウト (timeout_s=0.6)") * 4, encoding="utf-8")
        out_dir = tmp_path / "out"
        assert c2_probe.main(["timeouts", "--label", "t", "--logs-dir", str(runs), "--out-dir", str(out_dir)]) == 0
        out = capsys.readouterr().out
        assert "注: Agent の応答が 1 度も無かった回 1 本を含みます" in out
        saved = json.loads((out_dir / "c2_timeouts_t.json").read_text(encoding="utf-8"))
        assert saved["counts"]["side:before"]["timeout_c2"] == 3 + 4
        assert any("含みます" in note for note in saved["selection_notes"])


class TestBudgetHintWithFailures:
    def test_failures_count_as_the_full_timeout(self):
        # 200 件中 1 件の失敗（0.5%）: p99 は 200 件目に近い。失敗は 10000 ms として並ぶ。
        hint, reason = c2_probe.budget_hint([120.0] * 199, failed=1, timeout_ms=10000)
        assert reason is None and hint == 200  # 99% 点（198 番目）までは 120 ms

    def test_exactly_one_percent_failures_still_gives_a_hint_and_the_failure_is_the_tail(self):
        # 100 件中 1 件（ちょうど 1%）: 目安は出す。失敗の 1 件は p99 の外（最後の 1%）に並ぶ。
        hint, reason = c2_probe.budget_hint([120.0] * 99, failed=1, timeout_ms=10000)
        assert (hint, reason) == (200, None)
        # 失敗が p99 の内側に入る件数（2 件 / 150 = 1.3%）は、目安を出さない側へ倒れる。
        assert c2_probe.budget_hint([120.0] * 148, failed=2, timeout_ms=10000)[0] is None

    def test_more_than_one_percent_failures_gives_no_hint_and_says_why(self):
        hint, reason = c2_probe.budget_hint([120.0] * 97, failed=3, timeout_ms=10000)
        assert hint is None and "1% を超えた" in reason and "3 件" in reason

    def test_nothing_measured(self):
        assert c2_probe.budget_hint([]) == (None, None)


class TestHostCompareIsNotCold:
    def test_phase_names(self):
        def sample(group, pass_no):
            return c2_probe.Sample(group, "retrieve", "q", pass_no, 1.0, True, None, 1)

        assert sample("prod(primary)", 1).phase == "cold" and sample("prod(primary)", 2).phase == "warm"
        assert sample("prod(primary)@localhost", 1).phase == "host_compare"

    def test_the_report_prints_it_under_its_own_name(self, capsys):
        samples = [c2_probe.Sample("dev(primary)@localhost", "retrieve", "q", 1, 5.0, True, None, 1),
                   c2_probe.Sample("dev(primary)", "retrieve", "q", 1, 5.0, True, None, 1)]
        summaries, hints = c2_probe.report_latency(samples, 10000.0, 2)
        out = capsys.readouterr().out
        assert "dev(primary)@localhost retrieve host_compare" in out and "@localhost retrieve cold" not in out
        assert "dev(primary)@localhost/retrieve/host_compare" in summaries and hints == {}


class TestLatencyMetaOnStop:
    def test_meta_is_written_first_and_updated_with_the_partial_summary(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        out_dir = c2_env / "out"
        seen = {}
        original = c2_probe.get_once
        calls = []

        def interrupting(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                seen["first"] = json.loads((out_dir / "c2_latency_c2t.meta.json").read_text(encoding="utf-8"))["status"]
            if len(calls) == 4:
                raise KeyboardInterrupt
            return original(*args, **kwargs)

        monkeypatch.setattr(c2_probe, "get_once", interrupting)
        with pytest.raises(KeyboardInterrupt):
            _latency(c2_env, "--passes", "2")
        assert seen["first"] == "running"
        meta = json.loads((out_dir / "c2_latency_c2t.meta.json").read_text(encoding="utf-8"))
        assert meta["status"] == "aborted" and meta["finished"] and meta["summary"]
        assert meta["embedding_calls_estimate"] == 2 and "途中までの集計" in capsys.readouterr().out
        assert _latency(c2_env, "--passes", "1") == c2_probe.EXIT_REFUSED  # 同じ名前は、止まった回でも断る

    def test_the_first_request_in_the_process_is_marked(self, c2_env, fake_c2, monkeypatch):
        monkeypatch.setenv("L2_C2_URL", fake_c2().url)
        assert _latency(c2_env, "--passes", "1") == 0
        lines = [json.loads(line) for line in
                 (c2_env / "out" / "c2_latency_c2t.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [x["first_in_process"] for x in lines] == [True] + [False] * (len(lines) - 1)
        assert {x["phase"] for x in lines} == {"cold"}


class TestRemoteHostsAreNotShown:
    def test_loopback_is_shown_and_others_are_just_remote(self):
        local = c2_probe.Target("primary", "http://127.0.0.1:8100")
        remote = c2_probe.Target("readonly", "http://sensitive-internal-name.example:8100")
        assert (local.shown, local.shown_host) == ("127.0.0.1:8100", "127.0.0.1")
        assert (remote.shown, remote.shown_host) == ("remote:8100", "remote")
        assert remote.label == "prod(readonly)"  # 役割とポートの呼び名は残る

    def test_the_screen_and_meta_never_show_a_remote_host_name(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        monkeypatch.setenv("L2_C2_URL_READONLY", "http://sensitive-internal-name.example:8100")
        monkeypatch.setattr(c2_probe, "check_health", lambda target, timeout_sec=2.0: (
            "ConnectTimeout" if not target.is_loopback else None))
        assert _latency(c2_env) == 3
        out = capsys.readouterr().out
        assert "sensitive-internal-name" not in out and "remote:8100" in out

    def test_meta_has_remote_for_a_non_loopback_target(self, c2_env, monkeypatch):
        remote = FakeC2()
        try:
            monkeypatch.setenv("L2_C2_URL", remote.url)
            monkeypatch.setattr(c2_probe.Target, "is_loopback", property(lambda self: False))
            assert _latency(c2_env, "--passes", "1") == 0
        finally:
            remote.close()
        meta = json.loads((c2_env / "out" / "c2_latency_c2t.meta.json").read_text(encoding="utf-8"))
        assert [t["host"] for t in meta["targets"]] == ["remote"]


class TestProductionSettings:
    def test_recent_is_not_measured_when_production_does_not_call_it(self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        monkeypatch.setenv("L2_USE_C2_RECENT", "false")
        assert _latency(c2_env, "--passes", "1") == 0
        assert {path for path, _ in server.requests} == {"/retrieve"}
        out = capsys.readouterr().out
        assert "L2_USE_C2_RECENT が true でない" in out
        meta = json.loads((c2_env / "out" / "c2_latency_c2t.meta.json").read_text(encoding="utf-8"))
        assert meta["recent_measured"] is False and "L2_USE_C2_RECENT" in meta["recent_skipped_reason"]

    def test_recent_with_session_scope_is_not_measured_because_production_adds_a_stream_id(
            self, c2_env, fake_c2, monkeypatch, capsys):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        monkeypatch.setenv("L2_C2_RECENT_SCOPE", "Session ")
        assert _latency(c2_env, "--passes", "1") == 0
        assert {path for path, _ in server.requests} == {"/retrieve"}
        assert "session" in capsys.readouterr().out

    def test_recent_is_measured_with_the_default_global_scope(self, c2_env, fake_c2, monkeypatch):
        server = fake_c2()
        monkeypatch.setenv("L2_C2_URL", server.url)
        assert _latency(c2_env, "--passes", "1") == 0
        assert {path for path, _ in server.requests} == {"/retrieve", "/retrieve/recent"}

    def test_warns_when_production_does_not_use_c2_at_all(self, c2_env, fake_c2, monkeypatch, capsys):
        monkeypatch.setenv("L2_C2_URL", fake_c2().url)
        assert _latency(c2_env, "--passes", "1") == 0
        assert "L2_USE_C2_RETRIEVER が true でない" in capsys.readouterr().out
        monkeypatch.setenv("L2_USE_C2_RETRIEVER", "true")
        assert _latency(c2_env, "--passes", "1", label="c2u") == 0
        assert "L2_USE_C2_RETRIEVER" not in capsys.readouterr().out

    def test_the_rules_follow_the_production_source(self):
        from pathlib import Path

        graph_source = Path(retriever.__file__).with_name("graph.py").read_text(encoding="utf-8")
        assert 'use_c2 = os.environ.get("L2_USE_C2_RETRIEVER", "false").lower() == "true"' in graph_source
        assert 'use_recent = os.environ.get("L2_USE_C2_RECENT", "true").lower() == "true"' in graph_source
        assert 'recent_scope = os.environ.get("L2_C2_RECENT_SCOPE", "global").strip().lower()' in graph_source
        assert 'if recent_scope == "session":' in graph_source
