"""
spend.py（価格表、支出の記録、上限、HTTP の層の usage の写し、本文を出さない後始末）のテスト

HTTP は httpx.MockTransport で返し、ネットへは出ない。本文には目印の文字列（_SENTINEL）だけを使い、
支出の記録、画面、例外の文言に出ないことを確かめる。本物の SDK を通す確かめは
tests/contract/test_contract_spend.py にある。
"""

from __future__ import annotations

import ast
import gzip
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from lab_lounge import spend

_SENTINEL = "SENTINEL-spend-unit-91c2"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANTHROPIC = "https://api.anthropic.com/v1/messages"
_HAIKU = spend.PRICES["claude-haiku-4-5"]


def _anthropic_reply(input_tokens: int = 100, output_tokens: int = 10, **extra) -> dict:
    return {"model": "claude-haiku-4-5-20251001", "content": [{"type": "text", "text": _SENTINEL}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, **extra}}


def _body(**extra) -> dict:
    return {"model": "claude-haiku-4-5", "max_tokens": 20,
            "messages": [{"role": "user", "content": _SENTINEL}], **extra}


class _Chunks(httpx.SyncByteStream):
    """読むまで中身を出さない応答の本文（SDK の stream=True と同じ形）。"""

    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _calls(directory: Path) -> list[dict]:
    path = directory / "calls.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def _assert_no_text(directory: Path) -> None:
    for path in directory.iterdir():
        assert _SENTINEL not in path.read_text(encoding="utf-8"), path.name


@pytest.fixture
def ledger(tmp_path) -> spend.SpendLedger:
    return spend.SpendLedger(tmp_path)


# ─── 価格と usage ────────────────────────────────────────────────


class TestUsage:
    """3 社の usage の形を、共通の Usage にそろえる。"""

    def test_anthropic_cache_is_separate_from_input(self):
        raw = {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100,
               "cache_creation_input_tokens": 30, "cache_creation": {"ephemeral_1h_input_tokens": 20}}
        assert spend.normalize_usage("anthropic", raw) == spend.Usage(
            input=10, output=5, cache_read=100, cache_write_5m=10, cache_write_1h=20)

    def test_openai_chat_subtracts_cached_and_keeps_reasoning_in_output(self):
        raw = {"prompt_tokens": 100, "completion_tokens": 40, "prompt_tokens_details": {"cached_tokens": 60},
               "completion_tokens_details": {"reasoning_tokens": 30}}
        assert spend.normalize_usage("openai", raw) == spend.Usage(input=40, output=40, reasoning=30, cache_read=60)

    def test_openai_responses_shape(self):
        raw = {"input_tokens": 50, "output_tokens": 7, "input_tokens_details": {"cached_tokens": 10},
               "output_tokens_details": {"reasoning_tokens": 2}}
        assert spend.normalize_usage("openai", raw) == spend.Usage(input=40, output=7, reasoning=2, cache_read=10)

    def test_google_adds_thoughts_to_output_and_tool_prompt_to_input(self):
        raw = {"promptTokenCount": 100, "candidatesTokenCount": 5, "thoughtsTokenCount": 20,
               "cachedContentTokenCount": 30, "toolUsePromptTokenCount": 8}
        assert spend.normalize_usage("google", raw) == spend.Usage(input=78, output=25, reasoning=20, cache_read=30)

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "google"])
    def test_no_known_keys_is_none(self, provider):
        assert spend.normalize_usage(provider, {"total": 3}) is None

    def test_non_numbers_count_as_zero(self):
        raw = {"input_tokens": "9", "output_tokens": True}
        assert spend.normalize_usage("anthropic", raw) == spend.Usage()


class TestPrices:
    def test_every_price_names_its_official_source_and_date(self):
        for model, price in spend.PRICES.items():
            assert price.source_url.startswith("https://"), model
            assert len(price.as_of) == 10, model
            assert price.provider in ("anthropic", "openai", "google"), model

    def test_cost_counts_each_kind_at_its_rate(self):
        usage = spend.Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000,
                            cache_write_5m=1_000_000, cache_write_1h=1_000_000)
        assert spend.cost_usd(_HAIKU, usage) == pytest.approx(1.00 + 5.00 + 0.10 + 1.25 + 2.00)

    def test_long_prompts_use_the_long_rate(self):
        price = spend.PRICES["gpt-5.5"]
        short = spend.cost_usd(price, spend.Usage(input=272_000))
        long = spend.cost_usd(price, spend.Usage(input=200_000, cache_read=72_001))
        assert short == pytest.approx(272_000 * 5.00 / 1e6)
        assert long == pytest.approx((200_000 * 10.00 + 72_001 * 1.00) / 1e6)

    def test_estimate_is_above_the_real_cost(self):
        body = json.dumps(_body()).encode()
        real = spend.cost_usd(_HAIKU, spend.Usage(input=len(body) // 4, output=20))
        assert spend.estimate_usd(_HAIKU, len(body), 20) > real

    @pytest.mark.parametrize("name", ["claude-haiku-4-5", "claude-haiku-4-5-20251001", "models/gemini-3.1-flash-lite"])
    def test_lookup_strips_dates_and_prefixes(self, name):
        assert spend.lookup_price(name, spend.PRICES).provider in ("anthropic", "google")

    @pytest.mark.parametrize("name", [None, "", "claude-haiku-9"])
    def test_lookup_refuses_unknown_models(self, name):
        with pytest.raises(spend.UnpricedModel):
            spend.lookup_price(name, spend.PRICES)

    def test_unpriced_message_hides_text_that_is_not_a_model_name(self):
        with pytest.raises(spend.UnpricedModel) as caught:
            spend.lookup_price(f"{_SENTINEL} です", spend.PRICES)
        assert _SENTINEL not in str(caught.value)

    def test_price_override(self):
        name, price = spend.parse_price_override("gpt-9=1,2,0.5")
        assert (name, price.rate.input, price.rate.output, price.rate.cache_read) == ("gpt-9", 1, 2, 0.5)
        assert price.provider == "openai"

    @pytest.mark.parametrize("text", ["gpt-9", "gpt-9=1", "gpt-9=-1,2", "a b=1,2", "gpt-9=1,2,3,4,5"])
    def test_price_override_rejects_bad_shapes(self, text):
        with pytest.raises(ValueError):
            spend.parse_price_override(text)


# ─── 包み（同期、読み終えた応答） ────────────────────────────────


class TestMeter:
    def test_counts_usage_and_keeps_no_text(self, ledger, tmp_path, capsys):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=_anthropic_reply())

        with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter, _client(handler) as client:
            client.post(_ANTHROPIC, json=_body())

        assert len(seen) == 1
        assert (meter.stats.sent, meter.stats.counted) == (1, 1)
        (call,) = _calls(tmp_path)
        assert call["status"] == "usage" and call["model_on_wire"] == "claude-haiku-4-5-20251001"
        assert call["usd"] == pytest.approx((100 * 1.00 + 10 * 5.00) / 1e6)
        assert ledger.data["by_run"]["unit"]["calls"] == 1
        assert not (tmp_path / "spent.lock").exists()
        _assert_no_text(tmp_path)
        captured = capsys.readouterr()
        assert _SENTINEL not in captured.out + captured.err

    def test_send_is_restored_even_after_an_error(self, ledger):
        before = (httpx.Client.__dict__["send"], httpx.AsyncClient.__dict__["send"])
        with pytest.raises(ZeroDivisionError), spend.UsageMeter(ledger, "unit", max_usd=1.0):
            assert httpx.Client.__dict__["send"] is not before[0]
            raise ZeroDivisionError
        assert (httpx.Client.__dict__["send"], httpx.AsyncClient.__dict__["send"]) == before

    def test_requests_outside_the_meter_are_untouched(self, ledger, tmp_path):
        with spend.UsageMeter(ledger, "unit", max_usd=1.0):
            pass
        with _client(lambda r: httpx.Response(200, json=_anthropic_reply())) as client:
            client.post(_ANTHROPIC, json=_body())
        assert _calls(tmp_path) == []

    @pytest.mark.parametrize(("method", "url"), [
        ("POST", "https://c2.invalid/retrieve"),
        ("GET", "https://api.anthropic.com/v1/models"),
        ("POST", "https://tavily.invalid/search"),
    ])
    def test_other_requests_pass_through_uncounted(self, ledger, tmp_path, method, url):
        with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter, \
                _client(lambda r: httpx.Response(200, json={"usage": {"input_tokens": 1}})) as client:
            client.request(method, url, json={"q": _SENTINEL})
        assert meter.stats.sent == 0
        assert _calls(tmp_path) == []

    def test_unknown_route_on_a_provider_host_is_not_sent(self, ledger):
        seen = []
        with spend.UsageMeter(ledger, "unit", max_usd=1.0), _client(lambda r: seen.append(r)) as client, \
                pytest.raises(spend.UnpricedModel):
            client.post("https://api.anthropic.com/v1/messages/batches", json=_body())
        assert seen == []

    def test_unpriced_model_is_not_sent(self, ledger):
        seen = []
        with spend.UsageMeter(ledger, "unit", max_usd=1.0), _client(lambda r: seen.append(r)) as client, \
                pytest.raises(spend.UnpricedModel):
            client.post(_ANTHROPIC, json=_body(model="claude-unknown-1"))
        assert seen == []

    def test_gemini_model_is_read_from_the_path(self, ledger, tmp_path):
        reply = {"candidates": [], "modelVersion": "gemini-3.1-flash-lite",
                 "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 4}}
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent"
        with spend.UsageMeter(ledger, "unit", max_usd=1.0), \
                _client(lambda r: httpx.Response(200, json=reply)) as client:
            client.post(url, json={"contents": [{"parts": [{"text": _SENTINEL}]}]})
        (call,) = _calls(tmp_path)
        assert call["model"] == "gemini-3.1-flash-lite"
        assert call["tokens"]["input"] == 40

    def test_http_error_costs_nothing(self, ledger, tmp_path):
        reply = {"type": "error", "error": {"message": _SENTINEL}}
        with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter, \
                _client(lambda r: httpx.Response(400, json=reply)) as client:
            client.post(_ANTHROPIC, json=_body())
        assert meter.stats.http_errors == 1
        assert [c["status"] for c in _calls(tmp_path)] == ["http_error"]
        assert ledger.total_usd == 0
        _assert_no_text(tmp_path)

    def test_transport_error_is_charged_at_the_estimate(self, ledger, tmp_path):
        def handler(request):
            raise httpx.ConnectError(_SENTINEL)

        with spend.UsageMeter(ledger, "unit", max_usd=1.0), _client(handler) as client, \
                pytest.raises(httpx.ConnectError):
            client.post(_ANTHROPIC, json=_body())
        (call,) = _calls(tmp_path)
        assert call["status"] == "transport_error"
        assert call["usd"] > 0
        _assert_no_text(tmp_path)

    def test_repeated_ok_without_usage_stops_before_the_next_send(self, ledger, tmp_path):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"content": _SENTINEL})

        with spend.UsageMeter(ledger, "unit", max_usd=1.0), _client(handler) as client:
            for _ in range(3):
                client.post(_ANTHROPIC, json=_body())
            with pytest.raises(spend.UsageCaptureLost):
                client.post(_ANTHROPIC, json=_body())
        assert len(seen) == 3
        assert {c["status"] for c in _calls(tmp_path)} == {"no_usage"}
        assert ledger.total_usd > 0  # 拾えなかった分は見積もりで数える

    def test_check_capture_stops_when_nothing_was_seen(self, ledger):
        with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter:
            meter.check_capture(0)
            with pytest.raises(spend.UsageCaptureLost):
                meter.check_capture(5)

    def test_failure_inside_the_meter_is_not_an_exception_the_sdk_retries(self, ledger, monkeypatch):
        """支出の記録が書けないとき、SDK の except Exception（再試行）に入らない型で止まる。"""
        def broken(*_args, **_kwargs):
            raise OSError(_SENTINEL)

        with spend.UsageMeter(ledger, "unit", max_usd=1.0), \
                _client(lambda r: httpx.Response(200, json=_anthropic_reply())) as client:
            monkeypatch.setattr(ledger, "add", broken)
            with pytest.raises(spend.MeterFailure) as caught:
                client.post(_ANTHROPIC, json=_body())
        assert not isinstance(caught.value, Exception)
        assert _SENTINEL not in str(caught.value)


# ─── 上限 ────────────────────────────────────────────────────────


class TestBudget:
    def test_stops_before_sending(self, ledger):
        seen = []
        with spend.UsageMeter(ledger, "unit", max_usd=1e-6), _client(lambda r: seen.append(r)) as client, \
                pytest.raises(spend.BudgetExceeded):
            client.post(_ANTHROPIC, json=_body())
        assert seen == []

    def test_budget_stop_escapes_except_exception(self):
        """L2 の判定の関数は except Exception で失敗を握る。止める例外はそこを抜ける。"""
        def swallow():
            try:
                raise spend.BudgetExceeded("x")
            except Exception:  # noqa: BLE001 - router と同じ形
                return None

        with pytest.raises(spend.BudgetExceeded):
            swallow()

    def test_run_budget_counts_earlier_spend_under_the_same_name(self, tmp_path):
        handler = lambda r: httpx.Response(200, json=_anthropic_reply(input_tokens=1000, output_tokens=20))  # noqa: E731
        with spend.UsageMeter(spend.SpendLedger(tmp_path), "same", max_usd=1.0), _client(handler) as client:
            client.post(_ANTHROPIC, json=_body())
        reopened = spend.SpendLedger(tmp_path)
        spent = reopened.run_usd("same")
        assert spent > 0
        with spend.UsageMeter(reopened, "same", max_usd=spent), _client(handler) as client, \
                pytest.raises(spend.BudgetExceeded):
            client.post(_ANTHROPIC, json=_body())

    def test_total_cap_in_the_file_wins(self, tmp_path):
        ledger = spend.SpendLedger(tmp_path)
        ledger.data["total_cap_usd"] = 1e-6
        ledger._write()
        with spend.UsageMeter(spend.SpendLedger(tmp_path), "unit", max_usd=5.0), \
                _client(lambda r: httpx.Response(200, json=_anthropic_reply())) as client, \
                pytest.raises(spend.BudgetExceeded) as caught:
            client.post(_ANTHROPIC, json=_body())
        assert "全体" in str(caught.value)

    def test_missing_max_tokens_assumes_a_large_output(self, ledger):
        body = _body()
        del body["max_tokens"]
        limit = spend.estimate_usd(_HAIKU, len(json.dumps(body)), 1000)
        with spend.UsageMeter(ledger, "unit", max_usd=limit, assumed_max_output=8192), \
                _client(lambda r: httpx.Response(200, json=_anthropic_reply())) as client, \
                pytest.raises(spend.BudgetExceeded):
            client.post(_ANTHROPIC, json=body)

    def test_override_prices_a_new_model_and_is_recorded(self, ledger, tmp_path):
        name, price = spend.parse_price_override("claude-new-1=1,5")
        with spend.UsageMeter(ledger, "unit", max_usd=1.0, overrides={name: price}), \
                _client(lambda r: httpx.Response(200, json=_anthropic_reply())) as client:
            client.post(_ANTHROPIC, json=_body(model=name))
        assert [o["model"] for o in ledger.data["price_overrides"]] == [name]
        assert _calls(tmp_path)[0]["model"] == name


# ─── stream の応答 ───────────────────────────────────────────────


class TestStream:
    def _sse(self, *events: dict) -> bytes:
        return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)

    def _send_stream(self, ledger, payload: bytes, content_type: str, *, url: str = _ANTHROPIC,
                     body: dict | None = None, read: bool = True, headers: dict | None = None):
        stream = _Chunks(payload[: len(payload) // 2], payload[len(payload) // 2:])

        def handler(request):
            return httpx.Response(200, headers={"content-type": content_type, **(headers or {})}, stream=stream)

        with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter, _client(handler) as client:
            request = client.build_request("POST", url, json=body or _body(stream=True))
            response = client.send(request, stream=True)
            if read:
                response.read()
            response.close()
        return meter

    def test_anthropic_sse_merges_start_and_delta(self, ledger, tmp_path):
        payload = self._sse(
            {"type": "message_start", "message": {"model": "claude-haiku-4-5-20251001",
                                                  "usage": {"input_tokens": 50, "output_tokens": 1}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": _SENTINEL}},
            {"type": "message_delta", "usage": {"output_tokens": 9}},
        )
        self._send_stream(ledger, payload, "text/event-stream")
        (call,) = _calls(tmp_path)
        assert (call["status"], call["tokens"]["input"], call["tokens"]["output"]) == ("usage", 50, 9)
        assert call["model_on_wire"] == "claude-haiku-4-5-20251001"
        _assert_no_text(tmp_path)

    def test_openai_responses_completed_event(self, ledger, tmp_path):
        payload = self._sse(
            {"type": "response.output_text.delta", "delta": _SENTINEL},
            {"type": "response.completed", "response": {"model": "gpt-5.4-mini",
                                                       "usage": {"input_tokens": 30, "output_tokens": 4}}},
        )
        self._send_stream(ledger, payload, "text/event-stream", url="https://api.openai.com/v1/responses",
                          body={"model": "gpt-5.4-mini", "input": _SENTINEL, "stream": True})
        (call,) = _calls(tmp_path)
        assert (call["tokens"]["input"], call["tokens"]["output"]) == (30, 4)

    def test_gemini_stream_takes_the_last_usage(self, ledger, tmp_path):
        payload = self._sse(
            {"candidates": [{"content": {"parts": [{"text": _SENTINEL}]}}],
             "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 1}},
            {"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 6}},
        )
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:streamGenerateContent"
        self._send_stream(ledger, payload, "text/event-stream", url=url, body={"contents": []})
        (call,) = _calls(tmp_path)
        assert (call["tokens"]["input"], call["tokens"]["output"]) == (20, 6)

    def test_compressed_json_read_through_the_stream(self, ledger, tmp_path):
        payload = gzip.compress(json.dumps(_anthropic_reply(input_tokens=77)).encode())
        self._send_stream(ledger, payload, "application/json", body=_body(), headers={"content-encoding": "gzip"})
        (call,) = _calls(tmp_path)
        assert (call["status"], call["tokens"]["input"]) == ("usage", 77)

    def test_closed_before_reading_is_charged_at_the_estimate(self, ledger, tmp_path):
        payload = json.dumps(_anthropic_reply()).encode()
        meter = self._send_stream(ledger, payload, "application/json", body=_body(), read=False)
        (call,) = _calls(tmp_path)
        assert call["status"] == "stream_unfinished"
        assert call["usd"] > 0 and call["usd"] == pytest.approx(ledger.total_usd)
        assert meter.stats.uncounted == 1


async def test_async_client_is_counted(ledger, tmp_path):
    async def handler(request):
        return httpx.Response(200, json=_anthropic_reply(input_tokens=5, output_tokens=1))

    with spend.UsageMeter(ledger, "unit", max_usd=1.0) as meter:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await client.post(_ANTHROPIC, json=_body())
    assert meter.stats.counted == 1
    assert _calls(tmp_path)[0]["tokens"]["input"] == 5


# ─── 支出の記録 ──────────────────────────────────────────────────


class TestLedger:
    def test_lock_refuses_a_second_paid_run(self, tmp_path):
        with spend.UsageMeter(spend.SpendLedger(tmp_path), "one", max_usd=1.0):
            assert (tmp_path / "spent.lock").exists()
            with pytest.raises(spend.LedgerBusy):
                spend.SpendLedger(tmp_path).acquire()
        assert not (tmp_path / "spent.lock").exists()

    def test_nested_meters_are_refused(self, tmp_path):
        with spend.UsageMeter(spend.SpendLedger(tmp_path / "a"), "one", max_usd=1.0):
            with pytest.raises(RuntimeError):
                with spend.UsageMeter(spend.SpendLedger(tmp_path / "b"), "two", max_usd=1.0):
                    pass

    def test_written_atomically_and_read_back(self, ledger, tmp_path):
        ledger.add(run="r", model="claude-haiku-4-5", usd=0.5, usage=spend.Usage(input=3, output=1), status="usage")
        ledger.add(run="r", model="claude-haiku-4-5", usd=0.25, usage=None, status="no_usage")
        assert not list(tmp_path.glob("*.tmp"))
        reopened = spend.SpendLedger(tmp_path)
        assert reopened.total_usd == pytest.approx(0.75)
        assert reopened.data["by_run"]["r"]["calls"] == 2
        assert reopened.data["by_run"]["r"]["uncounted"] == 1
        assert reopened.data["by_model"]["claude-haiku-4-5"]["input"] == 3

    @pytest.mark.parametrize("run", ["", "a b", "x" * 65, _SENTINEL + "/.."])
    def test_run_names_are_short_safe_tokens(self, ledger, run):
        with pytest.raises(ValueError):
            spend.UsageMeter(ledger, run, max_usd=1.0)

    def test_unknown_schema_is_refused(self, tmp_path):
        (tmp_path / "spent.json").write_text('{"schema": 2}', encoding="utf-8")
        with pytest.raises(ValueError):
            spend.SpendLedger(tmp_path)


# ─── 本文を出さない後始末 ────────────────────────────────────────


def _run_python(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=tmp_path, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=120, check=False)


class TestQuietFailures:
    def test_describe_failure_has_no_message(self):
        try:
            raise ValueError(_SENTINEL)
        except ValueError as exc:
            text = spend.describe_failure(exc)
        assert text.startswith("ValueError")
        assert _SENTINEL not in text

    @pytest.mark.parametrize(("body", "code"), [
        ("raise ValueError(SENTINEL)", 1),
        ("raise spend.BudgetExceeded('上限で止めました')", 4),
        ("raise SystemExit(64)", 64),
        ("return 0", 0),
    ])
    def test_run_tool_exit_codes_and_no_text(self, tmp_path, body, code):
        done = _run_python(f"""
            from lab_lounge import spend
            SENTINEL = {_SENTINEL!r}
            def main():
                {body}
            spend.run_tool(main, name="t")
        """, tmp_path)
        assert done.returncode == code, done.stdout + done.stderr
        assert _SENTINEL not in done.stdout + done.stderr

    def test_thread_failures_show_only_the_type(self, tmp_path):
        done = _run_python(f"""
            import threading
            from lab_lounge import spend
            def main():
                worker = threading.Thread(target=lambda: (_ for _ in ()).throw(KeyError({_SENTINEL!r})))
                worker.start(); worker.join()
                return 0
            spend.run_tool(main, name="t")
        """, tmp_path)
        assert done.returncode == 0
        assert "KeyError" in done.stdout
        assert _SENTINEL not in done.stdout + done.stderr

    def test_hard_exit_releases_the_lock(self, tmp_path):
        done = _run_python(f"""
            from pathlib import Path
            from lab_lounge import spend
            meter = spend.UsageMeter(spend.SpendLedger(Path({str(tmp_path)!r})), "t", max_usd=1.0)
            meter.__enter__()
            spend.hard_exit(3)
        """, tmp_path)
        assert done.returncode == 3, done.stdout + done.stderr
        assert not (tmp_path / "spent.lock").exists()


class TestCommandLine:
    def test_show_prints_numbers_only(self, ledger, tmp_path, capsys):
        ledger.add(run="r1", model="claude-haiku-4-5", usd=0.01, usage=spend.Usage(input=3), status="usage")
        assert spend.main(["show", "--dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "$0.0100" in out and "r1" in out

    def test_prices_lists_every_model_with_its_source(self, capsys):
        assert spend.main(["prices"]) == 0
        out = capsys.readouterr().out
        for model, price in spend.PRICES.items():
            assert model in out and price.source_url in out

    def test_bad_arguments_are_usage_errors(self, capsys):
        assert spend.main(["nope"]) == spend.EXIT_USAGE


# ─── 本番のモジュールは計測の道具を import しない ────────────────


_TOOL_MODULES = {"spend", "replay_set", "replay", "probe", "c2_probe"}
# 道具どうし（と点検）は互いに import してよい。
_TOOL_FILES = _TOOL_MODULES | {"preflight"}


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return names


def test_production_modules_do_not_import_the_measuring_tools():
    """計測の道具は本番の経路の外に置く。本番のモジュールが import すると、包みや差し替えが配信に入りうる。"""
    package = _REPO_ROOT / "src" / "lab_lounge"
    offenders = []
    for path in package.rglob("*.py"):
        if path.parent == package and path.stem in _TOOL_FILES:
            continue
        for name in _imported_names(ast.parse(path.read_text(encoding="utf-8"))):
            last = name.rsplit(".", 1)[-1]
            if last in _TOOL_MODULES and (name.startswith(("lab_lounge", ".")) or name == last):
                offenders.append(f"{path.relative_to(package).as_posix()}: {name}")
    assert offenders == []
