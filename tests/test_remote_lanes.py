import asyncio
import json

import httpx
import pytest

from thought_leak_range.cli import build_parser
from thought_leak_range.remote_lanes import (
    RemoteLaneConfig,
    RemoteLanePoolClient,
    load_remote_lane_configs,
)
from thought_leak_range.runner import (
    MockReasoningPilot,
    RunArtifacts,
    probe_raw_reasoning,
)


def test_load_remote_lanes_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "TEST_REMOTE_LANES",
        json.dumps(
            {
                "lanes": [
                    {
                        "name": "t4-a",
                        "endpoint": "https://lane-a.example/",
                        "token": "a" * 32,
                    },
                    {
                        "name": "t4-b",
                        "endpoint": "https://lane-b.example",
                        "token_env": "TEST_LANE_B_TOKEN",
                    },
                ]
            }
        ),
    )
    monkeypatch.setenv("TEST_LANE_B_TOKEN", "b" * 32)

    configs = load_remote_lane_configs(env_name="TEST_REMOTE_LANES")

    assert [config.name for config in configs] == ["t4-a", "t4-b"]
    assert configs[0].endpoint == "https://lane-a.example"
    assert configs[1].bearer_token == "b" * 32


@pytest.mark.parametrize(
    "row, message",
    [
        (
            {
                "name": "bad-http",
                "endpoint": "http://lane.example",
                "token": "x" * 32,
            },
            "https URL",
        ),
        (
            {
                "name": "secret-in-url",
                "endpoint": "https://token@lane.example",
                "token": "x" * 32,
            },
            "credentials",
        ),
        (
            {
                "name": "short-secret",
                "endpoint": "https://lane.example",
                "token": "short",
            },
            "unexpectedly short",
        ),
    ],
)
def test_load_remote_lanes_rejects_unsafe_config(monkeypatch, row, message) -> None:
    monkeypatch.setenv("TEST_REMOTE_LANES", json.dumps([row]))
    with pytest.raises(ValueError, match=message):
        load_remote_lane_configs(env_name="TEST_REMOTE_LANES")


def test_remote_lane_pool_uses_each_physical_endpoint() -> None:
    async def exercise() -> None:
        seen_hosts: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["authorization"].startswith("Bearer ")
            host = request.url.host
            if request.url.path == "/health":
                return httpx.Response(
                    200,
                    json={
                        "ready": True,
                        "model": "meta-llama/Llama-3.1-8B-Instruct",
                        "gpu": "Tesla T4",
                        "quantization": "NF4",
                        "constrained_digits": False,
                        "prefix_tokens": 123,
                    },
                )
            seen_hosts.append(host)
            await asyncio.sleep(0.01)
            body = json.loads(request.content)
            token = "5" if "x=0" in body["observation"] else "4"
            return httpx.Response(
                200,
                json={
                    "request_id": body["request_id"],
                    "token": token,
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "compute_ms": 107.0,
                    "queue_ms": 0.1,
                    "suffix_tokens": 18,
                    "constrained_digits": False,
                },
            )

        configs = (
            RemoteLaneConfig("t4-a", "https://lane-a.example", "a" * 32),
            RemoteLaneConfig("t4-b", "https://lane-b.example", "b" * 32),
        )
        client = RemoteLanePoolClient(
            configs,
            transport=httpx.MockTransport(handler),
        )
        assert client.model == "remote-lane-pool"
        visible: list[str] = []
        try:
            health = await client.warmup()
            assert len(health) == 2
            results = await asyncio.gather(
                client.stream_motor(
                    observation_text="v=1 x=0 a=10",
                    run_id="run",
                    observation_seq=1,
                    on_visible=lambda token, _arrived: visible.append(token),
                ),
                client.stream_motor(
                    observation_text="v=0 x=9999 a=10",
                    run_id="run",
                    observation_seq=2,
                    on_visible=lambda token, _arrived: visible.append(token),
                ),
            )
        finally:
            await client.aclose()

        assert set(seen_hosts) == {"lane-a.example", "lane-b.example"}
        assert set(visible) == {"4", "5"}
        assert {result.provider for result in results} == {
            "remote-colab/t4-a",
            "remote-colab/t4-b",
        }
        snapshot = client.snapshot()
        assert [lane["requests"] for lane in snapshot["lanes"]] == [1, 1]
        assert all(
            lane["server_compute_ms"]["mean"] == 107.0
            for lane in snapshot["lanes"]
        )

    asyncio.run(exercise())


def test_remote_lane_pool_runs_vago_text_contract() -> None:
    async def exercise() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert request.url.path == "/vago-text"
            assert body["system_prompt"] == "s" * 20
            assert body["user_content"] == "u" * 100
            return httpx.Response(
                200,
                json={
                    "request_id": body["request_id"],
                    "action": "shoot+turn_left",
                    "buttons": [1, 0, 1, 0],
                    "completion": "shoot+turn_left",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "compute_ms": 401.0,
                    "queue_ms": 0.0,
                    "prompt_tokens": 2240,
                    "completion_tokens": 4,
                },
            )

        client = RemoteLanePoolClient(
            (RemoteLaneConfig("t4-a", "https://lane-a.example", "a" * 32),),
            transport=httpx.MockTransport(handler),
        )
        try:
            decision = await client.decide_vago_text(
                system_prompt="s" * 20,
                user_content="u" * 100,
                run_id="run",
                observation_seq=7,
            )
        finally:
            await client.aclose()

        assert decision.action == "shoot+turn_left"
        assert decision.buttons == (1, 0, 1, 0)
        assert decision.prompt_tokens == 2240
        assert decision.server_compute_ms == 401.0

    asyncio.run(exercise())


def test_remote_live_parser_defaults_to_three_direct_motor_lanes() -> None:
    args = build_parser().parse_args(["remote-live"])
    assert args.tap_mode == "direct-motor"
    assert args.lanes == 3
    assert args.lane_env == "LATENCY_KILLS_REMOTE_LANES"
    assert args.probe_profile == "holdout"


def test_holdout_motor_probe_uses_unseen_coordinates(tmp_path) -> None:
    async def exercise():
        artifacts = RunArtifacts(
            base_dir=tmp_path,
            run_id="holdoutprobe",
            save_thoughts=False,
        )
        try:
            return await probe_raw_reasoning(
                pilot=MockReasoningPilot(tap_mode="direct-motor"),
                run_id="holdoutprobe",
                artifacts=artifacts,
                show_thoughts=False,
                tap_mode="direct-motor",
                motor_probe_profile="holdout",
            )
        finally:
            artifacts.close()

    probe = asyncio.run(exercise())

    assert probe.passed
    assert probe.marker_action == "ALL_6"
    assert set(probe.specialist_results) == {
        "WAIT",
        "LEFT_SHORT",
        "LEFT_LONG",
        "RIGHT_SHORT",
        "RIGHT_LONG",
        "FIRE",
    }
