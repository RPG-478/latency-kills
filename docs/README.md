# Documentation map

Latency Kills grew by experiment, including the useful failures. This page is the shortest route
through the resulting paper trail.

## Start here

- [V4 direct motor](v4-direct-motor.md) — the one-character Cloud LLM motor protocol
- [Stopped V4, unpaused V4, unpaused VAGO 1.3M](experiment-v4-vago-three-way.md) — paired three-way result: 26.3 / 4.0 / 17.7 kills
- [Slow the 1.3M policy to 200 ms](experiment-vago-1.3m-200ms-latency.md) — latency alone drops it from 17.7 to 4.2 kills
- [Paper plan: action staleness](paper-plan-action-staleness.md) — hypothesis, contribution boundary, and the experiments still required
- [Physical Colab T4 lanes](experiment-three-t4-distributed-lanes.md) — the account allowed two, not three; throughput doubled while score moved only 3.9→4.4
- [V4 probe language failure](v4-probe-language-failure.md) — the old 6/6 smoke test fell to 29/53 on unseen coordinates; both T4s failed identically
- [Byte-aligned VAGO Cloud text input](experiment-vago-cloud-text-alignment.md) — OpenRouter and two T4s, including a system-prefix cache that cut compute 25% without changing kills
- [Prior art and fairness boundary](prior-art.md) — what this rebuts and what it does not
- [V4 Async × flat-4](experiment-v4-async-flat-4.md) — current unpaused result: 4.0 kill average
- [V4 Async versus VAGO MultiVec 1.3M](experiment-v4-async-vs-vago-1.3m.md) — older stopped-world CPU reproduction: 15.6 versus 4.0 kills
- [V4-S × VAGO flat-4](experiment-v4-s-vago-flat-4.md) — stopped-world diagnostic: 26.3 kill average
- [Replay archive](replays/README.md) — GIFs grouped by experiment and validity
- [Public release record](public-release-2026-08-21.md) — what was opened, what stayed private, and the final checks
- [Rename to Latency Kills](rename-latency-kills-2026-08-21.md) — why the V0 name became a compatibility alias

## Evolution

- [Original fixed-target experiment](experiment.md)
- [V3 four-agent blackboard](v3-four-agent-blackboard.md)
- [V3 blackboard interference failure](v3-blackboard-interference.md)
- [V2 versus V3, ten paired seeds](experiment-v2-v3-10x.md)
- [V4 ten-run comparison](experiment-v4-10x.md)
- [V4 probe-language failure](v4-probe-language-failure.md)
- [V4 multi-target thrash](v4-multi-target-thrash.md)

## Timing and control ablations

- [V4 stopped-world clock experiment](experiment-v4-vago-sync.md)
- [Repaired asynchronous clock, ten runs](experiment-v4-clock-async-10x.md)
- [V4 side lease](experiment-v4-side-lease.md)
- [VAGO frame-skip 4](experiment-v4-s-vago-frame-skip-4.md)
- [VAGO timing probe](vago-sync-probe.md)

## Reading the numbers

`vago-sync` stops simulation while the Cloud model answers. `clock-thread` keeps ViZDoom advancing at
35 Hz. Their kill totals are deliberately shown together to expose the latency cost, but they are not
interchangeable benchmark conditions. GIF capture can slow the native clock, so visual-only replay runs
are labeled and excluded from formal performance averages.
