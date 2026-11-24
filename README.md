# q8-opus-round003

A specialized inference engine for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) at revision `b968826d9c46dd6066d109eabc6255188de91218`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.5 LTS
- Reference: nxdi
- Judged under evaluation version 2
- Decode: not recorded (this run predates the one-forward-per-token rule)
- Source and data: https://github.com/xuanfeiren/enginesmith-q8-opus-round003
- Search: 12 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $472.66

## Notes from the author

- This run predates the held-out final evaluation and the one-forward-per-token rule (evaluation version 2), so the numbers below come from the search evaluation.
- Round 003 is the best round of this run that generates one token per forward pass. From round 004 on, the agent turned to speculative decoding, whose speed on the evaluation's random-token prompts overstates what it gives on real text, so those rounds (and their scores in the table below) are not what is shared here.
- A clean rebuild of round 003, checked later with the current procedure on fresh held-out prompts (5 rounds, the engine given only the reference's tensor-parallel degree in cores): passed, 1 of 2080 judged steps mismatch; score 1.56x NxD Inference (prefill 1.72x, decode 1.41x). Decode probe 1.01.

## Speed against nxdi (search evaluation)

Score 1.58x; prefill 1.73x, decode 1.44x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 2.06x | 1.39x | not computed | not computed |
| 1,024 | 2.11x | 1.49x | not computed | not computed |
| 4,096 | 1.19x | 1.43x | not computed | not computed |

## Speed in absolute terms

Medians over the paired rounds of the search evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 32.6 (nxdi 67.1) | 7,854.8 (3,817.2) | 43.3 (31.1) |
| 1,024 | 115.3 (nxdi 243.2) | 8,884.1 (4,209.9) | 42.5 (28.5) |
| 4,096 | 511.5 (nxdi 607.0) | 8,008.6 (6,748.2) | 41.3 (29.0) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.01x as long as one request for the same tokens at 256 prompt tokens (1,477.3 ms against 1,465.8 ms; each request 22.94 to 26.60 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 12 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 2918 min.
- First correct engine: round 000, 28 min after the start.
- First engine as fast as nxdi (score 1.0): round 000, 28 min after the start.
- Agent usage: 3,642 input and 1,811k output tokens, 413,778k cache reads, $472.66 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 000 | seed | accepted (first_passing_candidate) | 27.8 | 1.04x | 0.99x | 1.08x | - / - | - |
| 001 | optimize | accepted (score_improved) | 424.1 | 1.47x | 1.59x | 1.36x | 526 / 364k | 60,384 |
| 002 | optimize | accepted (score_improved) | 573.1 | 1.50x | 1.65x | 1.37x | 252 / 163k | 60,662 |
| 003 | optimize | accepted (score_improved) | 1,029.1 | 1.58x | 1.73x | 1.44x | 492 / 350k | 61,576 |
| 004 | optimize | accepted (score_improved) | 1,248.8 | 1.63x | 1.72x | 1.55x | 282 / 115k | 62,359 |
| 005 | optimize | accepted (score_improved) | 1,359.1 | 1.96x | 1.72x | 2.24x | 220 / 63k | 62,482 |
| 006 | optimize | accepted (score_improved) | 1,490.8 | 2.29x | 1.73x | 3.04x | 178 / 52k | 62,623 |
| 007 | optimize | rejected (insufficient_improvement) | 1,699.3 | 2.13x | 1.72x | 2.64x | 248 / 105k | 63,029 |
| 008 | optimize | accepted (score_improved) | 2,048.7 | 2.71x | 1.72x | 4.25x | 428 / 200k | 63,277 |
| 009 | optimize | accepted (case_improved) | 2,397.6 | 2.68x | 1.72x | 4.18x | 440 / 196k | 66,109 |
| 010 | optimize | accepted (score_improved) | 2,581.3 | 2.90x | 1.72x | 4.88x | 264 / 129k | 66,328 |
| 011 | optimize | rejected (insufficient_improvement) | 2,918.2 | 2.22x | 1.73x | 2.85x | 312 / 74k | 66,462 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-q8-opus-round003 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --local-dir models/Qwen3-8B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-q8-opus-round003 --model models/Qwen3-8B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
