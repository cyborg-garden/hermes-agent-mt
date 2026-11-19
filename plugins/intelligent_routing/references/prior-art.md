# Prior art: model-task-router (PR #43534)

This plugin adapts earlier community work.
What is new here is the plugin form and the interface change. The classifier is not.
This page is a short citation. The original data stays in the original PR.

## Source

- **PR #43534**, *"feat(skills): add model-task-router — automatic task-to-model routing backed by DeepSWE data"*, on `NousResearch/hermes-agent`. Closed. By [Sugumaran Balasubramaniyan](https://github.com/Sugumaran-Balasubramaniyan), MIT-licensed.
- It shipped as a skill (two markdown files) and routed by task type: code generation, hard architecture, orchestration, research, mechanical.
- The maintainer, teknium1, closed it with this note: *"If you do want it, please create a plugin. Happy to support adding to the plugin interface to make this work as one."* This plugin and `agent/routing_override.py` follow that suggestion.

## What we kept and changed

**Kept:** the five task categories, and the principle of routing by task type with benchmarks treated as rough guides.

**Changed:** the tiers. Premium is Claude, cheap is OpenRouter `deepseek/deepseek-v3.2`, and the local floor is ollama. The PR's defaults were GPT-5.4 and GPT-5.5.

## The data behind the cheap-tier choice

From the DeepSWE reference in #43534, and the correction thread in `datacurve-ai/deep-swe#21`. The original author withdrew the solve-rate analysis there; only the cost correction still stands.

- DeepSeek V4-Pro scored low on DeepSWE **coding**, about 8%. That number comes with caveats: no effort tuning, OpenRouter guardrail 404s, and a community range of about 5–8%. So hard code generation never goes to DeepSeek.
- DeepSeek V4-Pro is **competitive at CLI and tool orchestration**: 67.9% on Terminal-Bench at $0.87 per million output tokens. So DeepSeek gets only the turns where it is both competitive and cheap: mechanical and orchestration work.

That gap is why the cheap tier is `deepseek-v3.2` and code generation stays on Claude.
Treat the numbers as rough. Recalibrate from your own measurements before relying on them.
