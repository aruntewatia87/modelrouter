# modelrouter

> Cost-aware, health-aware LLM routing with multi-armed bandits — price, latency, quality and reliability balanced continuously, with no LLM in the decision loop.

[![CI](https://github.com/aruntewatia87/modelrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/modelrouter/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE

A static routing table — "always route to the cheapest model that supports
tools" — is the default in most AI gateways, and it breaks the moment that
model degrades, starts rate-limiting, or quietly returns worse answers. By the
time your error rate is up, the routing table is already wrong, and nothing in
the stack knows it. `modelrouter` replaces the table with a policy: it runs
**multi-armed bandit** algorithms (UCB1, Thompson sampling, epsilon-greedy,
softmax — all implemented from scratch) over a set of *model arms*, and scores
each arm on a single shaped reward that folds together **quality, price,
latency and health**. A capability filter, a per-request budget ceiling, a
health circuit breaker and a kill switch sit in front of the bandit, so the
router can never send a tool-call request to a model without tool support and
can never exceed the spend ceiling you set.

The point is not to add another routing layer. It is to make the trade-off
**explicit and measurable**: one knob (`lambda_cost`) decides whether a dollar
saved is worth a point of answer quality, and the router's behaviour changes
accordingly, live, from observed outcomes.

---

## 🔍 Why this exists

This is not a theoretical gap. The three problems below are **open issues on
LiteLLM** (61k+ stars, the de-facto standard LLM gateway) and are the reason a
dedicated router exists as a separate concern:

| Issue | Status | What it says |
|---|---|---|
| [BerriAI/litellm#35272](https://github.com/BerriAI/litellm/issues/35272) | open | *"Feature request: health-aware multi-backend routing for Anthropic models"* — the only **open feature request** for health-aware routing in the project. Routing there is health-blind: when a backend degrades or rate-limits, requests must be re-routed by hand. |
| [BerriAI/litellm#38064](https://github.com/BerriAI/litellm/issues/38064) | open | *"cost-based-routing ignores cache_read price; on an exact input+output tie the cheaper provider is picked only if it's listed first (10x)"* — the cost model has a hole, and it breaks routing correctness on ties. |
| [BerriAI/litellm#46469](https://github.com/BerriAI/litellm/issues/46469) | open | *"complexity_router LLM classifier cost is not attributed to spend or budgets"* — the classifier that picks the model costs money that nobody tracks (~1.5% of provider-billed spend in the reporter's deployment). |

And the gap they leave behind: a GitHub search for
`"llm quality slo monitoring production"` returns **0 repositories**. Quality
is not an SLO anyone is monitoring, so no router can balance against it.

`modelrouter` is the missing piece:

- **Health-aware** (#35272) — a sliding-window breaker that pulls a degraded
  model out of rotation and puts it back when it recovers.
- **Cost-model-correct** (#38064) — input, output and cache-read prices are
  three separate numbers, and the cache-read regression is a pinned test.
- **Classifier-free** (#46469) — request complexity is classified with a
  deterministic, free heuristic, so there is no untracked classifier spend to
  attribute in the first place.
- **Quality-aware** — quality is a first-class term in the reward, so it can
  finally be traded against dollars with an explicit, measurable price.

---

## ✨ Highlights
- **Multi-armed bandits from scratch** — UCB1, Thompson sampling (Beta-Bernoulli),
  epsilon-greedy with decaying epsilon, and a temperature-controlled softmax
  policy. No bandit library, no wrapper.
- **Reward shaping is the heart of it** — one equation turns
  `(success, quality, cost, latency)` into the scalar the bandit optimises, and
  `lambda_cost` re-orders your arms live.
- **Capability hard filter** — a request needing tools, JSON mode, vision or a
  long context can only reach arms that support *all* of them. Never a
  capability mismatch, never a 400 from the provider.
- **Health circuit breaker** — sliding-window success rate, p95 latency
  (via `numpy.percentile`) and per-error-type rates. Unhealthy arms are
  excluded from routing and re-admitted automatically once they recover.
- **Kill switch with recovery probes** — an arm whose rolling success rate
  collapses is pulled from rotation and re-probed after a cool-down.
- **Correct cost model** — input, output **and cache-read** prices are tracked
  separately, so a cache-warm provider is correctly identified as the cheaper
  one instead of losing a price tie.
- **Zero LLM calls in the control path** — request complexity is classified
  with a deterministic, free heuristic. No classifier spend, no classifier
  latency, no untracked budget.
- **Spend projection** — project a policy's cost over N requests and compare
  it against "always cheapest" and "always best" baselines.
- **Deterministic and offline** — numpy is the only runtime dependency, every
  policy is seeded, and nothing touches the network.

---

## 🚀 Quickstart
```bash
uv sync --all-extras --dev
uv run pytest -q
```

Run the end-to-end demo (200 simulated requests, ~1s):
```bash
uv run python examples/quickstart.py
```

---

## 🔧 API Tour

### Register your model arms

```python
from modelrouter import ArmRegistry, ModelArm

registry = ArmRegistry(
    [
        ModelArm(
            name="cheap-mini",
            provider="openai",
            input_cost_per_1k=0.00015,
            output_cost_per_1k=0.00060,
            cache_read_cost_per_1k=0.000015,  # 10x cheaper than input
            capabilities=frozenset({"chat", "json_mode"}),
            max_context_tokens=128_000,
            avg_latency_ms=420.0,
            quality_prior=0.62,
            reliability=0.97,
        ),
        ModelArm(
            name="flagship",
            provider="anthropic",
            input_cost_per_1k=0.003,
            output_cost_per_1k=0.015,
            cache_read_cost_per_1k=0.0003,
            capabilities=frozenset({"chat", "tools", "json_mode", "vision", "long_context"}),
            max_context_tokens=200_000,
            avg_latency_ms=1450.0,
            quality_prior=0.93,
            reliability=0.995,
        ),
    ]
)
```

### Build a router

```python
from modelrouter import HealthTracker, RewardShaper, Router, UCB1

router = Router(
    registry,
    UCB1(c=1.4, seed=2026),
    shaper=RewardShaper(
        lambda_cost=0.35,  # how much a dollar is worth
        lambda_latency=0.1,  # how much a millisecond is worth
        cost_reference=0.0081,  # price of the most expensive arm, per request
        latency_reference=1450.0,
    ),
    health=HealthTracker(window_size=30, min_success_rate=0.7, max_latency_multiplier=3.0),
    budget_per_request_usd=0.0081,  # hard per-request ceiling
    min_pulls=15,  # pulls before the kill switch can trip
    kill_threshold=0.5,  # rolling success rate below which it trips
)
```

### Route, then feed the outcome back

```python
from modelrouter import Outcome, RequestContext

request = RequestContext.from_prompt(
    "req-0001", "Summarise this incident report", needs_tools=False
)
decision = router.route(request)
print(decision.arm)  # -> 'cheap-mini'

router.observe(
    "req-0001",
    Outcome(
        success=True,
        quality=0.82,
        latency_ms=640.0,
        cost_usd=0.00036,
    ),
)
```

`router.observe()` is the only feedback path: it updates the per-arm
statistics, the health tracker, and the bandit (via the reward shaper). The
router learns purely from what you report back — no instrumentation hooks
required.

### Inspect what the router learned

```python
stats = router.stats()
stats.total_spend_usd  # total observed spend
stats.per_arm["cheap-mini"]["success_rate"]
stats.excluded_arms  # arms currently pulled from rotation
router.health.health_score("cheap-mini")
router.traffic_mix()  # {'cheap-mini': 0.61, 'flagship': 0.18, ...}
```

### The capability filter is a hard filter

```python
from modelrouter import RequestContext

tool_request = RequestContext(request_id="req-0002", needs_tools=True)
[arm.name for arm in router.candidate_arms(tool_request)]
# -> ['flagship']    (the cheap-mini arm does not support tools)
```

### Costing, including the cache-read price

```python
from modelrouter import estimate_cost

arm = registry.get("flagship")
cold = estimate_cost(arm, prompt_tokens=2000, completion_tokens=500)
warm = estimate_cost(arm, prompt_tokens=2000, completion_tokens=500, cached_input_tokens=1800)
# warm < cold -- and the saving is the 10x cache-read discount, applied
# correctly. Ignoring cache reads is what breaks price ties.
```

### Projecting spend against the baselines

```python
from modelrouter import project_spend

project_spend("always-cheapest", registry.arms, 10_000, mix={"cheap-mini": 1.0})
project_spend("always-best", registry.arms, 10_000, mix={"flagship": 1.0})
project_spend("router", registry.arms, 10_000, mix=router.traffic_mix())
```

### Other policies

```python
from modelrouter import EpsilonGreedy, SoftmaxRouter, ThompsonSampling

EpsilonGreedy(epsilon_start=0.2, epsilon_end=0.01, decay_steps=2000, seed=7)
ThompsonSampling(alpha=1.0, beta=1.0, seed=7)
SoftmaxRouter(temperature=0.1, seed=7)
```

All four satisfy the `Bandit` protocol (`select(arms, context) -> str`,
`update(arm, reward, context)`), so they are interchangeable.

---

## 🧠 How it works

### The routing pipeline

Every request goes through five stages, in order:

1. **Capability hard filter** — `ArmRegistry.eligible_for(capabilities)`
   returns only the arms that support *every* required capability. A tool
   request physically cannot reach an arm without tool support, regardless of
   what the bandit thinks.
2. **Budget filter** — arms whose estimated cost for the request exceeds
   `budget_per_request_usd` are dropped.
3. **Kill switch** — an arm whose *rolling* success rate falls below
   `kill_threshold` (after at least `min_pulls`) is pulled from rotation. It
   is re-probed after a cool-down so it can recover.
4. **Health breaker** — `HealthTracker` keeps a sliding window of signals per
   arm. If the window success rate drops below `min_success_rate`, or the
   window p95 latency exceeds `max_latency_multiplier` × the arm's baseline,
   the arm is excluded. Because the window slides, a recovered arm is
   automatically re-admitted.
5. **Bandit selection** — the policy picks among the survivors.

### UCB1

Upper Confidence Bound 1 scores each arm after `t` total pulls:

```
score_i = mean_reward_i + c * sqrt( ln(t) / n_i )
```

- `mean_reward_i` is the average shaped reward arm `i` has earned.
- `n_i` is the number of times arm `i` has been pulled.
- `c` is the exploration constant — how much the policy is willing to pay for
  information.

An arm with `n_i == 0` has an **infinite** score, so untried arms are always
explored first. The exploration bonus decays as `1/sqrt(n_i)`, so an arm that
has been tried a lot has to be *clearly* better to be re-picked. Ties break
deterministically on the arm name, so a given seed always produces the same
routing decisions.

### Reward shaping

The bandit needs a single scalar. `RewardShaper` produces it:

```
reward = quality * success
         - lambda_cost * normalized_cost
         - lambda_latency * normalized_latency
```

where `normalized_cost` is the observed cost divided by `cost_reference` and
`normalized_latency` is the observed latency divided by
`latency_reference`. A failed call zeroes the quality term, so failures are
always worth less than mediocre successes.

This is the lever that makes the router economically meaningful. Consider two
arms: a cheap one with `quality = 0.62` and an expensive one with
`quality = 0.93`.

- With `lambda_cost = 0.0` the expensive arm wins — you are buying quality.
- With `lambda_cost = 0.35` the cheap arm wins — a quarter of a percent of
  quality is worth more than the price difference.
- With `lambda_cost = 5.0` the cheap arm wins by a landslide — every dollar
  counts.

The ordering **flips** as `lambda_cost` moves. The demo below prints exactly
that table.

### The cost model

```
cost = (uncached_input / 1000) * input_cost_per_1k
     + (cached_input  / 1000) * cache_read_cost_per_1k
     + (completion     / 1000) * output_cost_per_1k
```

Cache reads are priced on their own sheet, typically 10× cheaper than input
tokens. Collapsing them into the input price is not a rounding error: it
inflates the cost of a cache-warm provider and can invert a price tie, so
"always cheapest" ends up choosing the *more* expensive provider. The
`estimate_cost` regression test pins this down.

### Complexity classification, for free

`RequestContext.is_complex` is a deterministic heuristic: prompt length above
2000 estimated tokens, a fenced code block, more than one question mark, or a
multi-step keyword ("then", "analyze", "compare", "first"). No LLM call, no
latency, no spend — and nothing for a budget to leak into.

### Health scoring

```
health_score = window_success_rate * 1 / (1 + max(0, p95_latency / baseline_latency - 1))
```

The score is in `[0, 1]`. A fast, reliable arm scores 1.0. A failing arm
scores near 0. The health score is what the breaker reads, and it is the
number worth graphing in your dashboard next to spend.

---

## 📊 Example output

Verbatim output of `uv run python examples/quickstart.py`:

```text
Simulating 200 requests across 4 model arms
  cost reference   : $0.008100 per request
  latency reference: 1450 ms
  policy           : UCB1(c=1.4) + reward shaping (lambda_cost=0.35, lambda_latency=0.1)
  'legacy-mini' (the cheapest arm) starts failing at request 90

Per-arm results
  arm                 pulls   share     ok%  quality   avg $/req   p95 ms  health
  ------------------ ------ ------- ------- -------- ----------- -------- -------
  legacy-mini            22  11.0%  90.9%    0.279    0.000169     2879   0.120
  cheap-mini             61  30.5% 100.0%    0.579    0.000360      461   0.911
  mid-haiku              80  40.0% 100.0%    0.824    0.002700      902   0.865
  flagship-sonnet        37  18.5% 100.0%    0.903    0.008100     1569   0.924

Total spend      : $0.541380 over 200 requests
Overall success  : 99.0%
Excluded arms    : legacy-mini

Cost projection (200 requests, same token volume, list price)
  policy                      total $    avg $/req
  ---------------------- ------------ ------------
  always-cheapest            0.037200     0.000186
  router (learned mix)       0.541752     0.002709
  always-best                1.620000     0.008100

  router saves 66.6% vs always-best on list price ($1.078248)

Realistic comparison (same 200 requests, 1 fallback retry on failure)
  policy               total $   success  avg quality
  ---------------- ----------- --------- ------------
  always-cheapest     0.063120   100.0%        0.583
  router              0.541380    99.0%        0.704
  always-best         1.620000   100.0%        0.928

  always-cheapest is the cheapest on paper but collapses when the cheap arm degrades;
  always-best is the most reliable and the most expensive; the router sits between them.

Reward-shaping knob (same outcome, different lambda_cost)
  lambda_cost    cheap-mini     flagship  preferred
  0.0                0.6200       0.9300  flagship-sonnet
  0.35               0.6044       0.5800  cheap-mini
  5.0                0.3978      -4.0700  cheap-mini
```

Read the last table: the *same two arms*, the *same outcomes*, and the
preferred arm flips as `lambda_cost` moves. That is the whole product in
three rows.

---

## 🧪 Testing
```bash
uv sync --all-extras --dev
uv run pytest -q --cov=modelrouter --cov-report=term-missing
uv run ruff check .
```

126 tests covering the capability filter, the budget filter, the kill switch
and its recovery probes, the health breaker's exclusion and re-admission
path, p95 matching `numpy.percentile`, the cache-read regression, the
projection arithmetic, the reward-ordering flip, and seed determinism.

---

## 📄 License
MIT © 2026 Arun Tewatia
