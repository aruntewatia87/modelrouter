# Design notes

## Why bandits

Routing across model arms is a sequential decision problem with delayed,
noisy feedback: you only learn the quality of an answer after the user sees
it, and the same request can fail for reasons unrelated to the model
(rate limits, timeouts, a bad prompt). A static table cannot express
"this arm is good *for this kind of request* and was good five minutes ago".
A bandit does, because it treats each arm as an experiment with a running
estimate of its value and a principled bonus for how little it has been
tried.

Four policies are provided because they fail in different ways:

| Policy | Best when | Weakness |
|---|---|---|
| `EpsilonGreedy` | you want predictable, tunable exploration | fixed exploration rate wastes traffic on bad arms |
| `UCB1` | you want deterministic, inspectable decisions | assumes a stationary reward distribution |
| `ThompsonSampling` | you want uncertainty-driven exploration | needs a binary reward; stochastic decisions |
| `SoftmaxRouter` | you want a smooth cost-quality trade-off | temperature is hard to tune |

`UCB1` is the default recommendation: its score is one line of arithmetic you
can check by hand, and its tie-breaking is deterministic.

## Why reward shaping is the core

A bandit needs one number. LLM serving produces four (success, quality, cost,
latency) plus a fifth that is hard to observe (is the answer *actually* good?).
The reward shaper is where the operator's policy lives:

```
reward = quality * success
         - lambda_cost * normalized_cost
         - lambda_latency * normalized_latency
```

`lambda_cost` is the exchange rate between money and quality. Setting it is a
business decision, not a technical one, which is why it is the most prominent
knob in the API. The shaper normalizes against a reference (usually the most
expensive arm), so the lambdas are dimensionless and comparable across
deployments.

## Why the filters are hard constraints

A bandit optimises the reward it is given. If an arm that cannot do tool
calls has a higher expected reward, the bandit will happily pick it. A
capability mismatch is not a low-reward outcome, it is an *impossible* one,
so it must be removed before the bandit ever sees the arm. The same applies
to the budget ceiling: exceeding it is not a trade-off, it is a violation.

Order of operations: capability -> budget -> kill switch -> health -> bandit.

## Why the kill switch uses a rolling rate

A lifetime success rate is monotone: an arm that failed during a 10-minute
outage can never fully recover its score. Using the rolling rate means a
transient outage is forgiven, but a *sustained* outage is not. The killed arm
is re-probed after a cool-down so it can prove itself again, which closes the
loop without any operator intervention.

## Why the health breaker is separate from the kill switch

They look similar but answer different questions:

- The **kill switch** is about *this arm, over all traffic* -- "is this model
  broken right now?"
- The **health breaker** is about *this arm, over a recent window* -- "is this
  model degraded relative to its own baseline?"

They can disagree, and when they do the router should trust the more recent
signal. The health tracker additionally considers latency against the arm's
own baseline, so a model that is slow but reliable is not treated the same as
one that is fast and broken.

## The cost model

Three independent prices, because providers price them independently:

```
cost = (uncached_input / 1000) * input_cost_per_1k
     + (cached_input  / 1000) * cache_read_cost_per_1k
     + (completion     / 1000) * output_cost_per_1k
```

Cache reads are usually ~10x cheaper than fresh input. Omitting them inflates
the cost of a cache-warm provider, which can invert a price comparison and
make "always cheapest" pick the more expensive provider. LiteLLM issue
#38064 is exactly this bug, and `tests/test_cost.py` pins the behaviour so it
cannot regress.

## Complexity classification

`RequestContext.is_complex` is a deterministic heuristic, deliberately:

- **Free.** A paid classifier is spend nobody tracks (LiteLLM #46469).
- **Deterministic.** The same prompt always gets the same treatment, so
  routing decisions are reproducible.
- **Cheap.** It runs inline in the request path with no added latency.
- **Inspectable.** A handful of rules a reviewer can read, versus a model
  whose behaviour nobody can predict.

The signal is intentionally coarse. It is a prior, not a verdict: the bandit
still learns the truth from outcomes.

## Determinism

Every stochastic component draws from a seeded `numpy.random.Generator`. The
same seed produces the same routing decisions, the same test results and the
same example output. This is what makes the test suite trustworthy and the
README's numbers reproducible.
