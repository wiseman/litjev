# HTTP API

Interactive API documentation is served at **http://127.0.0.1:8000/docs** while
the server runs. The server binds to loopback only, with one process owning the model.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

## Request

```json
{
  "model": "litjev",
  "state": "Answer each question using its listed options.",
  "questions": {
    "math": {
      "type": "choice",
      "instructions": "What is 2 + 3?",
      "criteria": {"A": "4", "B": "5", "C": "6"}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is the state?",
      "criteria": ["Not urgent", "Urgent", "Critical"]
    },
    "needs_review": {
      "type": "noul",
      "instructions": "Does this need human review?"
    }
  }
}
```

This is a **Jev questions mapping, not JSON Schema**. The bundled request and
playground example contain the same ten questions. `state` accepts a string, object,
or array. `instructions` and criterion descriptions accept strings, objects, arrays,
or null. `instructions` may be omitted. Question IDs are never sent to the model.

Request `model` must be `litjev` (local alias) or the loaded checkpoint ID; the
response reports the actual checkpoint, never pretends to be `jev-latest`.

## Question types

| Type | Criteria | Answer fields |
| --- | --- | --- |
| `choice` | Map of up to 255 option keys to descriptions (see [Option cap](#option-cap)) | `type`, `choice`, `probabilities`, `confidence` |
| `score` | Ordered array of 2–10 level descriptions | `type`, `score`, `legend`, `probabilities`, `confidence` |
| `noul` | Optional map with `true` and/or `false` descriptions | `type`, `noul` |

Choice returns the original option key, even when it contains multiple tokens.
Score returns the probability-weighted level index (0-based), not the winning level.
Noul returns P(yes), not a boolean. Score `legend` preserves original descriptions.

### Option cap

The 255-option limit on `choice` is a server default, not a model limit: every
option gets an internal single-token letter code at the answer boundary, and the
Qwen 3.x tokenizers supply well over 1,000 of them. Start the server with
`--max-choices N` to accept larger option lists (e.g. `litjev --max-choices 400`).
The server checks at startup that the tokenizer can supply `N` codes; a request
over the cap is rejected with a validation error naming the limit. Prompt length
still grows with the option list, so very large lists also need the input to fit
the model's context budget. When serving through a vLLM backend, the vLLM server's
`--max-logprobs` must be at least the cap, because all option codes of a decision
are requested as logprobs.

## Response

The standard response contains **only** `model`, `answers`, and
`usage: {input_tokens, output_tokens}`. `output_tokens` is zero because no answer
tokens are generated.

## Debug endpoint

For timings and raw logits, POST the same body to **`/v1/systemone/debug`**.
Its envelope is `{result, diagnostics}`: `result` is the standard response;
`diagnostics` holds `fields.*.{logits, probabilities, max_probability, provenance}`,
`forward_calls`, `confidence_method`, `calibration_fitted`, and `timing`.
The playground and visual games use this extension. Only the debug API accepts an
optional base64 `image` field. `GET /health` stays lightweight.

The playground displays probability bars, JSON, logit provenance, and timing.
`model_setup_seconds` includes first-use loading; `decision_seconds` includes
tokenization, inference-lock waiting, and inference. `total_seconds` sums these
server phases. Browser round-trip time also includes transport. These are not
GPU-kernel-only timings.

## Removed routes

The old `/v1/calibrated-schema` and `/v1/batch-mcq` routes have been removed;
old `schema`/`enum`/`boolean` request objects are rejected rather than silently
converted. See the [migration matrix](jev-schema.md).

## Python usage

```python
from dataclasses import asdict

from litjev.backend import ModelSettings, TransformersScorer
from litjev.decision import SchemaDecisionEngine
from litjev.schema import DecisionSchema

model = "Qwen/Qwen3.8-27B"
scorer = TransformersScorer.load(ModelSettings(model_id=model))
engine = SchemaDecisionEngine(scorer, model_id=model)
schema = DecisionSchema.from_mapping({
    "math": {
        "type": "choice",
        "instructions": "What is 2 + 3?",
        "criteria": {"A": "4", "B": "5"},
    }
})
result = engine.decide("Choose the correct answer.", schema)
print(result.answers["math"].choice)
print(asdict(result))
# For logits and provenance: engine.evaluate(state, schema) -> Evaluation(result, diagnostics)
```
