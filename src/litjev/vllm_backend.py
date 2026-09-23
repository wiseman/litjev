"""LogitProvider backed by a vLLM OpenAI-compatible server.

Same prompt, same token ids and same readout as the transformers path. One decision becomes ONE
completions request carrying every question's compiled row (state prefix + question suffix) as a
batch of prompts, generating exactly one token each with sampling restricted to the union of the
option-code tokens. vLLM returns the log-probabilities of those codes after the restriction
(processed logprobs); restricting each question to its own codes and renormalizing is the same
softmax LitJev computes over lm_head logits. Prefix caching in vLLM shares the state prefix across
the questions and across requests with the same state; the server holds no lock, so concurrent
decisions overlap.

Required vLLM server flags (tested with vllm/vllm-openai:v0.25.1):

    vllm serve --model <same model as litjev --model> \
        --logprobs-mode processed_logprobs \
        --max-logprobs <N >= largest option count in any decision, e.g. 320> \
        --enable-prefix-caching

processed_logprobs makes vLLM report log-probabilities after allowed_token_ids is applied (the
default raw mode reports the unrestricted distribution); --max-logprobs must cover the union of
option codes (LitJev asks for logprobs=len(union)) or vLLM rejects the request; prefix caching
makes one state prefix shared across questions cheap. Then run LitJev with
``litjev --backend vllm --vllm-url http://HOST:PORT --model <model>``.

The model id passed to LitJev is used only for the tokenizer; it must match the served weights'
tokenizer. The vLLM backend has no hidden states (no decision heads) and no slow thinking path.
"""

import http.client
import json
import threading
from urllib.parse import urlparse

import numpy as np
from transformers import AutoTokenizer

from litjev.decision import RawFieldScores
from litjev.slots import SLOT_FORMAT, compile_slots
from litjev.vision import VisualState

METHOD = "vllm_restricted_logprobs_batched_v2"


class VLLMScorer:
    def __init__(self, url, model_id, served_model, tokenizer, max_input_tokens=16384):
        parsed = urlparse(url)
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.model_id = model_id
        self.served_model = served_model
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self._local = threading.local()

    @classmethod
    def load(cls, url, model_id, revision="main", max_input_tokens=16384):
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=30)
        conn.request("GET", "/v1/models")
        models = json.load(conn.getresponse())["data"]
        conn.close()
        return cls(url, model_id, models[0]["id"], tokenizer, max_input_tokens)

    @property
    def hidden_size(self):
        raise RuntimeError("The vLLM backend does not expose hidden states (no decision head)")

    def _connection(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
            self._local.conn = conn
        return conn

    def _post(self, body):
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Connection": "keep-alive"}
        for attempt in range(2):
            conn = self._connection()
            try:
                conn.request("POST", "/v1/completions", body=payload, headers=headers)
                response = conn.getresponse()
                data = response.read()
                if response.status != 200:
                    raise RuntimeError(f"vLLM {response.status}: {data[:300]!r}")
                return json.loads(data)
            except (http.client.HTTPException, ConnectionError, OSError):
                conn.close()
                self._local.conn = None
                if attempt:
                    raise

    def score(self, state, schema):
        if isinstance(state, VisualState):
            raise TypeError("The vLLM backend serves text states only")
        compiled = compile_slots(self.tokenizer, state, schema, self.max_input_tokens)
        union = sorted({t for row in compiled.candidates for t in row})
        body = {
            "model": self.served_model,
            "prompt": compiled.input_ids,
            "max_tokens": 1,
            # Plain softmax over the option codes: disable every sampler the model's
            # generation_config would otherwise apply before processed logprobs are taken
            # (Qwen3.8-27B ships top_k=20 / top_p=0.95, which masks low-probability codes).
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logprobs": len(union),
            "allowed_token_ids": union,
            "return_tokens_as_token_ids": True,
        }
        result = self._post(body)
        by_index = {choice["index"]: choice for choice in result["choices"]}
        logits = []
        for i, candidates in enumerate(compiled.candidates):
            top = by_index[i]["logprobs"]["top_logprobs"][0]
            found = {
                int(key.split(":", 1)[1]): value
                for key, value in top.items()
                if key.startswith("token_id:")
            }
            missing = [t for t in candidates if t not in found]
            if missing:
                raise RuntimeError(
                    f"vLLM returned logprobs for {len(found)} tokens, missing {len(missing)} of "
                    f"{len(candidates)} option codes; serve with --logprobs-mode processed_logprobs "
                    f"and --max-logprobs >= {len(union)}"
                )
            logits.append(np.array([found[t] for t in candidates], dtype=np.float64))
        input_tokens = compiled.prefix_length + sum(map(len, compiled.slot_ids))
        usage = result.get("usage", {})
        return tuple(
            RawFieldScores(
                name,
                logits[i],
                input_tokens,
                {
                    "method": METHOD,
                    "system": "one",
                    "slot_format": SLOT_FORMAT,
                    "module": "lm_head",
                    "engine": "vllm",
                    "served_model": self.served_model,
                    "batch_index": i,
                    "selected_logit_index": len(compiled.slot_ids[i]) - 1,
                    "prefix_token_count": compiled.prefix_length,
                    "slot_text": compiled.slot_texts[i],
                    "slot_token_ids": compiled.slot_ids[i],
                    "absolute_position": compiled.positions[i],
                    "candidate_labels": list(schema[name].choices),
                    "candidate_codes": compiled.candidate_codes[i],
                    "candidate_token_ids": compiled.candidates[i],
                    "vllm_usage": usage,
                    "observation_modality": "text",
                },
            )
            for i, name in enumerate(schema.names)
        )

    def think(self, state, schema, names, budget):
        raise NotImplementedError("The vLLM backend has no slow thinking path")
