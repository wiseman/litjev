"""Two forwards: shared prefix prefill, then independent cached answer branches.

The optional slow path re-runs escalated branches with the backbone's own thinking
mode and reads the same answer boundary again. The backbone is never modified.
"""

import copy
import threading
from dataclasses import dataclass

import torch
from transformers.generation import StoppingCriteria, StoppingCriteriaList

from litjev.decision import RawFieldScores
from litjev.prompting import (
    ANSWER_BOUNDARY,
    build_decision_messages,
    build_thinking_messages,
    question_body,
)
from litjev.slots import SLOT_FORMAT, compile_prefix_slots, compile_slots
from litjev.vision import VisualState, validate_image

THINK_TOKENS = ("<think>", "</think>")


@dataclass(frozen=True)
class ModelSettings:
    model_id: str = "Qwen/Qwen3.8-27B"
    revision: str = "main"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_input_tokens: int = 16384
    feature_layers: tuple[int, ...] = ()


class SuffixStop(StoppingCriteria):
    """Stop a row once its generated tail equals the closing delimiter."""

    def __init__(self, suffix, prompt_width):
        self.suffix = suffix
        self.prompt_width = prompt_width

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.prompt_width :]
        if generated.shape[1] < len(self.suffix):
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        tail = torch.tensor(self.suffix, device=input_ids.device)
        return (generated[:, -len(self.suffix) :] == tail).all(dim=1)


class TransformersScorer:
    def __init__(
        self,
        model,
        tokenizer,
        max_input_tokens=16384,
        processor=None,
        feature_layers=(),
        think_tokens=THINK_TOKENS,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.lock = threading.Lock()
        self.processor = processor
        self.feature_layers = tuple(feature_layers)
        self.think_tokens = think_tokens

    @classmethod
    def load(cls, settings: ModelSettings):
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        config = AutoConfig.from_pretrained(settings.model_id, revision=settings.revision)
        loader = (
            AutoModelForImageTextToText if config.model_type == "qwen3_5" else AutoModelForCausalLM
        )
        model = loader.from_pretrained(
            settings.model_id,
            revision=settings.revision,
            dtype=getattr(torch, settings.dtype),
            device_map=settings.device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
        processor = (
            AutoProcessor.from_pretrained(settings.model_id, revision=settings.revision)
            if config.model_type == "qwen3_5"
            else None
        )
        return cls(model, tokenizer, settings.max_input_tokens, processor, settings.feature_layers)

    @property
    def hidden_size(self):
        config = getattr(self.model.config, "text_config", self.model.config)
        return config.hidden_size

    def score(self, state, schema):
        with self.lock, torch.inference_mode():
            return self._score(state, schema)

    def think(self, state, schema, names, budget):
        """Slow path for the named questions: backbone thinking, then the same readout."""
        with self.lock, torch.inference_mode():
            return self._think(state, schema, tuple(names), budget)

    def _compile(self, state, schema):
        return compile_slots(self.tokenizer, state, schema, self.max_input_tokens)

    def _pad_id(self):
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer requires a padding or EOS token")
        return pad

    def _readout_hidden(self, output, row, position):
        if not self.feature_layers:
            return None
        states = output.hidden_states
        if states is None:
            raise RuntimeError("Model did not return hidden states")
        return (
            torch.stack([states[layer][row, position] for layer in self.feature_layers])
            .float()
            .cpu()
            .numpy()
        )

    def _prepare(self, state, schema):
        device = self.model.get_input_embeddings().weight.device
        if isinstance(state, VisualState):
            if self.processor is None or self.model.config.model_type != "qwen3_5":
                raise ValueError("Image decisions require a Qwen qwen3_5 model and processor")
            validate_image(state.image)
            messages = build_decision_messages(state.text, schema)
            messages[-1]["content"] = [
                {"type": "text", "text": state.text},
                {"type": "image", "image": state.image.convert("RGB")},
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
            if "pixel_values" not in inputs or "mm_token_type_ids" not in inputs:
                raise ValueError("Processor must return pixels and multimodal token types")
            compiled = compile_prefix_slots(
                self.tokenizer,
                schema,
                inputs["input_ids"][0].tolist(),
                max_input_tokens=self.max_input_tokens,
            )
            return compiled, {key: value.to(device) for key, value in inputs.items()}
        compiled = self._compile(state, schema)
        prefix = compiled.input_ids[0][: compiled.prefix_length]
        prefix_ids = torch.tensor([prefix], device=device)
        return compiled, {"input_ids": prefix_ids, "attention_mask": torch.ones_like(prefix_ids)}

    @staticmethod
    def _length_groups(lengths, ratio=2.0):
        """Group branch indices so no branch is padded beyond `ratio`x its own length."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        groups, current = [], []
        for i in order:
            if current and lengths[i] > ratio * lengths[current[0]]:
                groups.append(current)
                current = []
            current.append(i)
        if current:
            groups.append(current)
        return groups

    def _score(self, state, schema):
        compiled, inputs = self._prepare(state, schema)
        device = inputs["input_ids"].device
        prefix_length = compiled.prefix_length
        base = self.model(**inputs, use_cache=True, logits_to_keep=1)
        cache = base.past_key_values
        if getattr(cache, "reorder_cache", None) is None:
            raise RuntimeError("Model cache does not support branch replication")
        suffixes = compiled.slot_ids
        pad = self._pad_id()
        delta = None
        if isinstance(state, VisualState):
            # Image patches consume sequence slots but have 3-D rotary coordinates.
            # Continue after the image prefix's M-RoPE extent, not its token count.
            delta = self.model.model.rope_deltas
            if delta is None or delta.shape[0] != 1:
                raise RuntimeError("Missing single-image-prefix M-RoPE state")
            delta = delta.to(device)
        # Branches padded to one common width re-process the longest suffix once per
        # question, so a short noul/score question pays for a long option list. Group
        # branches by length; every group replays the same state prefix cache.
        groups = self._length_groups([len(row) for row in suffixes])
        logits = [None] * len(suffixes)
        hidden = [None] * len(suffixes)
        rope = [None] * len(suffixes)
        for g, group in enumerate(groups):
            # Supports Qwen hybrid attention: both KV and convolution/recurrent states.
            group_cache = copy.deepcopy(cache) if g < len(groups) - 1 else cache
            group_cache.reorder_cache(torch.zeros(len(group), dtype=torch.long, device=device))
            rows = [suffixes[i] for i in group]
            width = max(map(len, rows))
            ids = torch.tensor([row + [pad] * (width - len(row)) for row in rows], device=device)
            lengths = torch.tensor(list(map(len, rows)), device=device)
            suffix_mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
            mask = torch.cat(
                [
                    torch.ones((len(group), prefix_length), device=device, dtype=torch.long),
                    suffix_mask.long(),
                ],
                dim=1,
            )
            positions = torch.arange(prefix_length, prefix_length + width, device=device)
            positions = positions[None, :].expand(len(group), -1)
            if delta is not None:
                positions = (positions + delta)[None, :, :].expand(3, -1, -1)
            # Only the readout positions need vocabulary logits.
            keep = sorted({len(suffixes[i]) - 1 for i in group})
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=group_cache,
                use_cache=True,
                output_hidden_states=bool(self.feature_layers),
                logits_to_keep=torch.tensor(keep, device=device),
            )
            for r, i in enumerate(group):
                last = len(suffixes[i]) - 1
                logits[i] = (
                    output.logits[r, keep.index(last), compiled.candidates[i]].float().cpu().numpy()
                )
                hidden[i] = self._readout_hidden(output, r, last)
                rope[i] = positions[:, r, last].tolist() if positions.ndim == 3 else None
        config = getattr(self.model.config, "text_config", self.model.config)
        return tuple(
            RawFieldScores(
                name,
                logits[i],
                prefix_length + sum(map(len, suffixes)),
                {
                    "method": "two_forward_cached_branches",
                    "system": "one",
                    "slot_format": SLOT_FORMAT,
                    "module": "lm_head",
                    "last_decoder_layer_index": config.num_hidden_layers - 1,
                    "decoder_layer_count": config.num_hidden_layers,
                    "batch_index": i,
                    "branch_groups": len(groups),
                    "selected_logit_index": len(suffixes[i]) - 1,
                    "prefix_token_count": compiled.prefix_length,
                    "slot_text": compiled.slot_texts[i],
                    "slot_token_ids": compiled.slot_ids[i],
                    "absolute_position": compiled.positions[i],
                    "candidate_labels": list(schema[name].choices),
                    "candidate_codes": compiled.candidate_codes[i],
                    "candidate_token_ids": compiled.candidates[i],
                    "feature_layers": list(self.feature_layers),
                    "observation_modality": "image" if isinstance(state, VisualState) else "text",
                    "image_grid_thw": inputs["image_grid_thw"].tolist()
                    if "image_grid_thw" in inputs
                    else None,
                    "readout_rope_positions": rope[i],
                },
                hidden=hidden[i],
            )
            for i, name in enumerate(schema.names)
        )

    def _think(self, state, schema, names, budget):
        if isinstance(state, VisualState):
            raise TypeError("The slow path does not support image states yet")
        if budget <= 0:
            raise ValueError("Thinking budget must be positive")
        unknown = [name for name in names if name not in schema]
        if unknown or not names:
            raise ValueError("Slow path names must be non-empty schema question IDs")
        compiled = self._compile(state, schema)
        device = self.model.get_input_embeddings().weight.device

        def encode(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        open_text, close_text = self.think_tokens
        close_ids = encode(close_text)
        tail_ids = encode("\n" + close_text + "\n" + ANSWER_BOUNDARY)
        if not encode(open_text) or not close_ids:
            raise ValueError("Tokenizer cannot encode the thinking delimiters")
        index = {name: i for i, name in enumerate(schema.names)}
        prompts = []
        for name in names:
            i = index[name]
            # The slow prompt is rendered with thinking enabled, so it carries no empty
            # <think></think> block; Qwen's template then ends the prompt with "<think>\n".
            text = self.tokenizer.apply_chat_template(
                build_thinking_messages(
                    state, question_body(schema[name], compiled.candidate_codes[i])
                ),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if not text.rstrip().endswith(open_text):
                text += open_text + "\n"
            prompts.append(encode(text))
        if max(map(len, prompts)) + budget + len(tail_ids) > self.max_input_tokens:
            raise ValueError("Thinking budget exceeds the input token limit")
        pad = self._pad_id()
        width = max(map(len, prompts))
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in prompts], device=device)
        mask = torch.tensor(
            [[0] * (width - len(row)) + [1] * len(row) for row in prompts], device=device
        )
        generated = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=budget,
            do_sample=False,
            pad_token_id=pad,
            stopping_criteria=StoppingCriteriaList([SuffixStop(close_ids, width)]),
        )
        eos = self.tokenizer.eos_token_id
        thoughts, sequences = [], []
        for row, prompt in zip(generated[:, width:].tolist(), prompts, strict=True):
            while row and row[-1] in {pad, eos}:
                row.pop()
            if len(row) >= len(close_ids) and row[-len(close_ids) :] == close_ids:
                row = row[: -len(close_ids)]
            thoughts.append(row)
            sequences.append(prompt + row + tail_ids)
        width = max(map(len, sequences))
        ids = torch.tensor([row + [pad] * (width - len(row)) for row in sequences], device=device)
        mask = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in sequences], device=device
        )
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            output_hidden_states=bool(self.feature_layers),
        )
        decode = getattr(self.tokenizer, "decode", None)
        results = []
        for j, name in enumerate(names):
            i = index[name]
            position = len(sequences[j]) - 1
            results.append(
                RawFieldScores(
                    name,
                    output.logits[j, position, compiled.candidates[i]].float().cpu().numpy(),
                    len(sequences[j]),
                    {
                        "method": "slow_thinking_full_input",
                        "slow_prompt_format": "user_turn_question_thinking_v1",
                        "system": "two",
                        "slot_format": SLOT_FORMAT,
                        "module": "lm_head",
                        "selected_logit_index": position,
                        "candidate_labels": list(schema[name].choices),
                        "candidate_codes": compiled.candidate_codes[i],
                        "candidate_token_ids": compiled.candidates[i],
                        "thinking_tokens": len(thoughts[j]),
                        "thinking_text": decode(thoughts[j]) if decode else None,
                        "budget": budget,
                        "feature_layers": list(self.feature_layers),
                    },
                    hidden=self._readout_hidden(output, j, position),
                    generated_tokens=len(thoughts[j]),
                )
            )
        return tuple(results)
