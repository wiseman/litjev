"""Compile independent question branches sharing only the state prefix."""

from dataclasses import dataclass
from itertools import product
from string import ascii_uppercase

from litjev.prompting import build_decision_messages, question_suffix

SLOT_FORMAT = "isolated_question_codes_v2"


@dataclass(frozen=True)
class CompiledSlots:
    input_ids: list[list[int]]
    positions: list[int]
    candidates: list[list[int]]
    slot_texts: list[str]
    slot_ids: list[list[int]]
    prefix_text: str
    prefix_length: int
    candidate_codes: list[list[str]]


TAIL_PROBE_CHARS = 256
_CODE_CACHE = {}


def candidate_codes(tokenizer, count):
    """Deterministic letter codes, accepting only exact one-token continuations (cached)."""
    key = (id(tokenizer), count)
    cached = _CODE_CACHE.get(key)
    if cached is not None:
        return cached
    codes = _candidate_codes(tokenizer, count)
    _CODE_CACHE[key] = codes
    return codes


def _candidate_codes(tokenizer, count):
    boundary = "Answer:"
    prefix = tokenizer.encode(boundary, add_special_tokens=False)
    codes, seen = [], set()
    for width in range(1, 4):
        for letters in product(ascii_uppercase, repeat=width):
            code = "".join(letters)
            ids = tokenizer.encode(boundary + " " + code, add_special_tokens=False)
            if len(ids) == len(prefix) + 1 and ids[:-1] == prefix and ids[-1] not in seen:
                codes.append(code)
                seen.add(ids[-1])
                if len(codes) == count:
                    return codes
    raise ValueError(f"Tokenizer cannot supply {count} distinct single-token answer codes")


def candidate_code_ids(tokenizer, codes):
    """Token id of each code after the answer boundary; one short encode per code (cached)."""
    key = (id(tokenizer), "ids", tuple(codes))
    cached = _CODE_CACHE.get(key)
    if cached is not None:
        return cached
    ids = _candidate_code_ids(tokenizer, codes)
    _CODE_CACHE[key] = ids
    return ids


def _candidate_code_ids(tokenizer, codes):
    boundary = "Answer:"
    prefix = tokenizer.encode(boundary, add_special_tokens=False)
    ids = {}
    for code in codes:
        encoded = tokenizer.encode(boundary + " " + code, add_special_tokens=False)
        if encoded[:-1] != prefix or len(encoded) != len(prefix) + 1:
            raise ValueError(f"Internal code {code!r} is not a single token at the answer boundary")
        ids[code] = encoded[-1]
    if len(set(ids.values())) != len(ids):
        raise ValueError("Candidate token collision")
    return ids


def compile_slots(tokenizer, state, schema, max_input_tokens=16384):
    prefix = tokenizer.apply_chat_template(
        build_decision_messages(state, schema),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    return compile_prefix_slots(tokenizer, schema, prefix_ids, prefix, max_input_tokens)


def compile_prefix_slots(tokenizer, schema, prefix_ids, prefix="", max_input_tokens=16384):
    """Share label-boundary checks between text and processor-expanded image prefixes."""
    prefix_length = len(prefix_ids)
    rows = []
    positions, candidates, texts, slot_ids = [], [], [], []
    codes = candidate_codes(tokenizer, max(len(field.choices) for field in schema.values()))
    code_ids = candidate_code_ids(tokenizer, codes)
    row_codes = []
    for field in schema.values():
        labels = codes[: len(field.choices)]
        text = question_suffix(field, labels)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Re-encoding the full suffix once per option is O(options x suffix tokens) of CPU
        # work. Each code's token id at the "Answer:" boundary comes from candidate_code_ids
        # instead, and one probe per question checks that this suffix still ends on a clean
        # boundary. The probe encodes only the tail of the text: byte-level BPE
        # pre-tokenization splits at whitespace/punctuation, so the tokens of the last
        # TAIL_PROBE_CHARS characters do not depend on what precedes them.
        tail = text[-TAIL_PROBE_CHARS:]
        tail_ids = tokenizer.encode(tail, add_special_tokens=False)
        probe = tokenizer.encode(tail + " " + labels[0], add_special_tokens=False)
        if (
            probe[:-1] != tail_ids
            or len(probe) != len(tail_ids) + 1
            or probe[-1] != code_ids[labels[0]]
        ):
            raise ValueError(
                f"Tokenizer must encode internal code {labels[0]!r} as one token at the answer boundary"
            )
        choices = [code_ids[code] for code in labels]
        if len(set(choices)) != len(choices):
            raise ValueError("Candidate token collision")
        # Exactly the original prefix IDs followed by this branch's suffix IDs.
        row = prefix_ids + tokens
        rows.append(row)
        positions.append(len(row) - 1)
        candidates.append(choices)
        texts.append(text)
        slot_ids.append(tokens)
        row_codes.append(labels)
    if max(map(len, rows)) > max_input_tokens:
        raise ValueError("Request exceeds input token limit; no truncation performed")
    return CompiledSlots(
        rows, positions, candidates, texts, slot_ids, prefix, prefix_length, row_codes
    )
