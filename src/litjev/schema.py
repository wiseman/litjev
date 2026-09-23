"""Canonical Jev questions, shared by HTTP, Python and prompt compilation."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator

# Default upper bound on Choice options. It is a server policy, not a tokenizer limit (the Qwen
# 3.x tokenizers supply well over 1,000 single-token answer codes); `litjev --max-choices`
# raises it for deployments that need larger option lists.
MAX_CHOICES = 255
_max_choices = MAX_CHOICES
Content = str | dict[str, JsonValue] | list[JsonValue] | None
State = str | dict[str, JsonValue] | list[JsonValue]


def max_choices():
    """Current Choice option cap enforced by request validation."""
    return _max_choices


def set_max_choices(limit):
    """Set the Choice option cap for this process; returns the previous cap."""
    global _max_choices
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("max_choices must be a positive integer")
    previous, _max_choices = _max_choices, limit
    return previous


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    instructions: Content = None


class Choice(QuestionBase):
    type: Literal["choice"] = "choice"
    criteria: dict[str, Content] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _within_option_cap(cls, value):
        if len(value) > _max_choices:
            raise ValueError(
                f"Choice allows at most {_max_choices} options, got {len(value)}"
                " (the server's --max-choices)"
            )
        return value

    @property
    def choices(self):
        return tuple(self.criteria)

    @property
    def descriptions(self):
        return tuple(self.criteria.values())


class Score(QuestionBase):
    type: Literal["score"] = "score"
    criteria: list[Content] = Field(min_length=2, max_length=10)

    @property
    def choices(self):
        return tuple(str(i) for i in range(len(self.criteria)))

    @property
    def descriptions(self):
        return tuple(self.criteria)


class Noul(QuestionBase):
    type: Literal["noul"] = "noul"
    criteria: dict[Literal["true", "false"], Content] | None = None

    @property
    def choices(self):
        return ("false", "true")

    @property
    def descriptions(self):
        return tuple((self.criteria or {}).get(key) for key in self.choices)


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]
QUESTION_ADAPTER = TypeAdapter(Question)


class DecisionSchema(Mapping[str, Question]):
    """Ids are bookkeeping only and must never enter model prompts."""

    def __init__(self, questions: Mapping[str, Question]):
        if not questions:
            raise ValueError("Questions must contain at least one question")
        self._questions = {
            key: QUESTION_ADAPTER.validate_python(value).model_copy(deep=True)
            for key, value in questions.items()
        }
        if any(not isinstance(key, str) for key in self._questions):
            raise ValueError("Question IDs must be strings")

    @classmethod
    def from_mapping(cls, questions):
        return cls(questions)

    def to_mapping(self):
        return {key: value.model_dump() for key, value in self.items()}

    @property
    def names(self):
        return tuple(self._questions)

    def __getitem__(self, key):
        return self._questions[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._questions)

    def __len__(self):
        return len(self._questions)


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str = Field(min_length=1)
    state: State
    questions: dict[str, Question] = Field(min_length=1)

    def to_schema(self):
        return DecisionSchema(self.questions)
