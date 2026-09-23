import pytest
from pydantic import ValidationError

from litjev.schema import (
    MAX_CHOICES,
    Choice,
    DecisionSchema,
    Noul,
    Score,
    SystemOneRequest,
    max_choices,
    set_max_choices,
)


def test_canonical_types_and_structured_content():
    schema = DecisionSchema(
        {
            "pick": Choice(instructions=["Choose"], criteria={"long key": None, "B": {"value": 2}}),
            "rate": Score(instructions=None, criteria=[None, "High"]),
            "yes": Noul(instructions="Is it?", criteria={"true": "Yes"}),
        }
    )
    assert schema["pick"].choices == ("long key", "B")
    assert schema["rate"].choices == ("0", "1")
    assert schema["yes"].choices == ("false", "true")
    assert DecisionSchema.from_mapping(schema.to_mapping()).to_mapping() == schema.to_mapping()


def test_schema_limits():
    assert len(Choice(criteria={str(i): None for i in range(255)}).choices) == 255
    with pytest.raises(ValueError):
        Choice(criteria={str(i): None for i in range(256)})
    with pytest.raises(ValueError):
        Score(criteria=[None] * 11)
    with pytest.raises(ValueError):
        DecisionSchema({})


@pytest.fixture
def option_cap():
    previous = max_choices()
    yield set_max_choices
    set_max_choices(previous)


def _request(options):
    return {
        "model": "litjev",
        "state": "s",
        "questions": {
            "pick": {"type": "choice", "criteria": {str(i): None for i in range(options)}}
        },
    }


def test_default_option_cap_is_255():
    assert max_choices() == MAX_CHOICES == 255
    with pytest.raises(ValidationError, match="at most 255 options"):
        SystemOneRequest.model_validate(_request(307))


def test_raised_option_cap_accepts_larger_choice(option_cap):
    option_cap(307)
    request = SystemOneRequest.model_validate(_request(307))
    assert len(request.to_schema()["pick"].choices) == 307
    with pytest.raises(ValidationError, match="at most 307 options"):
        SystemOneRequest.model_validate(_request(308))


def test_option_cap_must_be_positive(option_cap):
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            option_cap(bad)
    assert max_choices() == MAX_CHOICES
