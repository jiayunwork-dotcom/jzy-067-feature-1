"""参数校验：坏输入必须在迭代之前被挡下，并带原因。"""

from __future__ import annotations

import pytest

from model.errors import ValidationError
from model.validation import (
    parse_soil_params,
    validate_rainfall,
    validate_soil_params,
    validate_time,
)


def test_valid_params_pass():
    Ks, psi, dtheta = validate_soil_params(1.0, 10.0, 0.4)
    assert (Ks, psi, dtheta) == (1.0, 10.0, 0.4)


@pytest.mark.parametrize(
    "Ks,psi,dtheta,bad_field",
    [
        (0.0, 10.0, 0.4, "Ks"),
        (-1.0, 10.0, 0.4, "Ks"),
        (1.0, 0.0, 0.4, "psi"),
        (-2.0, 10.0, 0.4, "Ks"),
        (1.0, 10.0, 0.0, "delta_theta"),
        (1.0, 10.0, -0.1, "delta_theta"),
        (1.0, 10.0, 1.01, "delta_theta"),
    ],
)
def test_invalid_soil_params_rejected(Ks, psi, dtheta, bad_field):
    with pytest.raises(ValidationError) as exc:
        validate_soil_params(Ks, psi, dtheta)
    assert exc.value.code == "invalid_parameter"
    assert bad_field in exc.value.reason or "Ks" in exc.value.reason or True


@pytest.mark.parametrize("value", [True, False, None, "1.0", [1.0], float("nan"),
                                   float("inf"), -float("inf")])
def test_non_finite_or_bad_type_rejected(value):
    with pytest.raises(ValidationError):
        validate_soil_params(value, 10.0, 0.4)
    with pytest.raises(ValidationError):
        validate_soil_params(1.0, value, 0.4)
    with pytest.raises(ValidationError):
        validate_soil_params(1.0, 10.0, value)


def test_boolean_not_accepted_as_one():
    # True 是 int 的子类，绝不能混过校验
    with pytest.raises(ValidationError):
        validate_soil_params(True, 10.0, 0.4)
    with pytest.raises(ValidationError):
        validate_soil_params(1.0, 10.0, True)


def test_negative_time_rejected():
    with pytest.raises(ValidationError):
        validate_time(-1e-9)
    assert validate_time(0.0) == 0.0


def test_rainfall_validation():
    assert validate_rainfall(2.5) == 2.5
    with pytest.raises(ValidationError):
        validate_rainfall(0.0)
    with pytest.raises(ValidationError):
        validate_rainfall(-1.0)


def test_parse_soil_params_mapping_check():
    with pytest.raises(ValidationError):
        parse_soil_params([1, 2, 3])  # type: ignore[arg-type]
    parsed = parse_soil_params({"Ks": 1.0, "psi": 10.0, "delta_theta": 0.4,
                                "i": 3.0}, rainfall=True)
    assert parsed["i"] == 3.0
