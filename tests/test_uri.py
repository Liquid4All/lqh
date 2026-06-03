"""Tests for the pull/push location URI grammar."""

import pytest

from lqh.tools.uri import Location, LocationError, parse_location


def test_hf_basic():
    loc = parse_location("hf:meta-llama/Llama-3.2-1B")
    assert loc == Location("hf", "meta-llama/Llama-3.2-1B", None)


def test_hf_with_revision():
    loc = parse_location("hf:owner/repo@v2")
    assert loc.scheme == "hf" and loc.value == "owner/repo" and loc.revision == "v2"


def test_hf_requires_owner_slash():
    with pytest.raises(LocationError):
        parse_location("hf:justrepo")


def test_lqh_artifact():
    loc = parse_location("lqh:6f3a1c2d")
    assert loc == Location("lqh", "6f3a1c2d", None)


def test_local_relative_path():
    loc = parse_location("runs/run_001/model")
    assert loc.scheme == "local" and loc.value == "runs/run_001/model"


def test_local_absolute_path():
    loc = parse_location("/abs/path/model")
    assert loc.scheme == "local"


def test_unknown_scheme_rejected():
    with pytest.raises(LocationError):
        parse_location("s3:bucket/key")


def test_empty_rejected():
    with pytest.raises(LocationError):
        parse_location("   ")


def test_missing_value_rejected():
    with pytest.raises(LocationError):
        parse_location("lqh:")
