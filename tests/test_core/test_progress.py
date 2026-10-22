"""Tests for progress tracking functionality."""

from unittest.mock import MagicMock

import pytest
from rich.progress import Task
from rich.text import Text

from tinbox.core.progress import CurrentCostColumn, EstimatedCostColumn, format_cost


def test_current_cost_column_render():
    """Test CurrentCostColumn rendering."""
    column = CurrentCostColumn()

    # Test with zero cost
    task = MagicMock(spec=Task)
    task.fields = {}
    result = column.render(task)
    assert isinstance(result, Text)
    assert "$0.0000" in str(result)
    assert result.style == "dim"

    # Test with positive cost
    task.fields = {"total_cost": 0.1234}
    result = column.render(task)
    assert "$0.1234" in str(result)
    assert result.style == "yellow"

    # Test with missing cost field
    task.fields = {"other_field": "value"}
    result = column.render(task)
    assert "$0.0000" in str(result)
    assert result.style == "dim"


def test_estimated_cost_column_render():
    """Test EstimatedCostColumn rendering."""
    column = EstimatedCostColumn()

    # Test with no progress
    task = MagicMock(spec=Task)
    task.fields = {"total_cost": 0.05}
    task.total = 10
    task.completed = 0
    result = column.render(task)
    assert "$-.----" in str(result)
    assert result.style == "dim"

    # Test with progress
    task.completed = 2  # 20% complete
    result = column.render(task)
    # Should estimate 0.05 / 0.2 = 0.25 total cost
    assert "$0.2500" in str(result)
    assert result.style == "cyan"

    # Test with no total
    task.total = None
    result = column.render(task)
    assert "$-.----" in str(result)
    assert result.style == "dim"

    # Test with zero total
    task.total = 0
    result = column.render(task)
    assert "$-.----" in str(result)
    assert result.style == "dim"


def test_estimated_cost_column_edge_cases():
    """Test EstimatedCostColumn edge cases."""
    column = EstimatedCostColumn()
    task = MagicMock(spec=Task)

    # Test with missing cost field
    task.fields = {}
    task.total = 10
    task.completed = 5
    result = column.render(task)
    assert "$-.----" in str(result)
    assert result.style == "dim"

    # Test with 100% completion
    task.fields = {"total_cost": 0.1}
    task.total = 10
    task.completed = 10
    result = column.render(task)
    assert "$0.1000" in str(result)
    assert result.style == "cyan"


@pytest.mark.parametrize(
    "cost,expected",
    [
        (0.0001, "$0.0001"),
        (0.001, "$0.001"),
        (0.01, "$0.010"),
        (0.1, "$0.100"),
        (1.0, "$1.00"),
        (10.5, "$10.50"),
        (123.456, "$123.46"),
    ],
)
def test_format_cost(cost, expected):
    """Test cost formatting function."""
    assert format_cost(cost) == expected


def test_format_cost_edge_cases():
    """Test format_cost edge cases."""
    assert format_cost(0) == "$0.0000"
    assert format_cost(0.00001) == "$0.0000"
    assert format_cost(0.009999) == "$0.010"
    assert format_cost(0.999) == "$0.999"
