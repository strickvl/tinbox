"""Progress tracking functionality for Tinbox."""

from rich.progress import ProgressColumn, Task, Text
from rich.text import Text as RichText

from tinbox.utils.logging import get_logger

logger = get_logger(__name__)


class CurrentCostColumn(ProgressColumn):
    """Displays current accumulated cost."""

    def render(self, task: Task) -> Text:
        """Render the current cost.

        Args:
            task: The progress task

        Returns:
            Rich Text object with formatted cost
        """
        cost = task.fields.get("total_cost", 0.0)
        if cost > 0:
            return RichText(f"${cost:.4f}", style="yellow")
        else:
            return RichText("$0.0000", style="dim")


class EstimatedCostColumn(ProgressColumn):
    """Displays estimated total cost based on current progress."""

    def render(self, task: Task) -> Text:
        """Render the estimated total cost.

        Args:
            task: The progress task

        Returns:
            Rich Text object with formatted estimated cost
        """
        current_cost = task.fields.get("total_cost", 0.0)

        if (
            task.total is None
            or task.total == 0
            or task.completed == 0
            or current_cost == 0
        ):
            return RichText("$-.----", style="dim")

        # Calculate estimated total cost based on progress ratio
        progress_ratio = task.completed / task.total
        if progress_ratio > 0:
            estimated_total = current_cost / progress_ratio
            return RichText(f"${estimated_total:.4f}", style="cyan")
        else:
            return RichText("$-.----", style="dim")


def format_cost(cost: float) -> str:
    """Format cost for display.

    Args:
        cost: Cost value to format

    Returns:
        Formatted cost string
    """
    if cost >= 1.0:
        return f"${cost:.2f}"
    elif cost >= 0.01:
        return f"${cost:.3f}"
    elif cost >= 0.001:
        return f"${cost:.3f}"
    else:
        return f"${cost:.4f}"
