"""Pytest configuration and fixtures for tinbox tests."""

import sys
import pytest
import structlog
from rich.console import Console


@pytest.fixture(autouse=True)
def reset_logging_and_console():
    """Reset structlog and Rich Console between tests.

    This prevents test pollution from:
    1. Cached loggers holding references to file handles (stderr) that may
       have been captured/redirected by previous tests
    2. Rich Console objects with corrupted file handles after partial patching
       (e.g., patching only console.print instead of the whole console)
    3. Module-level logger objects holding stale file handles

    The issue occurs because:
    - structlog is configured with cache_logger_on_first_use=True
    - CLI tests patch tinbox.cli.console or use CliRunner which captures stderr
    - After the test exits, cached loggers and console objects hold stale file handles
    - Subsequent tests fail with "I/O operation on closed file"
    """
    # Reset structlog's logger cache before each test
    structlog.reset_defaults()

    # Reconfigure structlog with fresh stderr reference
    from tinbox.utils.logging import configure_logging
    configure_logging()

    # Reset the CLI module's console and logger with fresh instances
    import tinbox.cli
    tinbox.cli.console = Console(force_terminal=False)
    tinbox.cli.logger = structlog.get_logger()

    yield

    # Clean up after test - recreate fresh instances
    structlog.reset_defaults()
    from tinbox.utils.logging import configure_logging
    configure_logging()

    import tinbox.cli
    tinbox.cli.console = Console(force_terminal=False)
    tinbox.cli.logger = structlog.get_logger()
