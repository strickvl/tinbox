"""Diagnostic utilities for checking Tinbox setup and dependencies."""

import os
import shutil
from typing import Optional

from pydantic import BaseModel, Field


class DoctorCheck(BaseModel):
    """Result of a single diagnostic check."""

    name: str
    category: str
    ok: bool
    details: str | None = None
    hint: str | None = None


class DoctorReport(BaseModel):
    """Complete diagnostic report."""

    checks: list[DoctorCheck] = Field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        """Return True if all checks passed."""
        return all(check.ok for check in self.checks)

    @property
    def required_ok(self) -> bool:
        """Return True if all required checks passed (excludes optional API keys and local models)."""
        optional_categories = {"API Keys", "Local Models"}
        required_checks = [
            c for c in self.checks if c.category not in optional_categories
        ]
        return all(check.ok for check in required_checks)


def check_poppler() -> DoctorCheck:
    """Check if poppler-utils is installed."""
    pdfinfo_path = shutil.which("pdfinfo")
    if pdfinfo_path:
        return DoctorCheck(
            name="poppler-utils",
            category="System Tools",
            ok=True,
            details=f"Found at {pdfinfo_path}",
        )
    return DoctorCheck(
        name="poppler-utils",
        category="System Tools",
        ok=False,
        details="Not found in PATH",
        hint="Install poppler: brew install poppler (macOS) or apt-get install poppler-utils (Linux)",
    )


def check_pdf2image() -> DoctorCheck:
    """Check if pdf2image is installed."""
    try:
        import pdf2image

        version = getattr(pdf2image, "__version__", "unknown")
        return DoctorCheck(
            name="pdf2image",
            category="Python Packages",
            ok=True,
            details=f"Version {version}",
        )
    except ImportError:
        return DoctorCheck(
            name="pdf2image",
            category="Python Packages",
            ok=False,
            details="Not installed",
            hint="Install with: pip install tinbox[pdf]",
        )


def check_python_docx() -> DoctorCheck:
    """Check if python-docx is installed."""
    try:
        import docx

        # python-docx doesn't have __version__ in the main module
        return DoctorCheck(
            name="python-docx",
            category="Python Packages",
            ok=True,
            details="Installed",
        )
    except ImportError:
        return DoctorCheck(
            name="python-docx",
            category="Python Packages",
            ok=False,
            details="Not installed",
            hint="Install with: pip install tinbox[docx]",
        )


def check_pillow() -> DoctorCheck:
    """Check if Pillow is installed."""
    try:
        import PIL
        from PIL import Image

        version = getattr(PIL, "__version__", "unknown")
        return DoctorCheck(
            name="Pillow",
            category="Python Packages",
            ok=True,
            details=f"Version {version}",
        )
    except ImportError:
        return DoctorCheck(
            name="Pillow",
            category="Python Packages",
            ok=False,
            details="Not installed",
            hint="Install with: pip install pillow",
        )


def check_openai_key() -> DoctorCheck:
    """Check if OpenAI API key is set."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        # Mask the key for display
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        return DoctorCheck(
            name="OPENAI_API_KEY",
            category="API Keys",
            ok=True,
            details=f"Set ({masked})",
        )
    return DoctorCheck(
        name="OPENAI_API_KEY",
        category="API Keys",
        ok=False,
        details="Not set",
        hint="Set with: export OPENAI_API_KEY='your-key-here'",
    )


def check_anthropic_key() -> DoctorCheck:
    """Check if Anthropic API key is set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        return DoctorCheck(
            name="ANTHROPIC_API_KEY",
            category="API Keys",
            ok=True,
            details=f"Set ({masked})",
        )
    return DoctorCheck(
        name="ANTHROPIC_API_KEY",
        category="API Keys",
        ok=False,
        details="Not set",
        hint="Set with: export ANTHROPIC_API_KEY='your-key-here'",
    )


def check_google_key() -> DoctorCheck:
    """Check if Google API key is set."""
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        return DoctorCheck(
            name="GOOGLE_API_KEY",
            category="API Keys",
            ok=True,
            details=f"Set ({masked})",
        )
    return DoctorCheck(
        name="GOOGLE_API_KEY",
        category="API Keys",
        ok=False,
        details="Not set",
        hint="Set with: export GOOGLE_API_KEY='your-key-here'",
    )


def check_ollama() -> DoctorCheck:
    """Check if Ollama is available."""
    ollama_path = shutil.which("ollama")
    if ollama_path:
        return DoctorCheck(
            name="Ollama",
            category="Local Models",
            ok=True,
            details=f"Found at {ollama_path}",
        )
    return DoctorCheck(
        name="Ollama",
        category="Local Models",
        ok=False,
        details="Not found in PATH",
        hint="Install from: https://ollama.ai (optional, for local models)",
    )


def run_doctor_checks() -> DoctorReport:
    """Run all diagnostic checks and return a report.

    Returns:
        DoctorReport with all check results
    """
    checks = [
        # System tools
        check_poppler(),
        # Python packages
        check_pdf2image(),
        check_python_docx(),
        check_pillow(),
        # API keys
        check_openai_key(),
        check_anthropic_key(),
        check_google_key(),
        # Local models
        check_ollama(),
    ]

    return DoctorReport(checks=checks)
