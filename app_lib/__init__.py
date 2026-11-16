"""Presentation layer for the AnomaliRadar Streamlit application.

The application is a read-only viewer for the frozen experiment artefacts in
``reports/experiments``. It never fits, refits, calibrates or re-scores a model.
"""

from __future__ import annotations

import warnings

# pandas 2.3.3 + NumPy 2.5.4 emit this on Timedelta construction; it is an
# upstream compatibility warning and is also suppressed in the pytest config.
warnings.filterwarnings(
    "ignore",
    message=".*generic.*unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
)

__all__ = ["artifacts"]
