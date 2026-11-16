"""AnomaliRadar: state-aware, empirically calibrated anomaly detection.

The package contains the deterministic data layer, the frozen model and alarm
pipeline, and the experiment runner for MetroPT-3. The Streamlit viewer lives in
the repository-level ``app_lib`` and ``app_pages`` directories; it never refits
or re-scores the frozen artefacts.
"""

from __future__ import annotations

__version__ = "0.2.0"
