"""AnomaliRadar — read-only Streamlit viewer for frozen experiment artefacts.

Launch from the repository root::

    streamlit run app.py

The application reads ``reports/experiments`` and, for the data explorer, the
locally downloaded MetroPT-3 CSV. It does not fit, refit or retune any model.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import streamlit as st

# pandas 2.3.3 + NumPy 2.5.4 emit this on Timedelta construction; it is an
# upstream compatibility warning and is suppressed by the test configuration.
warnings.filterwarnings(
    "ignore",
    message=".*generic.*unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_lib.ui import sidebar_note  # noqa: E402

st.set_page_config(
    page_title="AnomaliRadar · MetroPT-3",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/data_explorer.py", title="Data Explorer", icon=":material/query_stats:"),
    st.Page("app_pages/anomaly_detection.py", title="Anomaly Detection", icon=":material/radar:"),
    st.Page("app_pages/alarm_details.py", title="Alarm Details", icon=":material/notification_important:"),
    st.Page("app_pages/model_comparison.py", title="Model Comparison", icon=":material/compare_arrows:"),
    st.Page("app_pages/limitations.py", title="Limitations", icon=":material/warning:"),
]

navigation = st.navigation(pages, position="sidebar")
sidebar_note()
navigation.run()
