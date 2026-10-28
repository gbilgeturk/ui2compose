# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Shared pytest configuration.

The test suite runs with only numpy, Pillow, PyYAML and pytest installed; CI
adds the layout evaluation's apted / zss / scipy. Tests that need torch,
ultralytics, OpenCV, matplotlib, apted / zss / scipy, the released
weights or the v1.1.0 reference outputs skip themselves when these are
missing; the report header lists which of them are available, so that the
number of skipped tests can be read off a test log. With
UI2COMPOSE_REQUIRE_REFERENCE_DATA=1 (set in CI) a missing reference data file
fails the test instead (see tests/_reference_data.py).
"""

import importlib.util
import os
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Optional packages whose absence makes some tests skip.
OPTIONAL_PACKAGES = ('torch', 'ultralytics', 'cv2', 'matplotlib', 'streamlit',
                     'apted', 'zss', 'scipy')

#: Optional data files whose absence makes some tests skip.
OPTIONAL_DATA = {
    'weights (default, app split)': REPO_ROOT / 'runs' / 'oversample_5k_appsplit' / 'weights' / 'best.pt',
    'weights (v1.1.0 release)': REPO_ROOT / 'runs' / 'oversample_5k' / 'weights' / 'best.pt',
    'v1.1.0 reference outputs': REPO_ROOT / 'results' / 'v1.1.0_examples',
    'v1.2.0 example outputs': REPO_ROOT / 'results' / 'v1.2.0_examples',
    'compile report': REPO_ROOT / 'results' / 'compile_report.json',
}


def pytest_report_header(config) -> List[str]:
    """Adds the availability of optional packages and data files to the pytest header.

    Input:  config — pytest config object (unused)
    Output: list of header lines
    """
    packages = ', '.join(f"{name}={'yes' if importlib.util.find_spec(name) else 'no'}"
                         for name in OPTIONAL_PACKAGES)
    data = ', '.join(f"{name}={'yes' if path.exists() else 'no'}"
                     for name, path in OPTIONAL_DATA.items())
    required = os.environ.get('UI2COMPOSE_REQUIRE_REFERENCE_DATA', '') == '1'
    return [f"ui2compose optional packages: {packages}",
            f"ui2compose optional data: {data}"
            f" (missing data {'fails' if required else 'skips'} the tests that need it)"]
