# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Skip-or-fail helper for tests that read reference data files of the repository.

Some tests compare against files that belong to the repository (the v1.1.0 and
v1.2.0 example outputs, results/compile_report.json, the released weights,
the archived training arguments). Locally such a test skips when its files are
missing. With the environment variable UI2COMPOSE_REQUIRE_REFERENCE_DATA=1 (set
in the CI workflow) a missing file is a test failure instead, so that CI cannot
pass without the reference data being committed. Missing optional *packages*
(torch, ultralytics, ...) are not affected: those tests use pytest.importorskip.
"""

import os
from pathlib import Path
from typing import Union

import pytest

#: Environment variable that turns missing reference data into failures.
REQUIRE_ENV = 'UI2COMPOSE_REQUIRE_REFERENCE_DATA'

REPO_ROOT = Path(__file__).resolve().parents[1]


def reference_data_required() -> bool:
    """Tells whether missing reference data must fail the test run.

    Input:  none (reads the environment)
    Output: True if UI2COMPOSE_REQUIRE_REFERENCE_DATA is '1'
    """
    return os.environ.get(REQUIRE_ENV, '') == '1'


def require_reference_data(*paths: Union[str, Path]) -> None:
    """Skips the calling test (or fails it, see the module docstring) if a reference file is missing.

    Input:  paths — files or directories the test reads; a directory must exist and
            not be empty
    Output: none; calls pytest.skip / pytest.fail when a path is missing
    """
    missing = []
    for p in paths:
        p = Path(p)
        if not p.exists() or (p.is_dir() and not any(p.iterdir())):
            try:
                missing.append(str(p.relative_to(REPO_ROOT)))
            except ValueError:
                missing.append(str(p))
    if not missing:
        return
    msg = f"reference data not available: {', '.join(missing)}"
    if reference_data_required():
        pytest.fail(f"{msg} ({REQUIRE_ENV}=1 requires it)")
    pytest.skip(msg)
