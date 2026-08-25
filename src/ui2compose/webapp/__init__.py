# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Streamlit user interface for ui2compose."""

import subprocess
import sys
from pathlib import Path

DEMO_SCRIPT = Path(__file__).parent / "demo.py"


def launch() -> int:
    """Starts the Streamlit demo via `streamlit run`.

    Input:  none (extra arguments are forwarded from the command line)
    Output: the exit code of the streamlit process
    """
    cmd = [sys.executable, "-m", "streamlit", "run", str(DEMO_SCRIPT), *sys.argv[1:]]
    return subprocess.call(cmd)
