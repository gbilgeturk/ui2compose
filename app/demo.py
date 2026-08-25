#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Entry point for the Streamlit demo.

Run with:
    streamlit run app/demo.py

The user interface itself lives in `ui2compose.webapp.demo`; this file only
starts it, so that running the project and reading its code stay separate.
"""

from ui2compose.webapp.demo import main

main()
