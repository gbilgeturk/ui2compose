#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT
#
# Compile + render harness for the Kotlin emitted by ui2compose.
#
# Usage: harness/compile_and_render.sh [options] <dir-or-.kt>...
#   Every *.kt below the given directories (or each given .kt file) is copied
#   into its own package generated.sNN of the Android module harness/android/screens,
#   compiled with :screens:compileDebugKotlin and rendered with Paparazzi
#   (:screens:recordPaparazziDebug).  Writes results/compile_report.json unless
#   --report is given.  Options: see `harness/compile_and_render.sh --help`
#   and harness/README.md.
#
# Environment: JAVA_HOME (default: Homebrew openjdk@17), ANDROID_HOME or
# ANDROID_SDK_ROOT (default: ~/Library/Android/sdk), PYTHON (default: python3).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$HERE/compile_and_render.py" "$@"
