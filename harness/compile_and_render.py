# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Compile + render harness for the Kotlin emitted by ui2compose.

What the harness does
---------------------
1. Inputs.  Every argument is a .kt file or a directory; a directory
   contributes every ``*.kt`` file below it (files under a ``build``
   directory are skipped).  The files are sorted by path and the i-th file
   (1-based) gets the Kotlin package ``generated.sNN`` (NN zero-padded to at
   least two digits), so that the identical ``GeneratedScreen`` functions do
   not clash.  An existing ``package`` line is replaced; otherwise one is
   inserted (after any ``@file:`` annotations).  The source files themselves
   are never modified.
2. Gradle project.  The copies go into the Android library module
   ``harness/android/screens`` (``src/main/java/generated/sNN/``).  For each
   file that defines ``fun GeneratedScreen()`` one Paparazzi snapshot test is
   written (``src/test/java/generated/sNN/GeneratedScreenSnapshotTest.kt``).
   It renders ``GeneratedScreen()`` inside
   ``MaterialTheme { Surface(Modifier.fillMaxSize()) { ... } }`` (the default
   Material 3 light theme; the generated code itself has no theme) on
   Paparazzi's ``DeviceConfig.NEXUS_7`` (1200 x 1920 px, xhdpi, the size of the
   ReDraw example screenshots).  Both source trees are deleted and rewritten
   on every run.
3. Compilation.  ``:screens:compileDebugKotlin`` runs (forced with
   ``--rerun``; the module compiles non-incrementally, so the diagnostics
   always cover every file).  Compiler diagnostics (``e:``/``w:`` lines) are
   mapped back to the input files.  When the build fails, the files with
   error diagnostics are marked as not compiled and left out, and the build is
   repeated with the remaining files, for at most ``MAX_COMPILE_ROUNDS``
   rounds.  A failure without any file-level error marks every remaining
   file as not compiled with the tail of the Gradle log.
4. Rendering.  ``:screens:recordPaparazziDebug`` runs on the compiled files.
   A file counts as rendered when its snapshot test passed (JUnit XML result)
   and its PNG exists.  Paparazzi stores the snapshot scaled down so that its
   longer side is at most 1000 px (625 x 1000 for the Nexus 7 frame).
5. Render files.  The PNG is copied next to its source as ``render.png`` when
   the source is the only .kt file in its directory (the pipeline layout
   ``<stem>/4_GeneratedScreen.kt``) and ``--no-copy`` is not given; otherwise
   to ``<render-dir>/<parent dir>__<file stem>.png``.  A file at the target
   path of a source that did not render is deleted, so no stale render
   survives.
6. Report.  A JSON report (default ``results/compile_report.json``) lists
   the toolchain and, per input file, its package, SHA-256, compile result
   with the compiler errors and warnings, and render result.  Paths inside
   the repository are written relative to the repository root.

With ``--compile-only`` no rendering is done, every entry has
``rendered: false``, and existing render files are left untouched.

Exit status: 0 when every file compiled and (unless ``--compile-only``)
every file with a ``GeneratedScreen()`` rendered, 1 otherwise, 2 for a usage
or setup error.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Parameters and fixed locations
# ---------------------------------------------------------------------------

#: Directory of this file (harness/).
HARNESS_DIR: Path = Path(__file__).resolve().parent

#: Repository root (paths in the report are relative to it).
REPO_ROOT: Path = HARNESS_DIR.parent

#: Gradle project of the harness.
ANDROID_DIR: Path = HARNESS_DIR / 'android'

#: Gradle module that receives the generated screens.
MODULE: str = 'screens'

#: Root package of the copied screens (generated.s01, generated.s02, ...).
PACKAGE_ROOT: str = 'generated'

#: Composable rendered by the snapshot test of each screen.
SCREEN_FUNCTION: str = 'GeneratedScreen'

#: Name of the snapshot test class written for each screen.
TEST_CLASS: str = 'GeneratedScreenSnapshotTest'

#: Paparazzi device configuration used for every snapshot.
DEVICE_CONFIG: str = 'DeviceConfig.NEXUS_7'

#: Default report path.
DEFAULT_REPORT: Path = REPO_ROOT / 'results' / 'compile_report.json'

#: Default directory for renders that cannot be placed next to their source.
DEFAULT_RENDER_DIR: Path = HARNESS_DIR / 'renders'

#: File name of a render placed next to its source.
RENDER_NAME: str = 'render.png'

#: Maximum number of compile rounds (each round drops the files with errors).
MAX_COMPILE_ROUNDS: int = 5

#: Maximum number of compiler messages kept per file and kind in the report.
MAX_MESSAGES: int = 20

#: Maximum length of a test failure message kept in the report.
MAX_MESSAGE_CHARS: int = 500

#: Number of trailing Gradle log lines kept for a failure without file-level errors.
LOG_TAIL_LINES: int = 25

#: JDK used when JAVA_HOME is not set (Homebrew openjdk@17 on macOS).
DEFAULT_JAVA_HOME: Path = Path('/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home')

#: Android SDK location used when neither ANDROID_HOME nor ANDROID_SDK_ROOT is set.
DEFAULT_SDK_DIR: Path = Path.home() / 'Library' / 'Android' / 'sdk'

_MAIN_SRC = Path(MODULE) / 'src' / 'main' / 'java' / PACKAGE_ROOT
_TEST_SRC = Path(MODULE) / 'src' / 'test' / 'java' / PACKAGE_ROOT
_SNAPSHOT_IMAGES = Path(MODULE) / 'src' / 'test' / 'snapshots' / 'images'
_TEST_RESULTS = Path(MODULE) / 'build' / 'test-results' / 'testDebugUnitTest'

_PACKAGE_RE = re.compile(r'^\s*package\s+[A-Za-z_][\w.]*\s*;?\s*$')
_FILE_ANNOTATION_RE = re.compile(r'^\s*@file:')
_SCREEN_FN_RE = re.compile(r'(?m)^\s*fun\s+' + SCREEN_FUNCTION + r'\s*\(\s*\)')
_DIAGNOSTIC_RE = re.compile(r'^(?P<kind>[ew]): (?P<path>(?:file:)?\S+?\.kt):(?P<line>\d+):(?P<col>\d+) (?P<msg>.*)$')
_RESOLVED_RE = re.compile(r'^RESOLVED (?P<module>\S+)=(?P<version>\S+)$')


# ---------------------------------------------------------------------------
# Inputs and source rewriting (pure)
# ---------------------------------------------------------------------------

def collect_sources(inputs: Sequence[str]) -> List[Path]:
    """Collects the .kt files to check from files and directories.

    Input:  inputs — paths of .kt files or of directories (searched
            recursively for *.kt; files below a 'build' directory are skipped)
    Output: absolute, de-duplicated paths sorted by path
    Raises: FileNotFoundError for a missing path; ValueError for a file that
            is not .kt or a directory without .kt files
    """
    found = set()
    for raw in inputs:
        p = Path(raw).resolve()
        if p.is_dir():
            files = [f for f in p.rglob('*.kt') if f.is_file() and 'build' not in f.relative_to(p).parts]
            if not files:
                raise ValueError(f"no .kt files below {raw}")
            found.update(f.resolve() for f in files)
        elif p.is_file():
            if p.suffix != '.kt':
                raise ValueError(f"not a .kt file: {raw}")
            found.add(p)
        else:
            raise FileNotFoundError(raw)
    return sorted(found)


def package_names(n: int) -> List[str]:
    """Returns the package of each of n screens: generated.s01 ... (at least two digits).

    Input:  n — number of screens (>= 0)
    Output: list of n package names
    """
    width = max(2, len(str(n)))
    return [f"{PACKAGE_ROOT}.s{i:0{width}d}" for i in range(1, n + 1)]


def with_package(code: str, package: str) -> str:
    """Puts a Kotlin source into the given package.

    An existing package directive is replaced.  Otherwise the directive is
    inserted before the first line, or after the last leading '@file:'
    annotation (which must precede it); comment lines may stay before it.

    Input:  code — Kotlin source; package — fully qualified package name
    Output: Kotlin source with exactly one package directive
    """
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if _PACKAGE_RE.match(line):
            lines[i] = f"package {package}"
            return '\n'.join(lines)
    insert_at = 0
    for i, line in enumerate(lines):
        if _FILE_ANNOTATION_RE.match(line):
            insert_at = i + 1
    return '\n'.join(lines[:insert_at] + [f"package {package}", ""] + lines[insert_at:])


def defines_screen(code: str) -> bool:
    """Tells whether a Kotlin source defines the parameterless top-level fun GeneratedScreen().

    Input:  code — Kotlin source
    Output: True if a line declares 'fun GeneratedScreen()'
    """
    return _SCREEN_FN_RE.search(code) is not None


def snapshot_test_source(package: str) -> str:
    """Returns the Paparazzi snapshot test of one screen.

    Input:  package — package of the screen (the test lives in the same package)
    Output: Kotlin source of the test class TEST_CLASS
    """
    return f"""package {package}

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import app.cash.paparazzi.DeviceConfig
import app.cash.paparazzi.Paparazzi
import org.junit.Rule
import org.junit.Test

// Written by harness/compile_and_render.py: renders {SCREEN_FUNCTION}() of this package.
class {TEST_CLASS} {{
    @get:Rule
    val paparazzi = Paparazzi(deviceConfig = {DEVICE_CONFIG})

    @Test
    fun render() {{
        paparazzi.snapshot {{
            MaterialTheme {{
                Surface(modifier = Modifier.fillMaxSize()) {{
                    {SCREEN_FUNCTION}()
                }}
            }}
        }}
    }}
}}
"""


def render_target(source: Path, render_dir: Path, copy_next_to_source: bool) -> Path:
    """Returns where the render of a source is stored.

    Input:  source — .kt file; render_dir — directory for renders that are not
            placed next to their source; copy_next_to_source — False forces render_dir
    Output: <source dir>/render.png when allowed and the source is the only .kt
            file in its directory, else <render_dir>/<parent dir>__<stem>.png
    """
    if copy_next_to_source:
        siblings = [f for f in source.parent.glob('*.kt') if f.is_file()]
        if len(siblings) == 1:
            return source.parent / RENDER_NAME
    return render_dir / f"{source.parent.name}__{source.stem}.png"


def report_path(path: Optional[Path]) -> Optional[str]:
    """Formats a path for the report: relative to the repository root when inside it.

    Input:  path — absolute path or None
    Output: POSIX path string (repo-relative or absolute), or None
    """
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


# ---------------------------------------------------------------------------
# Parsing tool output (pure)
# ---------------------------------------------------------------------------

def parse_kotlin_diagnostics(log: str) -> Dict[str, Dict[str, List[str]]]:
    """Extracts the Kotlin compiler errors and warnings per source file from a Gradle log.

    Lines have the form 'e: file:///abs/path/File.kt:12:5 message' (errors) or
    'w: ...' (warnings); file URIs are percent-decoded.

    Input:  log — Gradle console output
    Output: {absolute file path: {'errors': ['12:5 message', ...],
            'warnings': [...]}}
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for line in log.splitlines():
        m = _DIAGNOSTIC_RE.match(line.rstrip())
        if not m:
            continue
        path = m.group('path')
        if path.startswith('file:'):
            path = urllib.parse.unquote(urllib.parse.urlparse(path).path)
        entry = out.setdefault(os.path.normpath(path), {'errors': [], 'warnings': []})
        kind = 'errors' if m.group('kind') == 'e' else 'warnings'
        entry[kind].append(f"{m.group('line')}:{m.group('col')} {m.group('msg')}")
    return out


def parse_resolved_versions(log: str) -> Dict[str, str]:
    """Extracts the 'RESOLVED group:name=version' lines printed by :screens:printResolvedVersions.

    Input:  log — Gradle console output
    Output: {'group:name': 'version'}
    """
    out: Dict[str, str] = {}
    for line in log.splitlines():
        m = _RESOLVED_RE.match(line.strip())
        if m:
            out[m.group('module')] = m.group('version')
    return out


def parse_junit_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Reads the JUnit XML results of the snapshot tests.

    Input:  results_dir — directory with TEST-*.xml files
    Output: {test class name: {'passed': bool, 'message': str or None}}; a
            class passes when none of its test cases failed, errored or was skipped
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not results_dir.is_dir():
        return out
    for xml_file in sorted(results_dir.glob('TEST-*.xml')):
        root = ET.parse(xml_file).getroot()
        suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
        for suite in suites:
            for case in suite.iter('testcase'):
                cls = case.get('classname', '')
                entry = out.setdefault(cls, {'passed': True, 'message': None})
                for tag in ('failure', 'error', 'skipped'):
                    bad = case.find(tag)
                    if bad is None:
                        continue
                    entry['passed'] = False
                    msg = bad.get('message')
                    if not msg:
                        lines = (bad.text or '').strip().splitlines()
                        msg = lines[0] if lines else '(no message)'
                    entry['message'] = f"{tag}: {msg[:MAX_MESSAGE_CHARS]}"
    return out


def png_size(path: Path) -> Optional[List[int]]:
    """Reads the pixel size of a PNG file from its IHDR chunk.

    Input:  path — PNG file
    Output: [width, height], or None if the file is not a PNG
    """
    with open(path, 'rb') as f:
        head = f.read(24)
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        return None
    w, h = struct.unpack('>II', head[16:24])
    return [w, h]


def read_toolchain(android_dir: Path = ANDROID_DIR) -> Dict[str, Any]:
    """Reads the declared toolchain versions from the Gradle files of the harness.

    Input:  android_dir — harness Gradle project
    Output: {'gradle', 'agp', 'kotlin', 'compose_bom', 'paparazzi',
            'compile_sdk', 'min_sdk'} (None for a value that is not found)
    """
    catalog = (android_dir / 'gradle' / 'libs.versions.toml').read_text(encoding='utf-8')
    versions: Dict[str, str] = {}
    in_versions = False
    for line in catalog.splitlines():
        s = line.strip()
        if s.startswith('['):
            in_versions = s == '[versions]'
            continue
        m = re.match(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]+)"', s)
        if in_versions and m:
            versions[m.group(1)] = m.group(2)
    wrapper = (android_dir / 'gradle' / 'wrapper' / 'gradle-wrapper.properties').read_text(encoding='utf-8')
    gm = re.search(r'gradle-([0-9][^-/]*)-(?:bin|all)\.zip', wrapper)
    module = (android_dir / MODULE / 'build.gradle.kts').read_text(encoding='utf-8')
    csdk = re.search(r'compileSdk\s*=\s*(\d+)', module)
    msdk = re.search(r'minSdk\s*=\s*(\d+)', module)
    return {
        'gradle': gm.group(1) if gm else None,
        'agp': versions.get('agp'),
        'kotlin': versions.get('kotlin'),
        'compose_bom': versions.get('composeBom'),
        'paparazzi': versions.get('paparazzi'),
        'compile_sdk': int(csdk.group(1)) if csdk else None,
        'min_sdk': int(msdk.group(1)) if msdk else None,
    }


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

class Screen:
    """One input .kt file and its compile / render outcome."""

    def __init__(self, source: Path, package: str):
        """Reads the source and prepares its package copy.

        Input:  source — absolute path of the .kt file; package — assigned package
        Output: none
        """
        self.source = source
        self.package = package
        raw = source.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.code = with_package(raw.decode('utf-8'), package)
        self.has_screen = defines_screen(self.code)
        self.compiled: Optional[bool] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.compile_note: Optional[str] = None
        self.rendered: Optional[bool] = None
        self.render_error: Optional[str] = None
        self.render: Optional[Path] = None
        self.render_size: Optional[List[int]] = None

    @property
    def short(self) -> str:
        """Last package segment, e.g. 's01'.

        Input:  none
        Output: string
        """
        return self.package.rsplit('.', 1)[-1]

    def copy_path(self, android_dir: Path) -> Path:
        """Returns where the package copy of the source is written.

        Input:  android_dir — harness Gradle project
        Output: path below screens/src/main/java/generated/<short>/
        """
        return android_dir / _MAIN_SRC / self.short / self.source.name

    def test_path(self, android_dir: Path) -> Path:
        """Returns where the snapshot test of the screen is written.

        Input:  android_dir — harness Gradle project
        Output: path below screens/src/test/java/generated/<short>/
        """
        return android_dir / _TEST_SRC / self.short / f"{TEST_CLASS}.kt"

    def to_report(self) -> Dict[str, Any]:
        """Returns the report entry of the screen.

        Input:  none
        Output: JSON-serialisable dictionary
        """
        return {
            'source': report_path(self.source),
            'sha256': self.sha256,
            'package': self.package,
            'compiled': bool(self.compiled),
            'compile_errors': self.errors[:MAX_MESSAGES],
            'compile_warnings': self.warnings[:MAX_MESSAGES],
            'compile_note': self.compile_note,
            'has_screen_function': self.has_screen,
            'rendered': bool(self.rendered),
            'render_error': self.render_error,
            'render': report_path(self.render),
            'render_size_px': self.render_size,
        }


def write_sources(screens: Sequence[Screen], android_dir: Path, with_tests: bool):
    """Deletes and rewrites the generated main (and test) source trees of the module.

    Input:  screens — screens to write; android_dir — harness Gradle project;
            with_tests — also write the snapshot tests of screens with a GeneratedScreen()
    Output: none (side effect on the module's src/main and src/test trees)
    """
    for tree in (android_dir / _MAIN_SRC, android_dir / _TEST_SRC):
        if tree.exists():
            shutil.rmtree(tree)
    for s in screens:
        dst = s.copy_path(android_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(s.code, encoding='utf-8')
        if with_tests and s.has_screen:
            t = s.test_path(android_dir)
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_text(snapshot_test_source(s.package), encoding='utf-8')


def assign_diagnostics(screens: Sequence[Screen], diags: Dict[str, Dict[str, List[str]]],
                       android_dir: Path) -> List[Screen]:
    """Sets the compiler errors and warnings of each screen from the diagnostics of one round.

    A screen that the diagnostics do not name gets empty lists.

    Input:  screens — screens of the current round; diags — output of
            parse_kotlin_diagnostics; android_dir — harness Gradle project
    Output: the screens with at least one error
    """
    failing = []
    for s in screens:
        d = diags.get(os.path.normpath(str(s.copy_path(android_dir)))) or {'errors': [], 'warnings': []}
        s.errors, s.warnings = list(d['errors']), list(d['warnings'])
        if s.errors:
            failing.append(s)
    return failing


# ---------------------------------------------------------------------------
# Gradle
# ---------------------------------------------------------------------------

def run_gradle(tasks: Sequence[str], android_dir: Path, env: Dict[str, str], log_path: Path,
               offline: bool = False) -> Tuple[int, str]:
    """Runs the Gradle wrapper of the harness and keeps its console output.

    Input:  tasks — Gradle tasks and options; android_dir — harness Gradle
            project; env — process environment (JAVA_HOME set); log_path — file
            receiving the output; offline — pass --offline
    Output: (exit code, combined stdout/stderr)
    """
    cmd = [str(android_dir / 'gradlew'), '--console=plain']
    if offline:
        cmd.append('--offline')
    cmd += list(tasks)
    proc = subprocess.run(cmd, cwd=str(android_dir), env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"$ {' '.join(cmd)}\n{proc.stdout}", encoding='utf-8')
    return proc.returncode, proc.stdout


def _tail(text: str, n: int = LOG_TAIL_LINES) -> str:
    """Returns the last n non-empty lines of a log.

    Input:  text — log text; n — number of lines
    Output: joined lines
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return '\n'.join(lines[-n:])


def java_version(java_home: Path) -> Optional[str]:
    """Returns the version string of the JDK at java_home (e.g. '17.0.13').

    Input:  java_home — JDK home directory
    Output: version string, or None if it cannot be determined
    """
    try:
        out = subprocess.run([str(java_home / 'bin' / 'java'), '-version'], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True).stdout
    except OSError:
        return None
    m = re.search(r'version "([^"]+)"', out)
    return m.group(1) if m else None


def compile_screens(screens: List[Screen], android_dir: Path, env: Dict[str, str], log_dir: Path,
                    offline: bool) -> Dict[str, str]:
    """Compiles the screens, dropping the files with errors between rounds.

    Input:  screens — all screens; android_dir — harness Gradle project; env —
            process environment; log_dir — directory for the Gradle logs;
            offline — pass --offline to Gradle
    Output: resolved Compose versions printed in the first round (may be empty);
            side effect: sets compiled / errors / warnings / compile_note of each screen
    """
    remaining = list(screens)
    resolved: Dict[str, str] = {}
    for rnd in range(1, MAX_COMPILE_ROUNDS + 1):
        write_sources(remaining, android_dir, with_tests=False)
        tasks = [f':{MODULE}:compileDebugKotlin', '--rerun']
        if rnd == 1:
            tasks = [f':{MODULE}:printResolvedVersions'] + tasks
        code, log = run_gradle(tasks, android_dir, env, log_dir / f'compile_round{rnd}.log', offline)
        if rnd == 1:
            resolved = parse_resolved_versions(log)
        failing = assign_diagnostics(remaining, parse_kotlin_diagnostics(log), android_dir)
        if code == 0:
            for s in remaining:
                s.compiled = True
            return resolved
        if not failing:
            for s in remaining:
                s.compiled = False
                s.compile_note = f"build failed without file-level errors (round {rnd}):\n{_tail(log)}"
            return resolved
        for s in failing:
            s.compiled = False
        remaining = [s for s in remaining if s not in failing]
        if not remaining:
            return resolved
    for s in remaining:
        s.compiled = False
        s.compile_note = f"not compiled: still failing after {MAX_COMPILE_ROUNDS} rounds"
    return resolved


def render_screens(screens: List[Screen], android_dir: Path, env: Dict[str, str], log_dir: Path,
                   offline: bool, render_dir: Path, copy_next_to_source: bool):
    """Records the Paparazzi snapshots of the compiled screens and places the PNGs.

    Input:  screens — all screens (only compiled ones with a GeneratedScreen()
            are rendered); android_dir, env, log_dir, offline — as for
            compile_screens; render_dir, copy_next_to_source — see render_target
    Output: none (side effect: sets rendered / render_error / render / render_size,
            copies PNGs and deletes stale renders)
    """
    todo = [s for s in screens if s.compiled and s.has_screen]
    for s in screens:
        if not s.has_screen:
            s.rendered = False
            s.render_error = f"no 'fun {SCREEN_FUNCTION}()' in the file: not rendered"
        elif not s.compiled:
            s.rendered = False
            s.render_error = 'not compiled'
    images = android_dir / _SNAPSHOT_IMAGES
    if todo:
        for d in (images, android_dir / _TEST_RESULTS):
            if d.exists():
                shutil.rmtree(d)
        write_sources([s for s in screens if s.compiled], android_dir, with_tests=True)
        code, log = run_gradle([f':{MODULE}:recordPaparazziDebug'], android_dir, env,
                               log_dir / 'render.log', offline)
        results = parse_junit_results(android_dir / _TEST_RESULTS)
        for s in todo:
            res = results.get(f"{s.package}.{TEST_CLASS}")
            pngs = sorted(images.glob(f"{s.package}_*.png")) if images.is_dir() else []
            if res is None:
                s.rendered = False
                s.render_error = f"no test result (Gradle exit code {code}):\n{_tail(log)}"
            elif not res['passed']:
                s.rendered = False
                s.render_error = res['message']
            elif not pngs:
                s.rendered = False
                s.render_error = 'test passed but no snapshot PNG was written'
            else:
                target = render_target(s.source, render_dir, copy_next_to_source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(pngs[0], target)
                s.rendered, s.render, s.render_size = True, target, png_size(target)
    for s in screens:
        if not s.rendered:
            stale = render_target(s.source, render_dir, copy_next_to_source)
            if stale.is_file():
                stale.unlink()


def build_report(screens: Sequence[Screen], toolchain: Dict[str, Any], render_dir: Path) -> Dict[str, Any]:
    """Assembles the JSON report.

    Input:  screens — processed screens; toolchain — versions; render_dir —
            directory of renders not placed next to their source
    Output: report dictionary
    """
    entries = [s.to_report() for s in screens]
    with_fn = [s for s in screens if s.has_screen]
    return {
        'generated_by': 'harness/compile_and_render.sh',
        'toolchain': toolchain,
        'snapshot': {
            'device_config': DEVICE_CONFIG,
            'wrapper': 'MaterialTheme { Surface(Modifier.fillMaxSize()) { GeneratedScreen() } }',
            'render_dir': report_path(render_dir),
        },
        'summary': {
            'files': len(entries),
            'compiled': sum(1 for e in entries if e['compiled']),
            'with_screen_function': len(with_fn),
            'rendered': sum(1 for e in entries if e['rendered']),
        },
        'files': entries,
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _environment(java_home: Path, sdk_dir: Path) -> Dict[str, str]:
    """Builds the environment of the Gradle runs.

    Input:  java_home — JDK home; sdk_dir — Android SDK
    Output: environment dictionary
    """
    env = dict(os.environ)
    env['JAVA_HOME'] = str(java_home)
    env['ANDROID_HOME'] = str(sdk_dir)
    env.pop('ANDROID_SDK_ROOT', None)
    return env


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point (see the module docstring).

    Input:  argv — command-line arguments (default: sys.argv[1:])
    Output: exit status (0 all compiled and, unless --compile-only, rendered;
            1 some failed; 2 usage/setup error)
    """
    parser = argparse.ArgumentParser(
        description="Compile generated Compose screens and render them with Paparazzi.")
    parser.add_argument('inputs', nargs='+', help=".kt files or directories searched for *.kt")
    parser.add_argument('--report', default=str(DEFAULT_REPORT), help="JSON report path")
    parser.add_argument('--render-dir', default=str(DEFAULT_RENDER_DIR),
                        help="directory for renders that are not placed next to their source")
    parser.add_argument('--no-copy', action='store_true',
                        help="never write render.png next to a source; use --render-dir for all")
    parser.add_argument('--compile-only', action='store_true', help="skip the Paparazzi rendering")
    parser.add_argument('--offline', action='store_true', help="run Gradle with --offline")
    parser.add_argument('--java-home', help="JDK 17 home (default: $JAVA_HOME, else Homebrew openjdk@17)")
    parser.add_argument('--sdk-dir', help="Android SDK (default: $ANDROID_HOME / $ANDROID_SDK_ROOT, "
                                          "else ~/Library/Android/sdk)")
    args = parser.parse_args(argv)

    try:
        sources = collect_sources(args.inputs)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    java_home = Path(args.java_home or os.environ.get('JAVA_HOME') or DEFAULT_JAVA_HOME)
    sdk_dir = Path(args.sdk_dir or os.environ.get('ANDROID_HOME') or os.environ.get('ANDROID_SDK_ROOT')
                   or DEFAULT_SDK_DIR)
    toolchain = read_toolchain(ANDROID_DIR)
    platform = sdk_dir / 'platforms' / f"android-{toolchain['compile_sdk']}"
    if not (java_home / 'bin' / 'java').exists():
        print(f"error: no JDK at {java_home} (set JAVA_HOME or --java-home)", file=sys.stderr)
        return 2
    if not platform.is_dir():
        print(f"error: Android platform missing: {platform}", file=sys.stderr)
        return 2
    (ANDROID_DIR / 'local.properties').write_text(f"sdk.dir={sdk_dir}\n", encoding='utf-8')

    screens = [Screen(src, pkg) for src, pkg in zip(sources, package_names(len(sources)))]
    env = _environment(java_home, sdk_dir)
    log_dir = ANDROID_DIR / 'build' / 'harness-logs'
    render_dir = Path(args.render_dir).resolve()

    resolved = compile_screens(screens, ANDROID_DIR, env, log_dir, args.offline)
    if args.compile_only:
        for s in screens:
            s.rendered, s.render_error = False, 'rendering skipped (--compile-only)'
    else:
        render_screens(screens, ANDROID_DIR, env, log_dir, args.offline, render_dir, not args.no_copy)

    toolchain.update({
        'jdk': java_version(java_home),
        'resolved': resolved,
    })
    report = build_report(screens, toolchain, render_dir)
    report_file = Path(args.report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding='utf-8')

    for s in screens:
        state = 'compiled' if s.compiled else 'COMPILE FAILED'
        if s.has_screen:
            state += ', rendered' if s.rendered else ', NOT RENDERED'
        print(f"{s.short}  {report_path(s.source)}: {state}")
    summ = report['summary']
    print(f"{summ['compiled']}/{summ['files']} compiled, {summ['rendered']}/{summ['with_screen_function']} "
          f"rendered; report: {report_path(report_file)}")
    ok = summ['compiled'] == summ['files'] and (
        args.compile_only or summ['rendered'] == summ['with_screen_function'])
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
