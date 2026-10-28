#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Asks Claude, through the Claude Code CLI, for a Compose screen per example screenshot.

Qualitative MLLM baseline for the bundled examples: the same screenshots that
ui2compose turns into results/v1.2.0_examples/<stem>/4_GeneratedScreen.kt are
sent to a hosted multimodal model with one fixed prompt, and the Kotlin it
returns is stored for compilation and rendering with the same harness
(harness/compile_and_render.sh).  results/mllm_claude/README.md describes the
protocol and the published run.

Usage (from the repository root; RAW is a directory outside the repository):
    python scripts/mllm_compare_claude.py --probe --out results/mllm_claude/isolation_probe --raw-dir RAW
    python scripts/mllm_compare_claude.py --raw-dir RAW [--runs 2] [--only <stem>] [--run N]
                                          [--resume | --force] [--budget N] [--dry-run]
    harness/compile_and_render.sh --offline --report results/mllm_claude/compile_report.json \\
        --render-dir results/mllm_claude/renders results/mllm_claude
    python scripts/mllm_compare_claude.py --summarize
    python scripts/mllm_compare_claude.py --resanitize <call directory> --raw-dir RAW

What it does:
  1. Generation (default mode).  For every examples/*.png (or the --only
     stems) and every run 1..--runs (or the --run numbers), the CLI is called
     once, in print mode, with the flags of cli_argv(): no tools, no MCP
     servers, no setting sources (user/project/local settings, plugins and
     hooks are not loaded), no skills or slash commands, no session
     persistence, the fixed system prompt (PROMPT_SYSTEM) and the requested
     model.  The CLI runs in an empty working directory with a minimal
     environment (cli_environment()).  One stream-json user message is sent
     on stdin: [image block (the PNG, base64, unmodified), text block
     (PROMPT_USER)].  Per call two directories are written:
       RAW/<stem>/run<k>/     verbatim CLI output, never published
         stream.jsonl            the CLI's stdout (init event, assistant
                                 message(s), result event)
         stderr.txt              the CLI's stderr
         meta.json               the metadata below, unsanitized
       <out>/<stem>/run<k>/   the published copy
         stream.sanitized.jsonl  stdout passed through Redactor: session ids,
                                 event uuids, request and message ids, the
                                 working directory, memory and socket paths,
                                 thinking signatures and rate-limit details
                                 are replaced by placeholders, and every
                                 e-mail address, home-directory path and the
                                 user name are redacted
         stderr.sanitized.txt    stderr, redacted the same way
         response.txt            the final text of the model, verbatim
         GeneratedScreen.kt      the Kotlin extracted from the response (see
                                 extract_kotlin(); never edited otherwise)
         meta.json               model id, CLI version, timestamps, durations,
                                 tokens, exit code, error, isolation evidence,
                                 the extraction result, the SHA-256 of the raw
                                 and sanitized files and a privacy scan of the
                                 response (redacted like the stream)
     response.txt and GeneratedScreen.kt are not redacted; the privacy scan in
     meta.json counts the e-mail-like strings, home paths, user-name and
     session-id occurrences in the response so that every hit can be checked
     (an address shown in the screenshot is legitimate output) before publishing.
  2. Isolation probe (--probe).  One call with the same flags and system
     prompt and a text-only message (PROMPT_PROBE) that asks the model to
     quote everything it received before the message.  Writes the same files
     (without GeneratedScreen.kt) to --out and RAW/<--out name>/.
  3. Summary (--summarize).  Reads the meta.json and GeneratedScreen.kt of
     every <out>/<stem>/run<k>/, the harness report <out>/compile_report.json
     and, for reference, results/compile_report.json (ui2compose v1.2.0), and
     writes <out>/summary.json: per screen the compile and render result of
     every run, the run1/run2 determinism (byte identity, whitespace-
     insensitive identity, difflib similarity ratios), durations and tokens,
     and the totals.  No model is called.
  4. Re-sanitizing (--resanitize DIR).  For a call directory written by an
     earlier version of this script (verbatim stream.jsonl and stderr.txt in
     DIR): moves both files and a copy of meta.json to RAW/<DIR name>/, writes
     the sanitized copies to DIR and adds the 'raw' and 'sanitized' sections
     to DIR/meta.json.  No model is called.

Budget protection: existing outputs (in <out> or RAW) are never overwritten
without --force (--resume skips the targets that already have outputs); every
call is logged to --call-log (one 'start' and one 'end' JSON line per call)
before and after it runs; --budget N refuses to start when the calls already
logged plus the planned ones would exceed N; the batch stops at the first
failed call.  There are no automatic retries.
"""

import argparse
import base64
import collections
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = Path('examples')
OUT_DIR = Path('results/mllm_claude')
UI2COMPOSE_REPORT = Path('results/compile_report.json')
UI2COMPOSE_EXAMPLES = Path('results/v1.2.0_examples')

#: Prompt files (read from --prompts-dir, sent verbatim, no newline added or removed).
PROMPT_SYSTEM = 'prompt_system.txt'
PROMPT_USER = 'prompt_user.txt'
PROMPT_PROBE = 'prompt_probe.txt'

#: Model requested from the CLI (--model).
DEFAULT_MODEL = 'claude-sonnet-5-5'

#: Seconds before a CLI call is killed.
DEFAULT_TIMEOUT_S = 900

#: Verbatim files written per call to the raw directory.
STREAM_FILE = 'stream.jsonl'
STDERR_FILE = 'stderr.txt'
#: Files written per call to the published directory.
STREAM_SANITIZED_FILE = 'stream.sanitized.jsonl'
STDERR_SANITIZED_FILE = 'stderr.sanitized.txt'
RESPONSE_FILE = 'response.txt'
KOTLIN_FILE = 'GeneratedScreen.kt'
META_FILE = 'meta.json'
#: Written next to KOTLIN_FILE by harness/compile_and_render.sh.
RENDER_FILE = 'render.png'
#: Every file this script (or the harness) writes into a call directory.
CALL_FILES = (STREAM_FILE, STDERR_FILE, STREAM_SANITIZED_FILE, STDERR_SANITIZED_FILE, RESPONSE_FILE,
              KOTLIN_FILE, META_FILE, RENDER_FILE)

#: Init-event fields that must be empty lists for a call to count as isolated.
ISOLATION_EMPTY_FIELDS = ('tools', 'mcp_servers', 'skills', 'slash_commands')

#: Environment variables passed through to the CLI (everything else is dropped,
#: in particular CLAUDE*, ANTHROPIC* and MCP* variables of a calling session).
ENV_PASSTHROUGH = ('HOME', 'USER', 'LOGNAME', 'SHELL', 'TMPDIR', 'LANG', 'LC_ALL')
#: Environment variables set for the CLI: no self-update; no attribution
#: header in the system prompt; no CLAUDE.md memory files.
ENV_FIXED = {
    'DISABLE_AUTOUPDATER': '1',
    'CLAUDE_CODE_ATTRIBUTION_HEADER': '0',
    'CLAUDE_CODE_DISABLE_CLAUDE_MDS': '1',
}

_FENCE_RE = re.compile(r'^```[ \t]*([^\n`]*?)[ \t]*\n(.*?)\n```[ \t]*$', re.DOTALL | re.MULTILINE)
_OPEN_KOTLIN_RE = re.compile(r'^```[ \t]*(?:kotlin|kt)[ \t]*\n', re.IGNORECASE | re.MULTILINE)
_SCREEN_FN_RE = re.compile(r'(?m)^\s*fun\s+GeneratedScreen\s*\(\s*\)')
_PACKAGE_RE = re.compile(r'(?m)^\s*package\s+[A-Za-z_][\w.]*\s*;?\s*$')
_TEXT_CALL_RE = re.compile(r'\bText\s*\(')

#: JSON keys whose (non-empty) values are replaced in the sanitized copies.
REDACT_KEYS = {
    'session_id': '<session-id>',
    'uuid': '<uuid>',
    'request_id': '<request-id>',
    'cwd': '<cwd>',
    'memory_paths': '<redacted>',
    'messaging_socket_path': '<redacted>',
    'signature': '<redacted>',
    'rate_limit_info': '<redacted>',
}
_ID_VALUE_RE = re.compile(r'^(?:msg|req|toolu)_[A-Za-z0-9]+$')
_UUID_RE = re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b')
_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
_HOME_PATH_RE = re.compile(r'/(Users|home)/[^/\s"\'<>]+')


# ── Small helpers ───────────────────────────────────────────


def sha256_bytes(data: bytes) -> str:
    """Returns the hex SHA-256 of a byte string.

    Input:  data — bytes
    Output: 64-character hex digest
    """
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    """Returns the current UTC time as an ISO 8601 string with milliseconds.

    Input:  none
    Output: e.g. '2026-10-20T07:12:03.123Z'
    """
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def rel(path: Path) -> str:
    """Formats a path relative to the repository root when it lies inside it.

    Input:  path — any path
    Output: POSIX path string (repo-relative or absolute)
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def inside_repo(path: Path) -> bool:
    """Tells whether a path is the repository root or lies below it.

    Input:  path — any path
    Output: bool
    """
    p = path.resolve()
    return p == REPO_ROOT or REPO_ROOT in p.parents


def png_size(data: bytes) -> Optional[List[int]]:
    """Reads the pixel size of a PNG from its IHDR chunk.

    Input:  data — PNG file contents
    Output: [width, height], or None if the data is not a PNG
    """
    if len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
        return None
    w, h = struct.unpack('>II', data[16:24])
    return [w, h]


def read_prompt(prompts_dir: Path, name: str) -> Tuple[str, str]:
    """Reads a prompt file verbatim.

    Input:  prompts_dir — directory of the prompt files; name — file name
    Output: (text, SHA-256 of the file bytes)
    """
    raw = (prompts_dir / name).read_bytes()
    return raw.decode('utf-8'), sha256_bytes(raw)


def write_json(path: Path, data: Any):
    """Writes indented JSON with a trailing newline.

    Input:  path — target file; data — JSON-serialisable value
    Output: none
    """
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def non_empty(path: Path) -> bool:
    """Tells whether a directory exists and contains anything.

    Input:  path — directory
    Output: bool
    """
    return path.is_dir() and any(path.iterdir())


# ── Sanitizing (pure apart from Path.home()) ────────────────


def _path_forms(path: Path) -> List[str]:
    """Returns the spellings under which a path can appear in the CLI's output.

    The path as given and resolved, with and without the macOS '/private'
    prefix, and the CLI's project-directory encoding (every character that
    is not a letter or digit replaced by '-').

    Input:  path — directory
    Output: list of strings (no empty ones)
    """
    forms = set()
    for s in {str(path), str(path.resolve())}:
        forms.add(s)
        if s.startswith('/private/'):
            forms.add(s[len('/private'):])
    forms |= {re.sub(r'[^A-Za-z0-9]', '-', s) for s in list(forms)}
    return sorted(f for f in forms if len(f) > 1)


class Redactor:
    """Removes machine- and account-specific values from the CLI's output before it is published.

    Literal replacements (longest first): the given directories (working
    directory, raw directory, repository) and the home directory in all
    spellings of _path_forms(), and the session ids.  Then every UUID, every
    e-mail address, every remaining /Users/<name> or /home/<name> path and
    the user name (the last component of the home directory) are replaced.
    For JSON, the values of REDACT_KEYS and message/request ids are replaced
    as a whole.  The number of replacements per kind is kept in 'counts'.
    """

    def __init__(self, paths: Sequence[Tuple[Path, str]] = (), session_ids: Sequence[str] = (),
                 home: Optional[Path] = None):
        """Prepares the replacements.

        Input:  paths — [(directory, placeholder)]; session_ids — ids to
                replace by '<session-id>'; home — home directory (default Path.home())
        Output: none
        """
        home = Path.home() if home is None else home
        self.user = home.name if len(home.name) >= 3 else None
        lits: Dict[str, Tuple[str, str]] = {}
        for form in _path_forms(home):
            lits[form] = ('~', 'home_path')
        for path, placeholder in paths:
            for form in _path_forms(path):
                lits[form] = (placeholder, 'path')
        for sid in session_ids:
            if sid:
                lits[sid] = ('<session-id>', 'session_id')
        self.literals = sorted(lits.items(), key=lambda kv: -len(kv[0]))
        self.session_ids = [s for s in session_ids if s]
        self.home_forms = _path_forms(home)
        self.counts: collections.Counter = collections.Counter()

    def text(self, s: str) -> str:
        """Redacts a string.

        Input:  s — text
        Output: redacted text
        """
        for lit, (placeholder, kind) in self.literals:
            n = s.count(lit)
            if n:
                self.counts[kind] += n
                s = s.replace(lit, placeholder)
        for regex, repl, kind in ((_UUID_RE, '<uuid>', 'uuid'), (_EMAIL_RE, '<email>', 'email'),
                                  (_HOME_PATH_RE, r'/\1/<user>', 'home_path')):
            s, n = regex.subn(repl, s)
            self.counts[kind] += n
        if self.user:
            s, n = re.subn(r'\b' + re.escape(self.user) + r'\b', '<user>', s)
            self.counts['user_name'] += n
        return s

    def json(self, value: Any) -> Any:
        """Redacts a JSON value recursively.

        Input:  value — parsed JSON
        Output: redacted copy
        """
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                # Only text and structures are redacted as a whole; numbers
                # (e.g. the counts of this script's own scans) are kept.
                if k in REDACT_KEYS and isinstance(v, (str, list, dict)) and v:
                    out[k] = REDACT_KEYS[k]
                    self.counts[f'key:{k}'] += 1
                elif k == 'id' and isinstance(v, str) and _ID_VALUE_RE.match(v):
                    out[k] = '<message-id>'
                    self.counts['key:id'] += 1
                else:
                    out[k] = self.json(v)
            return out
        if isinstance(value, list):
            return [self.json(v) for v in value]
        if isinstance(value, str):
            return self.text(value)
        return value

    def jsonl(self, data: bytes) -> bytes:
        """Redacts JSON-lines output line by line (lines that are not JSON are redacted as text).

        Input:  data — raw bytes
        Output: redacted bytes (compact JSON per line, trailing newline kept)
        """
        lines = []
        for line in data.decode('utf-8', errors='replace').splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                lines.append(self.text(line))
                continue
            lines.append(json.dumps(self.json(ev), ensure_ascii=False))
        out = '\n'.join(lines)
        if data.endswith(b'\n'):
            out += '\n'
        return out.encode('utf-8')

    def scan(self, s: str) -> Dict[str, int]:
        """Counts what a text still contains that the redaction would remove (nothing is changed).

        Input:  s — text
        Output: {'email', 'home_path', 'user_name', 'session_id', 'uuid'}: counts
        """
        return {
            'email': len(_EMAIL_RE.findall(s)),
            'home_path': len(_HOME_PATH_RE.findall(s))
                         + sum(s.count(f) for f in self.home_forms if not f.startswith('/')),
            'user_name': len(re.findall(r'\b' + re.escape(self.user) + r'\b', s)) if self.user else 0,
            'session_id': sum(s.count(sid) for sid in self.session_ids),
            'uuid': len(_UUID_RE.findall(s)),
        }


def session_ids_of(stdout: bytes) -> List[str]:
    """Collects every 'session_id' value of a stream-json output.

    Input:  stdout — raw CLI stdout
    Output: sorted distinct ids
    """
    ids = set()
    for line in stdout.decode('utf-8', errors='replace').splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and isinstance(ev.get('session_id'), str):
            ids.add(ev['session_id'])
    return sorted(ids)


def sanitize_outputs(stdout: bytes, stderr: bytes, extra_paths: Sequence[Tuple[Path, str]]
                     ) -> Tuple[bytes, bytes, Redactor]:
    """Builds the published copies of a call's stdout and stderr.

    Input:  stdout, stderr — raw CLI output; extra_paths — [(directory,
            placeholder)] to replace besides the home directory and the
            repository (e.g. the CLI working directory and the raw directory)
    Output: (sanitized stdout, sanitized stderr, the Redactor with its counts)
    """
    events = parse_stream(stdout)
    paths = list(extra_paths) + [(REPO_ROOT, '<repo>')]
    init_cwd = (events.get('init') or {}).get('cwd')
    if isinstance(init_cwd, str) and init_cwd:
        paths.append((Path(init_cwd), '<cwd>'))
    red = Redactor(paths, session_ids_of(stdout))
    return red.jsonl(stdout), red.text(stderr.decode('utf-8', errors='replace')).encode('utf-8'), red


def residual_findings(red: Redactor, texts: Sequence[str]) -> Dict[str, int]:
    """Scans sanitized texts for anything the redaction should have removed.

    Input:  red — the Redactor used; texts — sanitized texts
    Output: {kind: count} of the kinds with a non-zero count ({} when clean)
    """
    total: collections.Counter = collections.Counter()
    for t in texts:
        total.update(red.scan(t))
    return {k: v for k, v in sorted(total.items()) if v}


# ── CLI invocation ──────────────────────────────────────────


def cli_argv(claude: str, model: str, system_prompt: str) -> List[str]:
    """Returns the CLI command line of one call.

    Flags: print mode; stream-json in and out (with --verbose, required for
    stream-json output); no built-in tools (--tools ""); no setting sources
    (--setting-sources "": user, project and local settings, their plugins
    and hooks are not loaded); only MCP servers from --mcp-config, of which
    there are none (--strict-mcp-config); no skills or slash commands; no
    session file; the fixed system prompt replaces the default one.

    Input:  claude — CLI executable; model — model name; system_prompt — text
    Output: argument list
    """
    return [claude, '-p',
            '--model', model,
            '--input-format', 'stream-json',
            '--output-format', 'stream-json', '--verbose',
            '--tools', '',
            '--setting-sources', '',
            '--strict-mcp-config',
            '--disable-slash-commands',
            '--no-session-persistence',
            '--system-prompt', system_prompt]


def cli_environment(claude: str, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Builds the minimal environment of a CLI call.

    Only ENV_PASSTHROUGH is copied; PATH is the CLI's directory plus the
    system directories; ENV_FIXED is added.

    Input:  claude — CLI executable; base — environment to copy from (default os.environ)
    Output: environment dictionary
    """
    base = dict(os.environ if base is None else base)
    env = {k: base[k] for k in ENV_PASSTHROUGH if k in base}
    env['PATH'] = os.pathsep.join([str(Path(claude).resolve().parent), '/usr/bin', '/bin', '/usr/sbin', '/sbin'])
    env.update(ENV_FIXED)
    return env


def cli_version(claude: str, env: Dict[str, str]) -> Optional[str]:
    """Returns the output of `claude --version` (no model call).

    Input:  claude — CLI executable; env — environment
    Output: version string, e.g. '2.1.267 (Claude Code)', or None on failure
    """
    try:
        out = subprocess.run([claude, '--version'], env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def user_message(text: str, png: Optional[bytes]) -> bytes:
    """Builds the single stream-json user message sent on stdin.

    Input:  text — prompt text; png — PNG bytes for an image block before the
            text, or None for a text-only message
    Output: one JSON line (UTF-8, newline-terminated)
    """
    content: List[Dict[str, Any]] = []
    if png is not None:
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                    'data': base64.b64encode(png).decode('ascii')}})
    content.append({'type': 'text', 'text': text})
    msg = {'type': 'user', 'message': {'role': 'user', 'content': content}, 'parent_tool_use_id': None}
    return (json.dumps(msg) + '\n').encode('utf-8')


def dir_entries(path: Path) -> List[str]:
    """Lists the names in a directory (hidden ones included).

    Input:  path — directory
    Output: sorted names
    """
    return sorted(os.listdir(path))


def run_cli(argv: Sequence[str], env: Dict[str, str], cwd: Path, stdin: bytes,
            timeout_s: int) -> Dict[str, Any]:
    """Runs one CLI call.

    Input:  argv — command line; env — environment; cwd — working directory;
            stdin — bytes written to stdin (then closed); timeout_s — seconds
            before the process is killed
    Output: {'exit_code', 'timed_out', 'stdout' (bytes), 'stderr' (bytes),
            'started_utc', 'finished_utc', 'wall_ms'}
    """
    started = utc_now()
    t0 = dt.datetime.now(dt.timezone.utc)
    proc = subprocess.Popen(list(argv), cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timed_out = False
    try:
        out, err = proc.communicate(input=stdin, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        out, err = proc.communicate()
    wall_ms = int((dt.datetime.now(dt.timezone.utc) - t0).total_seconds() * 1000)
    return {'exit_code': proc.returncode, 'timed_out': timed_out, 'stdout': out, 'stderr': err,
            'started_utc': started, 'finished_utc': utc_now(), 'wall_ms': wall_ms}


# ── Parsing the stream (pure) ───────────────────────────────


def parse_stream(stdout: bytes) -> Dict[str, Any]:
    """Splits the CLI's stream-json output into its events.

    Input:  stdout — raw CLI stdout (one JSON object per line)
    Output: {'init': init event or None, 'assistant': [assistant events],
            'result': result event or None, 'other_types': [types of other
            events], 'bad_lines': number of lines that are not JSON}
    """
    out: Dict[str, Any] = {'init': None, 'assistant': [], 'result': None, 'other_types': [], 'bad_lines': 0}
    for line in stdout.decode('utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            out['bad_lines'] += 1
            continue
        kind = ev.get('type')
        if kind == 'system' and ev.get('subtype') == 'init' and out['init'] is None:
            out['init'] = ev
        elif kind == 'assistant':
            out['assistant'].append(ev)
        elif kind == 'result':
            out['result'] = ev
        else:
            out['other_types'].append(f"{kind}/{ev.get('subtype')}" if ev.get('subtype') else str(kind))
    return out


def response_text(events: Dict[str, Any]) -> str:
    """Returns the final text of the model: the result event's text, else the assistant text blocks.

    Input:  events — output of parse_stream
    Output: text ('' if there is none)
    """
    res = events.get('result') or {}
    if isinstance(res.get('result'), str) and res['result']:
        return res['result']
    parts = []
    for ev in events.get('assistant', []):
        for block in (ev.get('message') or {}).get('content') or []:
            if block.get('type') == 'text':
                parts.append(block.get('text', ''))
    return ''.join(parts)


def isolation_evidence(init: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Collects what the init event says about the session's context.

    Input:  init — init event of the stream, or None
    Output: {'ok': bool (every ISOLATION_EMPTY_FIELDS entry is an empty
            list), the fields themselves, 'plugins', 'agents', 'model',
            'claude_code_version', 'apiKeySource', 'permissionMode',
            'output_style', 'memory_paths_keys' (names of the memory
            directories the init event lists; their paths are not kept)}
    """
    if init is None:
        return {'ok': False, 'reason': 'no init event'}
    ev = {k: init.get(k) for k in ISOLATION_EMPTY_FIELDS}
    ev['ok'] = all(init.get(k) == [] for k in ISOLATION_EMPTY_FIELDS)
    for k in ('plugins', 'agents', 'model', 'claude_code_version', 'apiKeySource', 'permissionMode',
              'output_style'):
        ev[k] = init.get(k)
    mem = init.get('memory_paths')
    ev['memory_paths_keys'] = sorted(mem) if isinstance(mem, dict) else None
    return ev


def call_metrics(events: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts model ids, durations, tokens and errors from the parsed stream.

    'model' is the modelUsage entry with the most output tokens; the other
    entries are the CLI's side requests ('side_requests', e.g. the session
    title).  'prompt_tokens_total' is the whole prompt of the main request
    (uncached + cache-write + cache-read input tokens): the size of
    everything the model received, including what the CLI adds.
    'thinking_tokens' (output_tokens_details) are part of 'output_tokens'.

    Input:  events — output of parse_stream
    Output: dictionary of metadata fields (None where the stream has no value)
    """
    res = events.get('result') or {}
    usage = res.get('usage') or {}
    model_usage = res.get('modelUsage') or {}
    msg_models = sorted({(ev.get('message') or {}).get('model') for ev in events.get('assistant', [])} - {None})
    thinking = sum(1 for ev in events.get('assistant', [])
                   for b in (ev.get('message') or {}).get('content') or []
                   if b.get('type') in ('thinking', 'redacted_thinking'))
    api_errors = [{k: ev.get(k) for k in ('error', 'api_error', 'request_id')}
                  for ev in events.get('assistant', []) if ev.get('is_api_error_message') or ev.get('error')]
    details = usage.get('output_tokens_details') or {}
    main_model = max(model_usage, key=lambda m: (model_usage[m] or {}).get('outputTokens') or 0, default=None)
    side = {m: {k: (u or {}).get(k) for k in ('inputTokens', 'outputTokens', 'cacheReadInputTokens',
                                               'cacheCreationInputTokens')}
            for m, u in sorted(model_usage.items()) if m != main_model}
    return {
        'model': main_model,
        'models_in_model_usage': sorted(model_usage),
        'side_requests': side,
        'assistant_message_models': msg_models,
        'init_model': (events.get('init') or {}).get('model'),
        'duration_ms': res.get('duration_ms'),
        'duration_api_ms': res.get('duration_api_ms'),
        'num_turns': res.get('num_turns'),
        'result_subtype': res.get('subtype'),
        'is_error': res.get('is_error'),
        'stop_reason': res.get('stop_reason'),
        'input_tokens': usage.get('input_tokens'),
        'output_tokens': usage.get('output_tokens'),
        'thinking_tokens': details.get('thinking_tokens'),
        'cache_creation_input_tokens': usage.get('cache_creation_input_tokens'),
        'cache_read_input_tokens': usage.get('cache_read_input_tokens'),
        'prompt_tokens_total': _total([usage.get('input_tokens'), usage.get('cache_creation_input_tokens'),
                                       usage.get('cache_read_input_tokens')]),
        'total_cost_usd_reported': res.get('total_cost_usd'),
        'usage': usage,
        'modelUsage': model_usage,
        'thinking_blocks': thinking,
        'api_errors': api_errors,
        'other_event_types': events.get('other_types'),
        'unparsed_lines': events.get('bad_lines'),
    }


def extract_kotlin(text: str) -> Dict[str, Any]:
    """Extracts the Kotlin source from a model response.

    Rules: the content of the single ```kotlin (or ```kt) fenced block; with
    several such blocks, the first one that declares 'fun GeneratedScreen()'
    (else the first); with an opening ```kotlin fence that is never closed,
    everything after it; with no Kotlin fence at all, the whole text.  The
    selected lines are kept verbatim (a final newline is added).

    Input:  text — response text
    Output: {'code', 'method' ('kotlin_block', 'kotlin_block_of_several',
            'unterminated_kotlin_block' or 'whole_text'), 'kotlin_blocks',
            'other_blocks', 'flag' (True unless the method is 'kotlin_block')}
    """
    blocks = [(m.group(1).strip().lower(), m.group(2)) for m in _FENCE_RE.finditer(text)]
    kotlin = [body for info, body in blocks if info in ('kotlin', 'kt')]
    other = len(blocks) - len(kotlin)
    if len(kotlin) == 1:
        code, method = kotlin[0], 'kotlin_block'
    elif kotlin:
        code = next((b for b in kotlin if _SCREEN_FN_RE.search(b)), kotlin[0])
        method = 'kotlin_block_of_several'
    else:
        m = _OPEN_KOTLIN_RE.search(text)
        if m:
            code, method = text[m.end():], 'unterminated_kotlin_block'
        else:
            code, method = text, 'whole_text'
    if not code.endswith('\n'):
        code += '\n'
    return {'code': code, 'method': method, 'kotlin_blocks': len(kotlin), 'other_blocks': other,
            'flag': method != 'kotlin_block'}


def kotlin_facts(code: str) -> Dict[str, Any]:
    """Describes an extracted Kotlin source.

    Input:  code — Kotlin source
    Output: {'sha256', 'bytes', 'lines', 'defines_GeneratedScreen',
            'has_package_line', 'text_calls' (occurrences of 'Text(')}
    """
    raw = code.encode('utf-8')
    return {'sha256': sha256_bytes(raw), 'bytes': len(raw), 'lines': len(code.splitlines()),
            'defines_GeneratedScreen': _SCREEN_FN_RE.search(code) is not None,
            'has_package_line': _PACKAGE_RE.search(code) is not None,
            'text_calls': len(_TEXT_CALL_RE.findall(code))}


def call_error(run: Dict[str, Any], metrics: Dict[str, Any], text: str,
               isolation: Dict[str, Any]) -> Optional[str]:
    """Decides whether a call failed and why.

    Input:  run — output of run_cli; metrics — output of call_metrics;
            text — response text; isolation — output of isolation_evidence
    Output: error description (all reasons, '; '-separated), or None for a
            successful call
    """
    reasons = []
    if run['timed_out']:
        reasons.append(f"timed out after {run['wall_ms']} ms")
    if run['exit_code'] != 0:
        reasons.append(f"CLI exit code {run['exit_code']}")
    if metrics['api_errors']:
        reasons.append(f"API error: {metrics['api_errors'][0]}")
    if metrics['is_error']:
        reasons.append(f"result is_error ({metrics['result_subtype']}): {text[:300]}")
    if not isolation.get('ok'):
        reasons.append(f"session not isolated: {isolation}")
    if not text.strip():
        reasons.append('empty response')
    return '; '.join(reasons) or None


# ── One call and its files ──────────────────────────────────


class CallLog:
    """Append-only JSON-lines log of CLI calls (one 'start' and one 'end' line per call)."""

    def __init__(self, path: Path):
        """Opens (creates) the log.

        Input:  path — log file
        Output: none
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def started_calls(self) -> int:
        """Counts the calls logged so far (their 'start' lines).

        Input:  none
        Output: number of calls
        """
        n = 0
        for line in self.path.read_text(encoding='utf-8').splitlines():
            try:
                n += json.loads(line).get('event') == 'start'
            except json.JSONDecodeError:
                continue
        return n

    def write(self, entry: Dict[str, Any]):
        """Appends one entry and flushes it to disk.

        Input:  entry — JSON-serialisable dictionary
        Output: none
        """
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())


def write_sanitized(out_dir: Path, raw_dir: Path, stdout: bytes, stderr: bytes,
                    extra_paths: Sequence[Tuple[Path, str]]) -> Tuple[Redactor, Dict[str, Any], Dict[str, Any]]:
    """Writes the sanitized copies of stdout and stderr to the published directory.

    Input:  out_dir — published call directory; raw_dir — raw call directory
            (replaced by '<raw>'); stdout, stderr — raw CLI output;
            extra_paths — further [(directory, placeholder)]
    Output: (Redactor, meta 'raw' section, meta 'sanitized' section without
            the residual check)
    """
    s_out, s_err, red = sanitize_outputs(stdout, stderr, [(raw_dir, '<raw>')] + list(extra_paths))
    (out_dir / STREAM_SANITIZED_FILE).write_bytes(s_out)
    (out_dir / STDERR_SANITIZED_FILE).write_bytes(s_err)
    raw = {'files': [STREAM_FILE, STDERR_FILE, META_FILE],
           'stream_sha256': sha256_bytes(stdout), 'stderr_sha256': sha256_bytes(stderr),
           'note': 'verbatim CLI output, kept outside the repository (not published)'}
    sanitized = {'files': [STREAM_SANITIZED_FILE, STDERR_SANITIZED_FILE],
                 'stream_sha256': sha256_bytes(s_out), 'stderr_sha256': sha256_bytes(s_err),
                 'redacted_keys': sorted(REDACT_KEYS),
                 'replacements': {k: v for k, v in sorted(red.counts.items()) if v}}
    return red, raw, sanitized


def finish_meta(out_dir: Path, raw_dir: Path, meta: Dict[str, Any], red: Redactor,
                published_texts: Sequence[str]) -> Dict[str, Any]:
    """Writes meta.json raw and sanitized, with the residual privacy check of the published files.

    Input:  out_dir, raw_dir — call directories; meta — metadata (with its
            'sanitized' section); red — Redactor of the call;
            published_texts — sanitized stream and stderr
    Output: the sanitized meta (as written to <out_dir>/meta.json)
    """
    write_json(raw_dir / META_FILE, meta)
    pub = red.json(meta)
    residual = residual_findings(red, list(published_texts) + [json.dumps(pub, ensure_ascii=False)])
    pub['sanitized']['residual_findings'] = residual
    write_json(out_dir / META_FILE, pub)
    if residual:
        print(f"  WARNING: sanitized files of {rel(out_dir)} still contain {residual}", file=sys.stderr)
    return pub


def call_and_save(kind: str, label: str, out_dir: Path, raw_dir: Path, text: str, png: Optional[bytes],
                  base_meta: Dict[str, Any], argv: List[str], env: Dict[str, str], cwd: Path,
                  timeout_s: int, log: CallLog) -> Dict[str, Any]:
    """Makes one CLI call and writes its files.

    Input:  kind — 'generate' or 'probe'; label — '<stem>/run<k>' or 'probe';
            out_dir — published directory of the call (created); raw_dir —
            raw directory of the call (created); text, png — message content
            (see user_message); base_meta — fields copied into meta.json;
            argv, env, cwd, timeout_s — see run_cli; log — call log
    Output: the sanitized meta dictionary (also written to <out_dir>/meta.json)
    """
    before = dir_entries(cwd)
    if before:
        raise RuntimeError(f"working directory is not empty: {before}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    call_no = log.started_calls() + 1
    log.write({'event': 'start', 'call': call_no, 'time_utc': utc_now(), 'kind': kind, 'label': label,
               'argv': list(argv), 'out_dir': rel(out_dir)})
    run = run_cli(argv, env, cwd, user_message(text, png), timeout_s)
    after = dir_entries(cwd)
    (raw_dir / STREAM_FILE).write_bytes(run['stdout'])
    (raw_dir / STDERR_FILE).write_bytes(run['stderr'])
    events = parse_stream(run['stdout'])
    reply = response_text(events)
    (out_dir / RESPONSE_FILE).write_text(reply, encoding='utf-8')
    metrics = call_metrics(events)
    isolation = isolation_evidence(events['init'])
    error = call_error(run, metrics, reply, isolation)
    if error is None and after:
        error = f"working directory not empty after the call: {after}"
    red, raw_sec, san_sec = write_sanitized(out_dir, raw_dir, run['stdout'], run['stderr'], [(cwd, '<cwd>')])
    meta = dict(base_meta)
    meta.update({
        'call': call_no,
        'kind': kind,
        'started_utc': run['started_utc'],
        'finished_utc': run['finished_utc'],
        'wall_ms': run['wall_ms'],
        'exit_code': run['exit_code'],
        'timed_out': run['timed_out'],
        'error': error,
        'argv': list(argv),
        'environment': sorted(env),
        'environment_set': dict(ENV_FIXED),
        'cwd_empty_before': not before,
        'cwd_empty_after': not after,
        'isolation': isolation,
        'message_blocks': (['image'] if png is not None else []) + ['text'],
        'response_sha256': sha256_bytes(reply.encode('utf-8')),
        'response_chars': len(reply),
        'response_privacy_scan': red.scan(reply),
        'stream_sha256': raw_sec['stream_sha256'],
        'raw': raw_sec,
        'sanitized': san_sec,
    })
    meta.update(metrics)
    if kind == 'generate':
        ext = extract_kotlin(reply)
        (out_dir / KOTLIN_FILE).write_text(ext['code'], encoding='utf-8')
        meta['extraction'] = {k: v for k, v in ext.items() if k != 'code'}
        meta['kotlin'] = kotlin_facts(ext['code'])
    pub = finish_meta(out_dir, raw_dir, meta, red,
                      [(out_dir / STREAM_SANITIZED_FILE).read_text(encoding='utf-8'),
                       (out_dir / STDERR_SANITIZED_FILE).read_text(encoding='utf-8')])
    log.write({'event': 'end', 'call': call_no, 'time_utc': utc_now(), 'kind': kind, 'label': label,
               'exit_code': run['exit_code'], 'timed_out': run['timed_out'], 'error': error,
               'model': metrics['model'], 'duration_ms': metrics['duration_ms'],
               'input_tokens': metrics['input_tokens'], 'output_tokens': metrics['output_tokens']})
    return pub


# ── Generation and probe ────────────────────────────────────


def plan_targets(examples: Sequence[Path], runs: Sequence[int], out: Path,
                 raw: Path) -> List[Tuple[Path, int, Path, Path]]:
    """Lists the (image, run, published directory, raw directory) of every planned call.

    Input:  examples — PNG files; runs — run numbers; out — published root; raw — raw root
    Output: list ordered by image, then run
    """
    return [(img, k, out / img.stem / f"run{k}", raw / img.stem / f"run{k}") for img in examples for k in runs]


def prepare_cwd(arg: Optional[str]) -> Tuple[Path, bool]:
    """Returns the empty working directory of the CLI calls.

    Input:  arg — --cwd value (must exist and be empty), or None for a new
            temporary directory
    Output: (directory, True if it was created here and may be removed)
    Raises: ValueError when the given directory is missing or not empty
    """
    if arg is None:
        return Path(tempfile.mkdtemp(prefix='mllm_cwd_')), True
    p = Path(arg).resolve()
    if not p.is_dir():
        raise ValueError(f"--cwd is not a directory: {arg}")
    if dir_entries(p):
        raise ValueError(f"--cwd is not empty: {arg}")
    if inside_repo(p):
        raise ValueError("--cwd must lie outside the repository")
    return p, False


def common_meta(args: argparse.Namespace, claude: str, env: Dict[str, str], version: Optional[str],
                prompts: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    """Returns the meta.json fields shared by every call of one invocation.

    Input:  args — parsed arguments; claude — CLI path; env — CLI environment;
            version — `claude --version`; prompts — {file name: (text, sha256)}
    Output: dictionary
    """
    return {
        'requested_model': args.model,
        'cli': {'executable': claude, 'version': version},
        'prompts_sha256': {name: sha for name, (_, sha) in prompts.items()},
        'runner': rel(Path(__file__)),
        'runner_sha256': sha256_bytes(Path(__file__).read_bytes()),
    }


def generate(args: argparse.Namespace) -> int:
    """Generation mode (see the module docstring, step 1).

    Input:  args — parsed arguments
    Output: exit status (0 all planned calls succeeded, 1 a call failed or
            was refused, 2 usage error)
    """
    examples_dir = Path(args.examples)
    examples = sorted(examples_dir.glob('*.png'))
    if args.only:
        wanted = set(args.only)
        missing = wanted - {p.stem for p in examples}
        if missing:
            print(f"error: no example image for {sorted(missing)}", file=sys.stderr)
            return 2
        examples = [p for p in examples if p.stem in wanted]
    if not examples:
        print(f"error: no *.png in {examples_dir}", file=sys.stderr)
        return 2
    runs = sorted(set(args.run)) if args.run else list(range(1, args.runs + 1))
    targets = plan_targets(examples, runs, Path(args.out), Path(args.raw_dir))
    existing = [t for t in targets if non_empty(t[2]) or non_empty(t[3])]
    if existing and args.resume:
        targets = [t for t in targets if t not in existing]
        print(f"--resume: skipping {len(existing)} target(s) that already have outputs")
    elif existing and not args.force:
        print("error: outputs exist (use --force to overwrite or --resume to skip them):\n  "
              + '\n  '.join(rel(t[2]) for t in existing), file=sys.stderr)
        return 1
    return run_calls(args, 'generate', [(f"{img.stem}/run{k}", d, r, img) for img, k, d, r in targets])


def probe(args: argparse.Namespace) -> int:
    """Probe mode (see the module docstring, step 2).

    Input:  args — parsed arguments
    Output: exit status (0 success, 1 failure or refused)
    """
    out = Path(args.out)
    raw = Path(args.raw_dir) / out.name
    if (non_empty(out) or non_empty(raw)) and not args.force:
        print(f"error: {rel(out)} or {raw} is not empty (use --force to overwrite)", file=sys.stderr)
        return 1
    return run_calls(args, 'probe', [('probe', out, raw, None)])


def run_calls(args: argparse.Namespace, kind: str,
              calls: List[Tuple[str, Path, Path, Optional[Path]]]) -> int:
    """Runs the planned calls sequentially and stops at the first failure.

    Input:  args — parsed arguments; kind — 'generate' or 'probe'; calls —
            [(label, published directory, raw directory, image or None)]
    Output: exit status (0 all succeeded, 1 a call failed or the budget was exceeded)
    """
    claude = shutil.which(args.claude) or args.claude
    env = cli_environment(claude)
    version = cli_version(claude, env)
    prompts_dir = Path(args.prompts_dir)
    names = (PROMPT_SYSTEM, PROMPT_PROBE if kind == 'probe' else PROMPT_USER)
    prompts = {n: read_prompt(prompts_dir, n) for n in names}
    system_text, user_text = prompts[names[0]][0], prompts[names[1]][0]
    log = CallLog(Path(args.call_log))
    done = log.started_calls()
    print(f"CLI {claude} ({version}); model {args.model}; {len(calls)} call(s) planned; "
          f"{done} already in {args.call_log}")
    print(f"environment: {sorted(env)}; set: {ENV_FIXED}")
    if args.budget is not None and done + len(calls) > args.budget:
        print(f"error: budget exceeded: {done} logged + {len(calls)} planned > {args.budget}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"argv: {cli_argv(claude, args.model, system_text)}")
        for label, d, r, _ in calls:
            print(f"  would call: {label} -> {rel(d)} (raw: {r})")
        return 0
    if not calls:
        return 0
    try:
        cwd, temporary = prepare_cwd(args.cwd)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    argv = cli_argv(claude, args.model, system_text)
    base = common_meta(args, claude, env, version, prompts)
    status = 0
    try:
        for label, d, r, img in calls:
            meta_base = dict(base)
            png = None
            if img is not None:
                png = img.read_bytes()
                meta_base.update({'stem': img.stem, 'run': int(d.name[3:]), 'image': rel(img),
                                  'image_sha256': sha256_bytes(png), 'image_bytes': len(png),
                                  'image_size_px': png_size(png)})
            if args.force:
                for f in CALL_FILES:
                    (d / f).unlink(missing_ok=True)
                    (r / f).unlink(missing_ok=True)
            meta = call_and_save(kind, label, d, r, user_text, png, meta_base, argv, env, cwd,
                                 args.timeout, log)
            state = 'ok' if meta['error'] is None else f"FAILED: {meta['error']}"
            extra = f", {meta['extraction']['method']}" if 'extraction' in meta else ''
            print(f"  call {meta['call']}: {label}: {state} (model {meta['model']}, "
                  f"{meta['duration_ms']} ms, prompt {meta['prompt_tokens_total']} / out "
                  f"{meta['output_tokens']} (thinking {meta['thinking_tokens']}) tokens{extra})", flush=True)
            if meta['error'] is not None:
                print("stopping at the first failed call (no retries)", file=sys.stderr)
                status = 1
                break
    finally:
        if temporary and not dir_entries(cwd):
            cwd.rmdir()
    return status


def resanitize(args: argparse.Namespace) -> int:
    """Re-sanitizing mode (see the module docstring, step 4).

    Input:  args — parsed arguments
    Output: exit status (0 done, 1 refused)
    """
    d = Path(args.resanitize)
    raw = Path(args.raw_dir) / d.name
    if not (d / STREAM_FILE).is_file() or not (d / STDERR_FILE).is_file() or not (d / META_FILE).is_file():
        print(f"error: {rel(d)} has no verbatim {STREAM_FILE}, {STDERR_FILE} and {META_FILE}", file=sys.stderr)
        return 1
    if non_empty(raw):
        print(f"error: {raw} is not empty", file=sys.stderr)
        return 1
    raw.mkdir(parents=True, exist_ok=True)
    stdout, stderr = (d / STREAM_FILE).read_bytes(), (d / STDERR_FILE).read_bytes()
    meta = json.loads((d / META_FILE).read_text(encoding='utf-8'))
    for f in (STREAM_FILE, STDERR_FILE):
        shutil.move(str(d / f), str(raw / f))
    extra = [(Path(args.cwd), '<cwd>')] if args.cwd else []
    red, raw_sec, san_sec = write_sanitized(d, raw, stdout, stderr, extra)
    reply = (d / RESPONSE_FILE).read_text(encoding='utf-8') if (d / RESPONSE_FILE).is_file() else ''
    meta['raw'] = raw_sec
    meta['sanitized'] = dict(san_sec, note=f"re-sanitized {utc_now()} by --resanitize; the call itself "
                                           f"was made by an earlier version of the runner")
    meta['response_privacy_scan'] = red.scan(reply)
    meta.setdefault('isolation', {})['memory_paths_keys'] = isolation_evidence(
        parse_stream(stdout)['init']).get('memory_paths_keys')
    pub = finish_meta(d, raw, meta, red, [(d / STREAM_SANITIZED_FILE).read_text(encoding='utf-8'),
                                          (d / STDERR_SANITIZED_FILE).read_text(encoding='utf-8')])
    print(f"{rel(d)}: raw files moved to {raw}; replacements {pub['sanitized']['replacements']}; "
          f"residual {pub['sanitized']['residual_findings'] or 'none'}")
    return 0


# ── Summary ─────────────────────────────────────────────────


def normalized(code: str) -> str:
    """Removes all whitespace (for a whitespace-insensitive comparison).

    Input:  code — source text
    Output: text without any whitespace character
    """
    return re.sub(r'\s+', '', code)


def determinism(a: str, b: str) -> Dict[str, Any]:
    """Compares two runs of the same screen.

    Input:  a, b — Kotlin sources of run 1 and run 2
    Output: {'byte_identical', 'whitespace_insensitive_identical',
            'similarity_lines' (difflib.SequenceMatcher ratio over the lists of
            lines), 'similarity_chars' (the same over the characters)}
    """
    la, lb = a.splitlines(), b.splitlines()
    return {
        'byte_identical': a.encode('utf-8') == b.encode('utf-8'),
        'whitespace_insensitive_identical': normalized(a) == normalized(b),
        'similarity_lines': round(difflib.SequenceMatcher(None, la, lb, autojunk=False).ratio(), 4),
        'similarity_chars': round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 4),
    }


def _report(report_file: Path) -> Dict[str, Any]:
    """Reads a harness report.

    Input:  report_file — compile_report.json written by the harness
    Output: report dictionary; {} when the file is missing
    """
    if not report_file.is_file():
        return {}
    return json.loads(report_file.read_text(encoding='utf-8'))


def _report_entries(report_file: Path) -> Dict[str, Dict[str, Any]]:
    """Reads a harness report and indexes its entries by source path.

    Input:  report_file — compile_report.json written by the harness
    Output: {repo-relative source path: entry}; {} when the file is missing
    """
    return {e['source']: e for e in _report(report_file).get('files', [])}


def _first_error(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    """Returns the first compiler error (or the build note) of a harness report entry.

    Input:  entry — report entry or None
    Output: message or None
    """
    if not entry or entry.get('compiled'):
        return None
    if entry.get('compile_errors'):
        return entry['compile_errors'][0]
    return entry.get('compile_note')


def _total(values: Sequence[Optional[float]]) -> Optional[float]:
    """Sums the values that are not None.

    Input:  values — numbers or None
    Output: sum, or None when every value is None
    """
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    """Averages the values that are not None (rounded to one decimal).

    Input:  values — numbers or None
    Output: mean, or None when every value is None
    """
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def summarize(args: argparse.Namespace) -> int:
    """Summary mode (see the module docstring, step 3).

    Input:  args — parsed arguments
    Output: exit status (0 written, 2 nothing to summarise)
    """
    out = Path(args.out)
    report_file = out / 'compile_report.json'
    claude_report = _report_entries(report_file)
    ref_report = _report_entries(REPO_ROOT / UI2COMPOSE_REPORT)
    stems = sorted(p.stem for p in Path(args.examples).glob('*.png'))
    screens, all_runs, metas = [], [], []
    for stem in stems:
        run_dirs = sorted((out / stem).glob('run*'), key=lambda d: int(d.name[3:]) if d.name[3:].isdigit() else 0)
        runs = []
        for d in run_dirs:
            meta_file = d / META_FILE
            if not meta_file.is_file():
                continue
            meta = json.loads(meta_file.read_text(encoding='utf-8'))
            metas.append(meta)
            src = rel(d / KOTLIN_FILE)
            entry = claude_report.get(src)
            side = meta.get('side_requests') or {}
            runs.append({
                'run': meta.get('run'),
                'call': meta.get('call'),
                'error': meta.get('error'),
                'model': meta.get('model'),
                'side_request_models': sorted(side),
                'started_utc': meta.get('started_utc'),
                'duration_ms': meta.get('duration_ms'),
                'duration_api_ms': meta.get('duration_api_ms'),
                'wall_ms': meta.get('wall_ms'),
                'prompt_tokens_total': meta.get('prompt_tokens_total'),
                'input_tokens': meta.get('input_tokens'),
                'cache_creation_input_tokens': meta.get('cache_creation_input_tokens'),
                'cache_read_input_tokens': meta.get('cache_read_input_tokens'),
                'output_tokens': meta.get('output_tokens'),
                'thinking_tokens': meta.get('thinking_tokens'),
                'side_request_input_tokens': _total([(u or {}).get('inputTokens') for u in side.values()]),
                'side_request_output_tokens': _total([(u or {}).get('outputTokens') for u in side.values()]),
                'stop_reason': meta.get('stop_reason'),
                'extraction': (meta.get('extraction') or {}).get('method'),
                'kotlin_lines': (meta.get('kotlin') or {}).get('lines'),
                'kotlin_sha256': (meta.get('kotlin') or {}).get('sha256'),
                'defines_GeneratedScreen': (meta.get('kotlin') or {}).get('defines_GeneratedScreen'),
                'text_calls': (meta.get('kotlin') or {}).get('text_calls'),
                'response_privacy_scan': meta.get('response_privacy_scan'),
                'source': src,
                'in_harness_report': entry is not None,
                'compiled': bool(entry and entry.get('compiled')),
                'first_compile_error': _first_error(entry),
                'compile_errors': len(entry.get('compile_errors', [])) if entry else None,
                'compile_warnings': len(entry.get('compile_warnings', [])) if entry else None,
                'rendered': bool(entry and entry.get('rendered')),
                'render': entry.get('render') if entry else None,
                'render_error': entry.get('render_error') if entry else None,
            })
        if not runs:
            continue
        codes = {r['run']: (out / stem / f"run{r['run']}" / KOTLIN_FILE).read_text(encoding='utf-8')
                 for r in runs}
        det = determinism(codes[1], codes[2]) if 1 in codes and 2 in codes else None
        ref = ref_report.get((UI2COMPOSE_EXAMPLES / stem / '4_GeneratedScreen.kt').as_posix())
        ref_kt = REPO_ROOT / UI2COMPOSE_EXAMPLES / stem / '4_GeneratedScreen.kt'
        screens.append({
            'stem': stem,
            'runs': runs,
            'determinism_run1_run2': det,
            'ui2compose_v1.2.0': {
                'compiled': bool(ref and ref.get('compiled')),
                'rendered': bool(ref and ref.get('rendered')),
                'render': ref.get('render') if ref else None,
                'kotlin_lines': len(ref_kt.read_text(encoding='utf-8').splitlines()) if ref_kt.is_file() else None,
            },
        })
        all_runs += runs
    if not screens:
        print(f"error: no run outputs below {out}", file=sys.stderr)
        return 2
    dets = [s['determinism_run1_run2'] for s in screens if s['determinism_run1_run2']]
    models = sorted({r['model'] for r in all_runs if r['model']})
    starts = sorted(r['started_utc'] for r in all_runs if r['started_utc'])
    harness = _report(report_file)
    col = lambda key: [r[key] for r in all_runs]  # noqa: E731
    summary = {
        'generated_by': 'scripts/mllm_compare_claude.py --summarize',
        'protocol': 'results/mllm_claude/README.md',
        'model': models[0] if len(models) == 1 else models,
        'requested_model': sorted({m.get('requested_model') for m in metas}),
        'side_request_models': sorted({m for r in all_runs for m in r['side_request_models']}),
        'cli_version': sorted({(m.get('cli') or {}).get('version') for m in metas} - {None}),
        'claude_code_version_init': sorted({(m.get('isolation') or {}).get('claude_code_version')
                                            for m in metas} - {None}),
        'argv_variants': len({json.dumps(m.get('argv')) for m in metas}),
        'argv': metas[0].get('argv') if metas else None,
        'environment_set': [json.loads(e) for e in sorted({json.dumps(m.get('environment_set'), sort_keys=True)
                                                            for m in metas})],
        'runner_sha256': sorted({m.get('runner_sha256') for m in metas} - {None}),
        'prompts_sha256': metas[0].get('prompts_sha256') if metas else None,
        'date_utc': starts[0][:10] if starts else None,
        'first_call_utc': starts[0] if starts else None,
        'last_call_utc': starts[-1] if starts else None,
        'harness_report': rel(report_file) if report_file.is_file() else None,
        'harness_toolchain': harness.get('toolchain'),
        'harness_snapshot': harness.get('snapshot'),
        'definitions': {
            'compiled': "harness/compile_and_render.sh compiled the extracted GeneratedScreen.kt unchanged "
                        "(the harness only inserts a package line in its copy)",
            'rendered': "the Paparazzi snapshot test passed and its PNG exists (same set-up as for ui2compose)",
            'byte_identical': "GeneratedScreen.kt of run 1 and run 2 have identical bytes",
            'whitespace_insensitive_identical': "identical after removing every whitespace character",
            'similarity_lines': "difflib.SequenceMatcher(None, run1.splitlines(), run2.splitlines(), "
                                "autojunk=False).ratio()",
            'similarity_chars': "the same ratio over the characters of the two files",
            'duration_ms': "duration_ms of the CLI result event (whole call, CLI start-up included)",
            'duration_api_ms': "duration_api_ms of the CLI result event (time spent in API requests, side "
                               "requests included)",
            'prompt_tokens_total': "input + cache-creation + cache-read input tokens of the main request: "
                                   "everything the model received (CLI preamble, system prompt, image, text)",
            'input_tokens': "usage.input_tokens of the result event (uncached input only)",
            'output_tokens': "usage.output_tokens of the main model (thinking tokens included)",
            'thinking_tokens': "usage.output_tokens_details.thinking_tokens (part of output_tokens)",
            'side_request': "further modelUsage entries of the call (the CLI's own requests, e.g. the session "
                            "title); their output is not part of the response",
        },
        'totals': {
            'screens': len(screens),
            'calls': len(all_runs),
            'failed_calls': sum(1 for r in all_runs if r['error']),
            'extraction_methods': dict(collections.Counter(col('extraction'))),
            'stop_reasons': dict(collections.Counter(col('stop_reason'))),
            'defines_GeneratedScreen': sum(1 for r in all_runs if r['defines_GeneratedScreen']),
            'compiled': sum(1 for r in all_runs if r['compiled']),
            'rendered': sum(1 for r in all_runs if r['rendered']),
            'screens_compiled_both_runs': sum(1 for s in screens if all(r['compiled'] for r in s['runs'])),
            'screens_compiled_any_run': sum(1 for s in screens if any(r['compiled'] for r in s['runs'])),
            'screens_rendered_both_runs': sum(1 for s in screens if all(r['rendered'] for r in s['runs'])),
            'screens_rendered_any_run': sum(1 for s in screens if any(r['rendered'] for r in s['runs'])),
            'pairs': len(dets),
            'byte_identical_pairs': sum(1 for d in dets if d['byte_identical']),
            'whitespace_insensitive_identical_pairs': sum(1 for d in dets if d['whitespace_insensitive_identical']),
            'similarity_lines_mean': round(sum(d['similarity_lines'] for d in dets) / len(dets), 4) if dets else None,
            'similarity_lines_min': min((d['similarity_lines'] for d in dets), default=None),
            'similarity_lines_max': max((d['similarity_lines'] for d in dets), default=None),
            'similarity_chars_mean': round(sum(d['similarity_chars'] for d in dets) / len(dets), 4) if dets else None,
            'duration_ms_sum': _total(col('duration_ms')),
            'duration_ms_mean': _mean(col('duration_ms')),
            'duration_ms_min': min((v for v in col('duration_ms') if v is not None), default=None),
            'duration_ms_max': max((v for v in col('duration_ms') if v is not None), default=None),
            'duration_api_ms_sum': _total(col('duration_api_ms')),
            'duration_api_ms_mean': _mean(col('duration_api_ms')),
            'prompt_tokens_total_sum': _total(col('prompt_tokens_total')),
            'prompt_tokens_total_mean': _mean(col('prompt_tokens_total')),
            'input_tokens_sum': _total(col('input_tokens')),
            'output_tokens_sum': _total(col('output_tokens')),
            'output_tokens_mean': _mean(col('output_tokens')),
            'thinking_tokens_sum': _total(col('thinking_tokens')),
            'thinking_tokens_mean': _mean(col('thinking_tokens')),
            'side_request_input_tokens_sum': _total(col('side_request_input_tokens')),
            'side_request_output_tokens_sum': _total(col('side_request_output_tokens')),
            'kotlin_lines_mean': _mean(col('kotlin_lines')),
            'ui2compose_v1.2.0_compiled': sum(1 for s in screens if s['ui2compose_v1.2.0']['compiled']),
            'ui2compose_v1.2.0_rendered': sum(1 for s in screens if s['ui2compose_v1.2.0']['rendered']),
        },
        'screens': screens,
    }
    write_json(out / 'summary.json', summary)
    t = summary['totals']
    print(f"{t['compiled']}/{t['calls']} compiled, {t['rendered']}/{t['calls']} rendered; "
          f"{t['byte_identical_pairs']}/{t['pairs']} byte-identical pairs; written {rel(out / 'summary.json')}")
    return 0


# ── Command line ────────────────────────────────────────────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point (see the module docstring).

    Input:  argv — command-line arguments (default: sys.argv[1:])
    Output: exit status
    """
    p = argparse.ArgumentParser(description="Claude (Claude Code CLI) baseline on the example screenshots.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--probe', action='store_true', help="isolation probe (one text-only call)")
    mode.add_argument('--summarize', action='store_true', help="write <out>/summary.json (no model call)")
    mode.add_argument('--resanitize', metavar='DIR',
                      help="move the verbatim files of an older call directory to --raw-dir and sanitize it")
    p.add_argument('--examples', default=str(EXAMPLES_DIR), help="directory of the example PNGs")
    p.add_argument('--prompts-dir', default=str(OUT_DIR), help="directory of the prompt files")
    p.add_argument('--out', default=str(OUT_DIR), help="output root (for --probe: the probe directory)")
    p.add_argument('--raw-dir', help="root of the verbatim CLI output, outside the repository "
                                     "(required except for --summarize)")
    p.add_argument('--runs', type=int, default=2, help="runs per screen (run1..runN)")
    p.add_argument('--run', type=int, action='append', help="only this run number (repeatable)")
    p.add_argument('--only', action='append', help="only this example stem (repeatable)")
    p.add_argument('--force', action='store_true', help="overwrite existing outputs of the selected calls")
    p.add_argument('--resume', action='store_true', help="skip selected calls that already have outputs")
    p.add_argument('--model', default=DEFAULT_MODEL, help="model passed to --model")
    p.add_argument('--claude', default='claude', help="Claude Code CLI executable")
    p.add_argument('--cwd', help="existing empty directory to run the CLI in (default: a new temporary one)")
    p.add_argument('--timeout', type=int, default=DEFAULT_TIMEOUT_S, help="seconds per call")
    p.add_argument('--call-log', default=str(OUT_DIR / 'calls.log'), help="JSON-lines call log")
    p.add_argument('--budget', type=int, help="refuse to exceed this many calls in the call log")
    p.add_argument('--dry-run', action='store_true', help="print the planned calls and exit")
    args = p.parse_args(argv)
    if args.force and args.resume:
        p.error("--force and --resume exclude each other")
    if not args.summarize:
        if not args.raw_dir:
            p.error("--raw-dir is required (a directory outside the repository)")
        args.raw_dir = str(Path(args.raw_dir).resolve())
        if inside_repo(Path(args.raw_dir)):
            p.error("--raw-dir must lie outside the repository")
    os.chdir(REPO_ROOT)
    if args.summarize:
        return summarize(args)
    if args.resanitize:
        return resanitize(args)
    if args.probe:
        return probe(args)
    return generate(args)


if __name__ == '__main__':
    sys.exit(main())
