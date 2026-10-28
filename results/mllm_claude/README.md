# MLLM baseline: Claude on the bundled example screenshots

Reviewer 2 (R2.11) asked for a qualitative comparison with a multimodal LLM
that generates Compose code from the same screenshots. This folder holds that
comparison. Each of the 10 ReDraw test screens in `examples/` was sent twice
to Claude through the Claude Code CLI, with one fixed prompt and no tools.
The Kotlin that came back was compiled and rendered with the same harness and
toolchain as the ui2compose v1.2.0 output (`harness/`, `results/compile_report.json`).
The model's code was not edited or repaired.

The runner is `scripts/mllm_compare_claude.py`. In the run, `summary.json` held the
numbers and every call had its own folder, `<stem>/run<k>/`; these files are not in
this release (see the note below).

> **Note (27 October 2026): the per-call outputs are not in this release.**
> The local working copy holding the 20 October run was lost before it was
> committed, and the outputs could not be recovered: `calls.log`,
> `summary.json`, `compile_report.json`, `isolation_probe/`, the
> `<stem>/run<k>/` folders, `side_by_side*.png`, and the verbatim CLI output
> kept outside the repository. This folder now holds only the fixed prompts
> (`prompt_*.txt`, `prompts.sha256`), this protocol and `ANALYSIS.md`.
> The numbers in this README, in `ANALYSIS.md` and in the paper were taken
> from those outputs while they existed (20–27 October 2026). They are
> reported as they were, but they can no longer be checked against the files.
> The paper's Fig. 4 (`docs/figures/fig4_claude_comparison.png`) is the image
> drawn from those renders, identical to the one in the manuscript. The run
> was not repeated, because the model is not deterministic and a new run
> would give different code and different numbers. "How to rerun" below
> describes how to produce a new, independent run. The "Files" list
> describes the folder as it was.

## Protocol

| Item | Value |
|---|---|
| Date | 20 October 2026. Generation calls ran from 07:16:25 to 07:24:07 UTC. The isolation probe ran at 06:49:49 UTC. |
| CLI | Claude Code 2.1.267 (`claude --version`: `2.1.267 (Claude Code)`). Logged in with the authors' Claude subscription, not an API key (`apiKeySource: none`). Not updated during the run. |
| Requested model | `--model claude-sonnet-5-5` |
| Model that answered | `claude-sonnet-5-5`, according to `modelUsage` and every assistant message. CLI 2.1.267 does not accept `claude-opus-5-5`: an earlier attempt with that name got a synthetic CLI reply, with no API request and 0 tokens. |
| Runner | `scripts/mllm_compare_claude.py`. All 20 generation calls ran with the same file, SHA-256 `cc30259b7a296f8cdd40698affea2f75c40136040de4a9136f87692c6acf3879` (`runner_sha256` in each `meta.json`). |
| Calls | 10 screens x 2 runs, sequential, each a fresh CLI process. There were no retries and no failed calls. |
| Sampling | CLI defaults. The CLI cannot set temperature. Extended thinking is on by default in this CLI version and was left on. |

### Command line

The command line is identical for every call (`argv` in each `meta.json`):

```
claude -p --model claude-sonnet-5-5 \
  --input-format stream-json --output-format stream-json --verbose \
  --tools "" --setting-sources "" --strict-mcp-config \
  --disable-slash-commands --no-session-persistence \
  --system-prompt "<prompt_system.txt>"
```

- `--tools ""` gives the model no built-in tools.
- `--strict-mcp-config` without `--mcp-config` loads no MCP servers.
- `--setting-sources ""` loads no user, project or local settings, and so no plugins or hooks.
- `--disable-slash-commands` turns off skills and slash commands.
- `--no-session-persistence` writes no session file.
- `--system-prompt` replaces the CLI's default system prompt with `prompt_system.txt`.

Each call reads one stream-json user message from stdin with two blocks: the
example PNG, base64-encoded and unmodified (1200 x 1920 px), followed by the
text of `prompt_user.txt`.

### Environment and working directory

The CLI runs in an empty directory outside the repository. The directory was
checked to be empty before and after every call (`cwd_empty_before` and
`cwd_empty_after`). The environment is minimal: only `HOME`, `USER`, `LOGNAME`,
`SHELL`, `TMPDIR`, `LANG` and `LC_ALL` are passed through. `PATH` is set to the
CLI's directory plus the system directories, and three variables are set:

| Variable | Value | Purpose |
|---|---|---|
| `DISABLE_AUTOUPDATER` | `1` | Stops the CLI from updating itself. |
| `CLAUDE_CODE_ATTRIBUTION_HEADER` | `0` | Removes the attribution header from the system prompt. |
| `CLAUDE_CODE_DISABLE_CLAUDE_MDS` | `1` | Stops the CLI from loading CLAUDE.md memory files. |

All other `CLAUDE*`, `ANTHROPIC*` and `MCP*` variables of the calling shell are dropped.

The CLI's init event shows that each session had no tools, MCP servers, skills,
slash commands or plugins (`isolation` in each `meta.json`). The init event
still lists the CLI's built-in agent names. Without tools, the model cannot
invoke them.

### Prompts

The prompts are sent verbatim, with no newline added or removed.

| File | Bytes | SHA-256 |
|---|---|---|
| `prompt_system.txt` | 41 | `e66836cb4dceb5414ac105d08886dd9f80e399a704346788213ed652485d19b9` |
| `prompt_user.txt` | 630 | `a0d075184886e8f3ccf385ad7687a3a5c2f9ae658d8fbefc3694daa63c60e5e0` |
| `prompt_probe.txt` (probe only) | 453 | `1e014c33929138b54846445b489304d592e6a71e907acc1c3f746aae1a2da45d` |

`prompts.sha256` lists the same hashes.

### What the CLI adds to the prompt

A subscription login cannot stop the CLI from adding to the model's prompt.
The additions are listed below. None of them describes the task or the screen.

- **A fixed preamble before our system prompt.** It contains an identity line
  ("You are a Claude agent, built on Anthropic's Claude Agent SDK."), a
  reminder with the account's e-mail address and the current date, and an
  attribution header. The attribution header is removed by
  `CLAUDE_CODE_ATTRIBUTION_HEADER=0` in the generation runs. The size comes
  from the isolation probe (`isolation_probe/`), which was run before the two
  environment variables were added: about 100 tokens of system prompt and
  probe text produced 630 prompt tokens, so the preamble was about 500
  tokens. About 300 of those tokens were not identified. They may be the
  CLI's memory instructions: the init event lists an automatic memory
  directory (`memory_paths_keys: ["auto"]`).
- **No second probe.** By decision, the probe was not repeated with the two
  environment variables set, so we did not measure how much they shorten the
  preamble. Every generation call had exactly 3,721 prompt tokens
  (`prompt_tokens_total`). That count covers the preamble, both prompts and
  the image, and is the same for every call because every screenshot has the
  same size.
- **No quoted preamble.** In the probe, the model declined to quote its
  context (`isolation_probe/response.txt`). The composition above was
  identified by the authors for CLI 2.1.267 with a subscription login.
- **Paper wording.** The paper describes this as "a short fixed CLI preamble
  (identity line, date, account metadata)".
- **A side request to Haiku.** Every call also sends one request to
  `claude-haiku-4-5-20251001`, which the CLI uses to title the session: 1,058
  input tokens and 15 to 16 output tokens per call. The same request in the
  text-only probe had 981 input tokens. The difference of 77 tokens is far
  smaller than the image, which suggests the title request carries the text
  but not the screenshot. We did not verify this. The output of the side
  request does not reach the main request. Its time is included in
  `duration_api_ms`, which is why `duration_api_ms` is larger than
  `duration_ms` for every call.
- **Hidden thinking.** Extended thinking is on by default. Its text is not
  returned, only an empty thinking block with a signature. The thinking token
  count is reported per call and is part of `output_tokens`. Six of the 20
  calls used no thinking tokens.
- **A model-name warning.** On stderr the CLI prints
  `[claude-code:unrecognized_model] {"model":"claude-sonnet-5-5",...}`
  because version 2.1.267 does not list this model name. The request still
  reached `claude-sonnet-5-5`. The CLI reports its generic limits for it
  (context window 200,000 tokens, maximum output 32,000 tokens), and no
  response came near the output limit: every call ended with `end_turn`.

### Extracting the Kotlin

The extraction rule was fixed in the runner before the runs (`extract_kotlin()`):

- If the response has one ```` ```kotlin ```` block, its content is used.
- If it has several, the first block that declares `fun GeneratedScreen()` is used.
- An unterminated block, or a response with no Kotlin fence, has fallbacks.

`GeneratedScreen.kt` is the selected text, unchanged. In its own copy, the
harness adds only a package line, as it does for every input.

19 of the 20 responses contain exactly one Kotlin block. The exception is
`com.sign.language.learn1234-screens_screenshot_2/run1`. That response
contains a first block, then the sentence "Wait, that draft contains a
broken placeholder. Here is the corrected, complete version:", then a second
block. The rule takes the first block, which fails to compile at the
placeholder the model mentions (`.size width@{ ... }`). We kept the rule
unchanged, so this run counts as a compile failure. The full response,
including the second block, is in `response.txt`.

### Compiling and rendering

The harness compiled and rendered all 20 files in one run:
`harness/compile_and_render.sh --offline` with `--report
results/mllm_claude/compile_report.json`. The settings are the ui2compose
ones:

- Kotlin 2.0.21
- Compose BOM 2024.06.00 (ui 1.6.8, Material3 1.2.1)
- AGP 8.5.2, JDK 17.0.13
- Paparazzi 1.3.4 with `DeviceConfig.NEXUS_7`
- the wrapper `MaterialTheme { Surface(Modifier.fillMaxSize()) { GeneratedScreen() } }`

The harness compiles and does not repair. A file with errors is dropped from
the next compile round. Each render is saved as `render.png` next to its
`GeneratedScreen.kt`.

## Results

**Compilation and rendering.** 18 of the 20 files compiled and rendered.
Every screen has at least one run that compiled and rendered, and 8 of the 10
screens compiled in both runs. The two failures:

| Call | First compiler error | Cause |
|---|---|---|
| `com.mineworld.swordsaddon-screens_screenshot_2/run1` | `16:43 Cannot access 'val RowColumnParentData?.weight: Float': it is internal in file.` | The code imports `androidx.compose.foundation.layout.weight`, which is a scope member and cannot be imported. |
| `com.sign.language.learn1234-screens_screenshot_2/run1` | `146:38 None of the following candidates is applicable:` | The draft block described in "Extracting the Kotlin". |

**Warnings.** Both runs of `com.andromo.dev231870.app482676 #5` produced two
deprecation warnings (`Divider` renamed to `HorizontalDivider`). The other
files compiled without warnings.

**Run-to-run comparison.** No screen produced the same code twice, either
byte for byte or after removing all whitespace. The difflib similarity
between run 1 and run 2 averages 0.44 over lines (range 0.23 to 0.63) and
0.64 over characters. For comparison, all 10 ui2compose v1.2.0 files compiled
and rendered (`results/compile_report.json`).

| Screen | Run 1 compiled / rendered | Run 2 compiled / rendered | Lines run 1 / 2 | Byte-identical | Whitespace-insensitive identical | difflib ratio (lines / chars) | ui2compose v1.2.0 compiled / rendered (lines) |
|---|---|---|---|---|---|---|---|
| com.andromo.dev231870.app482676 #5 | yes / yes | yes / yes | 164 / 206 | no | no | 0.51 / 0.77 | yes / yes (187) |
| com.foxfi #1 | yes / yes | yes / yes | 231 / 214 | no | no | 0.49 / 0.73 | yes / yes (277) |
| com.luigiedeveloper.battlesimulator #3 | yes / yes | yes / yes | 174 / 186 | no | no | 0.63 / 0.83 | yes / yes (470) |
| com.mcdeliveryonline.app #1 | yes / yes | yes / yes | 254 / 224 | no | no | 0.38 / 0.45 | yes / yes (89) |
| com.mineworld.swordsaddon #2 | no / no | yes / yes | 211 / 211 | no | no | 0.48 / 0.66 | yes / yes (141) |
| com.mynamecubeapps.ball #3 | yes / yes | yes / yes | 269 / 220 | no | no | 0.38 / 0.58 | yes / yes (67) |
| com.netbeast.yeti #1 | yes / yes | yes / yes | 158 / 145 | no | no | 0.23 / 0.58 | yes / yes (54) |
| com.shazam.android #3 | yes / yes | yes / yes | 290 / 217 | no | no | 0.28 / 0.38 | yes / yes (159) |
| com.sign.language.learn1234 #2 | no / no | yes / yes | 168 / 210 | no | no | 0.44 / 0.63 | yes / yes (112) |
| westport.andrewirwin.com.drivesafe #3 | yes / yes | yes / yes | 219 / 214 | no | no | 0.58 / 0.76 | yes / yes (148) |

**Timing and tokens.** These are the CLI's own figures from the result event.

| Measure | Value |
|---|---|
| `duration_ms` per call | mean 21.2 s, range 10.2 to 32.8 s, total 7.1 min |
| Prompt tokens per call | 3,721 (identical for every call) |
| Output tokens | mean 3,509, total 70,177 |
| Thinking tokens (part of output) | mean 595, total 11,891 |
| Haiku side requests | 21,160 input and 301 output tokens in total |
| Kotlin length | mean 209 lines |

| Screen | Run | Call | duration_ms | duration_api_ms | Output tokens | Thinking tokens | Extraction |
|---|---|---|---|---|---|---|---|
| com.andromo.dev231870.app482676 #5 | 1 | 3 | 12838 | 13943 | 2263 | 0 | kotlin_block |
| com.andromo.dev231870.app482676 #5 | 2 | 4 | 14903 | 15948 | 2634 | 0 | kotlin_block |
| com.foxfi #1 | 1 | 5 | 21015 | 22102 | 3633 | 601 | kotlin_block |
| com.foxfi #1 | 2 | 6 | 15289 | 16589 | 2896 | 0 | kotlin_block |
| com.luigiedeveloper.battlesimulator #3 | 1 | 7 | 20160 | 21316 | 3310 | 569 | kotlin_block |
| com.luigiedeveloper.battlesimulator #3 | 2 | 8 | 19136 | 20130 | 3218 | 455 | kotlin_block |
| com.mcdeliveryonline.app #1 | 1 | 2 | 26942 | 28026 | 4230 | 1136 | kotlin_block |
| com.mcdeliveryonline.app #1 | 2 | 9 | 23156 | 24385 | 3608 | 988 | kotlin_block |
| com.mineworld.swordsaddon #2 | 1 | 10 | 22107 | 23183 | 3495 | 661 | kotlin_block |
| com.mineworld.swordsaddon #2 | 2 | 11 | 20440 | 21518 | 3199 | 606 | kotlin_block |
| com.mynamecubeapps.ball #3 | 1 | 12 | 21241 | 22398 | 3642 | 0 | kotlin_block |
| com.mynamecubeapps.ball #3 | 2 | 13 | 17471 | 19816 | 3279 | 0 | kotlin_block |
| com.netbeast.yeti #1 | 1 | 14 | 16268 | 17378 | 2536 | 658 | kotlin_block |
| com.netbeast.yeti #1 | 2 | 15 | 10207 | 11281 | 1787 | 0 | kotlin_block |
| com.shazam.android #3 | 1 | 16 | 28953 | 30227 | 4623 | 1209 | kotlin_block |
| com.shazam.android #3 | 2 | 17 | 27239 | 28371 | 4227 | 1314 | kotlin_block |
| com.sign.language.learn1234 #2 | 1 | 18 | 32767 | 33933 | 5840 | 690 | kotlin_block_of_several |
| com.sign.language.learn1234 #2 | 2 | 19 | 20854 | 22135 | 3397 | 646 | kotlin_block |
| westport.andrewirwin.com.drivesafe #3 | 1 | 20 | 30406 | 31677 | 4693 | 1505 | kotlin_block |
| westport.andrewirwin.com.drivesafe #3 | 2 | 21 | 22348 | 24287 | 3667 | 853 | kotlin_block |

## Files

```
prompt_system.txt, prompt_user.txt, prompt_probe.txt, prompts.sha256   the fixed prompts
calls.log                       every CLI call of the published run (start and end line per call)
compile_report.json             harness report for the 20 GeneratedScreen.kt files
summary.json                    per screen and per run results and totals (scripts/mllm_compare_claude.py --summarize)
isolation_probe/                the text-only probe (call 1)
<stem>/run<k>/
  GeneratedScreen.kt            the extracted Kotlin, unchanged
  render.png                    Paparazzi render (only when compiled and rendered)
  response.txt                  the model's final text, verbatim
  stream.sanitized.jsonl        the CLI's stream-json output, sanitized (see below)
  stderr.sanitized.txt          the CLI's stderr, sanitized
  meta.json                     metadata of the call (sanitized)
```

### Sanitizing

The verbatim CLI output contains machine paths and the session id. It is
kept outside the repository and is not published. The published copies were
sanitized as follows:

- **Replaced keys.** The values of `session_id`, `uuid`, `request_id`, `cwd`,
  `memory_paths`, `messaging_socket_path`, `signature` (the opaque
  thinking-block signature) and `rate_limit_info` are replaced by
  placeholders, and so are the message ids.
- **Redacted text.** E-mail addresses, home-directory paths, the user name
  and UUIDs are redacted wherever they appear.
- **Records in `meta.json`.** `raw` has the SHA-256 of the verbatim files.
  `sanitized` has the SHA-256 of the published copies and the number of
  replacements. `sanitized.residual_findings` is empty for every call.
- **Unredacted files.** `response.txt` and `GeneratedScreen.kt` are never
  redacted. `response_privacy_scan` in `meta.json` counts e-mail addresses,
  home paths, user names, session ids and UUIDs in each response, and it is
  zero for all 20 responses.
- **The probe folder.** It was written by an earlier version of the runner
  and was sanitized afterwards with `--resanitize`.

## Calls used

21 calls are logged in `calls.log`, and each made one model request plus the
Haiku title request:

- call 1: the isolation probe;
- call 2: the pilot on `com.mcdeliveryonline.app-screens_screenshot_1`. It
  ran with the final runner and the same settings as the batch, so it counts
  as that screen's run 1;
- calls 3 to 21: the remaining 19 generation calls.

There were no retries. The earlier `claude-opus-5-5` attempt made no API
request and is not in the log.

## How to rerun

From the repository root, with Claude Code 2.1.267 logged in. Keep RAW, CWD
and the call log outside the repository; CWD must be an existing empty
directory. The model output is not deterministic, so a rerun gives different
code. The runner refuses to overwrite existing outputs unless given `--force`.

```bash
python scripts/mllm_compare_claude.py --raw-dir RAW --cwd CWD --call-log RAW/calls.log --runs 2 --budget 20 --dry-run
python scripts/mllm_compare_claude.py --raw-dir RAW --cwd CWD --call-log RAW/calls.log --runs 2 --budget 20 --force
harness/compile_and_render.sh --offline --report results/mllm_claude/compile_report.json \
    --render-dir results/mllm_claude/renders results/mllm_claude
python scripts/mllm_compare_claude.py --summarize
```

The isolation probe is `--probe --out results/mllm_claude/isolation_probe --raw-dir RAW`.
The batch stops at the first failed call. A single call is repeated with
`--only <stem> --run <k> --force`, and `--resume` skips the calls that
already have outputs.
