# Claude baseline: verification and qualitative analysis (R2.11)

This file checks the Claude run described in `README.md` and compares its output with ui2compose v1.2.0 on the ten bundled example screens. No model was called for this analysis. All numbers come from the files in this folder, `results/v1.2.0_examples/`, `results/compile_report.json`, `results/examples_comparison.md`, `results/benchmark_m2.json` and the raw CLI output, which is kept outside the repository. The per-call outputs and the raw CLI output were later lost and are not in this release (see the note at the top of `README.md`). The checks below were run while those files existed and cannot be repeated from the published files.

The ui2compose side is the current v1.2.0 default output: default weights `runs/oversample_5k_appsplit/weights/best.pt` (app-level split) and the thresholds selected on its validation split (`results/eval_appsplit/selected_thresholds.json`; de-duplication 0.4 / 0.85 / 0.5, `containment_ratio` 0.9, `cut_tol_frac` 0, `y_sorted`), as in `results/v1.2.0_examples/` and the harness renders recorded in `results/compile_report.json`. The Claude numbers are unchanged: they are model outputs of 20 October 2026 and do not depend on the ui2compose weights.

## 1. Verification

| Check | Method | Result |
|---|---|---|
| Model | `model` of the init event, of every assistant message, and `modelUsage` of the result event, in the raw stream of all 20 generation calls | `claude-sonnet-5-5` everywhere. The only other `modelUsage` entry is the CLI's session-title request to `claude-haiku-4-5-20251001`. |
| Isolation | `tools`, `mcp_servers`, `skills`, `plugins`, `slash_commands` of every init event | All empty in all 20 calls. No `tool_use` block, `num_turns` = 1, no permission denials, `apiKeySource` = `none` (subscription login), CLI 2.1.267. The working directory was empty before and after every call. |
| Input | The stream does not echo the input, so this check is indirect. Evidence: `user_message()` in the runner builds [image, text]; `message_blocks` = `[image, text]` in every `meta.json`; every `image_sha256` equals the SHA-256 of the matching `examples/*.png`; every call has the same prompt size (3,721 tokens). | Consistent with one unmodified screenshot plus the fixed text per call. |
| Prompts | SHA-256 now, in `prompts.sha256` and in every `meta.json` | Identical (`e66836cb…`, `a0d07518…`). |
| Runner | `runner_sha256` in every `meta.json` against the current file | One value for all 20 calls (`cc30259b…`), equal to the current `scripts/mllm_compare_claude.py`. One `argv`. |
| Call budget | `calls.log` | 21 calls: 1 probe, 1 pilot, 19 batch, numbered 1–21 with no gaps. All exited 0, with no timeout, error or retry. 20 of the 25 allowed calls were used in this round. The published `calls.log` is byte-identical to the original. |
| Integrity | Raw and sanitized stream hashes against `meta.json`; `response.txt` against text rebuilt from the raw stream; `GeneratedScreen.kt` against an independent re-implementation of the extraction rule | All 20 calls match on every point. |
| Privacy | All 133 files under `results/mllm_claude/` (PNGs included) plus the runner. Searched for e-mail patterns, the account address, home and temp paths, the user name, UUID patterns, and the 20 session ids, 184 UUIDs, 20 request ids and 20 message ids taken from the raw streams | No hit in `results/mllm_claude/`. The runner's only hit is its SPDX copyright header, which 25 other source files also carry: authorship, not CLI output. |
| Compilation | 3 files recompiled in a separate copy with `harness/compile_and_render.py --compile-only --offline` | mineworld run 1 fails at `16:43 Cannot access 'val RowColumnParentData?.weight: Float': it is internal in file.`; sign.language run 1 fails at `146:38 None of the following candidates is applicable:`; shazam run 1 compiles with no warnings. Same results and file hashes as `compile_report.json`. Both compile reports and `harness/renders/` are unchanged. |
| Renders | Every input, every ui2compose v1.2.0 render and all 18 Claude renders were inspected | No render is blank or truncated. |

**Verdict.** The run matches the protocol in `README.md`: one model, no tools, fixed prompts, one image and one text block per call, 20 generation calls within budget, and no personal data in the published files. The compile results reproduce.

## 2. Figures

- `side_by_side_all.png` (1209×6790): all ten screens, one per row, with three columns: input, ui2compose v1.2.0 render, Claude run 1 render.
  - The two run-1 compile failures are drawn as a hatched panel with a red frame, the first compiler error and the note "run 2 compiled and rendered".
  - Each row label also gives the result of both Claude runs.
- `side_by_side.png` (1654×1946): four screens as columns, with the same three rows. The four screens were chosen as follows:
  - `com.foxfi #1`: the typical case, where Claude is much closer to the input.
  - `westport.andrewirwin.com.drivesafe #3`: the worked example of the paper (Section 3, Figs. 2 and 3) and a test screen of the app-level split, so the default weights never saw it. ui2compose turns the back arrow into a `Button`, detects the open drawer as one large image and draws the three checkbox rows in a `Box` on top of its placeholder; Claude reproduces the drawer, the dimmed screen and the checked states.
  - `com.netbeast.yeti #1`: the closest case, where ui2compose gets the element inventory and order right.
  - `com.mineworld.swordsaddon #2`: Claude run 1 does not compile, while ui2compose does.
  - No compiled Claude render is further from the input than the ui2compose render, so compilation is the only area where ui2compose does better.
  - One panel in four is a failure, close to the run-1 failure rate of 2 in 10.

Both figures are drawn by `scripts/make_mllm_side_by_side.py` (PIL, DejaVu Sans), which first checks that every ui2compose render belongs to the current `4_GeneratedScreen.kt` (SHA-256 recorded with the render in `results/compile_report.json`), as `scripts/make_figures.py` does for Fig. 4. They show static Paparazzi renders (`DeviceConfig.NEXUS_7`, saved at 625×1000 px), and the inputs are scaled to the same size. The paper's Fig. 4 (`docs/figures/fig4_claude_comparison.png`) shows drivesafe, battlesimulator, foxfi and mineworld from the same renders.

## 3. Per-screen comparison

- **Texts:** visible UI strings transcribed by hand from each input. The status-bar clock, text inside photos, logos and illustrations, and strings cut off beyond reading are excluded.
  - A string counts as recovered when it appears as a Kotlin string literal (case and whitespace normalised).
  - ui2compose writes no texts by design (its README: "Texts, images and states are placeholders"), so its count is 0 on every screen.
- **Emitted:** the number of detections in the ui2compose Kotlin, from `results/examples_comparison.md` (current default weights and thresholds).
- **Split:** the split of the screen on the app-level split the default weights were trained on. Eight of the ten examples are training screens there, so the ui2compose detections on them are in-sample; battlesimulator is a validation and drivesafe a test screen.

| Screen | Split | Content of the input | ui2compose v1.2.0 (emitted; lines) | Claude run 1 | Claude run 2 | Texts run 1 / run 2 | Line similarity, run 1 vs 2 |
|---|---|---|---|---|---|---|---|
| andromo #5 | train | app bar, banner, popup menu, 8-row list | compiles; 15; 187 | compiles (2 warnings) | compiles (2 warnings) | 17/17, 17/17 | 0.51 |
| foxfi #1 | train | dark settings list, 3 sections, 4 checkboxes | compiles; 24; 277 | compiles | compiles | 18/18, 18/18 | 0.49 |
| battlesimulator #3 | val | red toolbar, 9 video rows | compiles; 38; 454 | compiles | compiles | 18/18, 18/18 | 0.63 |
| mcdelivery #1 | train | login form, tab row, logo | compiles; 7; 73 | compiles | compiles | 6/6, 6/6 | 0.38 |
| mineworld #2 | train | toolbar, 3 image cards, bottom tabs | compiles; 12; 138 | **compile error** | compiles | 5/5, 5/5 | 0.48 |
| ball #3 | train | dimmed settings list, radio dialog | compiles; 6; 76 | compiles | compiles | 20/20, 20/20 | 0.38 |
| yeti #1 | train | logo, 2 buttons, link, footer | compiles; 5; 59 | compiles | compiles | 6/6, 6/6 | 0.23 |
| shazam #3 | train | tab bar, illustration, title, button | compiles; 11; 136 | compiles | compiles | 7/7, 7/7 | 0.28 |
| sign.language #2 | train | app bar, popup menu, 10-row list on a photo | compiles; 6; 69 | **compile error** | compiles | 13/15, 15/15 | 0.44 |
| drivesafe #3 | test | drawer with 3 checkboxes over a dimmed screen | compiles; 9; 137 | compiles | compiles | 5/5, 5/5 | 0.58 |

**andromo #5**
- *ui2compose:* 5 of the 8 row icons, 4 of them with title and subtitle, so 4 of the 8 list rows are complete. The "About" popup is detected as a `ListView` and becomes a one-item `LazyColumn` with its text. There is no app bar and no banner. Each row's text column sits far to the right of its icon. Light default theme.
- *Claude:* a black app bar with the title as text, a grey banner with "PRESIDENT TRUMP" as text, the "About" popup in the right place, and all 8 rows verbatim with navy icon placeholders.
  - Run 2 uses shorter rows.
  - Neither run draws a navigation bar.
  - Both runs use the deprecated `Divider`.

**foxfi #1**
- *ui2compose:* all 4 checkboxes and all 18 texts are detected, and the list becomes a `LazyColumn`. Each checkbox sits in a `Row` with its title and subtitle, but the section headers and the entries without a checkbox end up in separate columns, so the list does not read as one sequence of rows. Light theme instead of black.
- *Claude:* both runs are very close to the input: black background, grey section headers, all strings, version text, status and navigation bars.

**battlesimulator #3**
- *ui2compose:* the back arrow becomes a Button, and the search icon a text. All 9 thumbnails and all titles, channel names and durations are detected; each thumbnail becomes a `Box` with its duration text on top, as in the input. The `LazyColumn` holds 8 rows, and its render shows the first 5 (height capped at 565 dp). The cut-off ninth row lies outside the detected list and is drawn as an image box with its text below the list.
- *Claude:* a red toolbar with Material icons, all 9 titles copied character by character, duration badges, and flat colour boxes that approximate the thumbnails. Both runs invent a duration, "12:30", for the cut-off last row.

**mcdelivery #1**
- *ui2compose:* the 2 text fields, the button, the "Forgot password?" link, the "LOGIN" title and one tab label are detected; the logo is detected as a text. No `WebView` is detected with the default weights. The tab row, the tab indicator and the red background are missing.
- *Claude:* a nearly identical layout, including the tab indicator and the yellow SUBMIT button. The logo becomes a yellow letter "M" rather than a placeholder box.

**mineworld #2**
- *ui2compose:* the `RecyclerView` becomes a `LazyColumn` of the 3 cards, each a `Box` with its caption over its image, and the toolbar becomes an image box with "MODS" over it. The two bottom tabs become 2 Buttons, plus one tab label.
- *Claude:* run 1 imports the internal `foundation.layout.weight` and does not compile. Run 2 is faithful: green "MODS" toolbar, 3 cards with gradient placeholders and caption bars in a monospace font, and a MODS/FAVORITES bottom bar.

**ball #3**
- *ui2compose:* the dialog title, its three options as a `LazyColumn` of texts, and "Cancel" as a Button. The dialog frame, the radio buttons, the background list and the dark theme are missing.
- *Claude:* both runs draw the dimmed list and the "Model" dialog with Sphere selected, plus the correct checkbox states.
  - The three rows hidden behind the dialog differ between runs.
  - Run 1 leaves them out.
  - Run 2 invents labels from the visible initials ("Size"/"Change", "Font"/"Set", "Speed"/"Set").

**yeti #1**
- *ui2compose:* the closest case. Logo, 2 buttons, the link and the footer are detected in the right order, but they are stacked at the top with no background colour.
- *Claude:* teal background, a rounded logo shape next to "yeti." as text, correct pill buttons, link, and footer links. Run 2 drops the system bars.

**shazam #3**
- *ui2compose:* all 4 tab icons and 4 labels are detected and grouped as one `Column` per tab inside a `Row`, as in the input. The illustration is not detected. Title, subtitle and the button are detected in order.
- *Claude:* tab bar with Discover selected, the background split, all texts and the button. Both runs redraw the illustration from shapes, although the prompt asks for placeholder boxes.

**sign.language #2**
- *ui2compose:* the Settings/About popup is detected as a `ListView` and becomes a `LazyColumn` of its 2 texts. Of the 10 list rows only 3 icons are detected, with no row text. The app bar and the photo are missing.
- *Claude:* run 1's response contains a draft block with a broken placeholder, then a corrected block. The fixed extraction rule takes the draft.
  - The draft also lacks the popup, hence the 13/15 text count.
  - The corrected block compiles; this was checked outside the published numbers.
  - Run 2 is faithful: app bar, Settings/About popup, 10 rows, and subtitles completed where the photo hides them. It adds an invented 11th row, "Animals", for a sliver of a row at the bottom edge.

**drivesafe #3**
- *ui2compose:* the back arrow becomes a Button next to the "Menu" text. The open drawer is detected as one large image, so the 3 label-plus-checkbox rows are drawn in a `Box` on top of its placeholder; the checkboxes are all unchecked because states are placeholders. The dimming, the car control and the bottom pill are lost.
- *Claude:* both runs reproduce the drawer with the right checked states, the dimmed background, the red "DRIVE SAFE … Off" control, the icons, and the "Call Reply ON" pill.

## 4. Cross-screen findings

**Compilation with no repairs**
- Claude: 18 of 20 files (90 %). 8 of 10 screens compiled in both runs and 10 of 10 in at least one run.
- ui2compose: 10 of 10.
- The two Claude failures are of different kinds:
  - an API error: importing a scope-internal extension;
  - a response-format error: a draft followed by a correction, which no fixed extraction rule can resolve.

**Visual fidelity**
- On every screen where Claude's code compiled, its render is much closer to the input than the ui2compose render. It reproduces:
  - theme and colours;
  - typography;
  - widget states (checked boxes, selected radio button, selected tab);
  - overlays (dialog, popup menus, drawer).
- ui2compose renders a light Material 3 skeleton with placeholders.
  - Its inventory is right on simple screens: yeti, the foxfi checkboxes, the shazam tab bar (now one `Column` per tab).
  - It misses list rows (andromo 4 of 8 complete, sign.language 3 icons of 10 rows) and loses most overlays; the popup menus of andromo and sign.language are kept only as small lists of their texts.
  - It misclassifies some elements: back arrows and tabs as Buttons, popup menus as `ListView`s, the drivesafe drawer as an image, the mcdelivery logo as a text.

**Text recovery**
- Claude: 115 of 117 strings in run 1 and 117 of 117 in run 2.
- The two misses are in the draft that did not compile, so the 18 compiled files hold every string (214 of 214).
- ui2compose emits no text, by design.

**Invented content**
- Claude added strings that cannot be read in the input in 4 of 20 files:
  - "12:30" in both battlesimulator runs;
  - three row labels behind the dialog in ball run 2;
  - the row "Animals" in sign.language run 2.
- It drew a status bar with the clock in 19 of 20 files and a navigation bar in 17 of 20. Both are system UI and would have to be removed in an app.
- It drew illustrations and logos from shapes and letters in shazam, mcdelivery and yeti, instead of placeholder boxes.
- ui2compose invents no content, but turns detector errors into components: Buttons for arrows and tabs, an image placeholder for a drawer, a list for a popup menu.

**Run-to-run variation**
- No pair of Claude runs is identical, byte for byte or after removing whitespace.
- Mean difflib similarity is 0.44 over lines (0.23–0.63) and 0.64 over characters.
- The visual differences are smaller than the code differences: row heights, logo treatment, system bars, hidden rows.
- ui2compose is deterministic for fixed weights and settings: `tests/test_golden.py` checks that replaying the saved detections reproduces the published Kotlin exactly.

**Latency**
- Claude: `duration_ms` mean 21.2 s per call, median 20.9 s, range 10.2–32.8 s.
  - Mean wall time is 22.6 s, and the 20 calls took 7.1 min in total.
  - This includes CLI start-up, network and server time.
- ui2compose: `ui2compose-benchmark --images examples --model runs/oversample_5k_appsplit/weights/best.pt --repeats 5` on an Apple M2 laptop CPU (8 cores, 16 GB, no GPU, macOS), 27 October 2026, saved as `results/benchmark_m2.json`. The load average was 4.10 / 7.29 / 8.58 (1 / 5 / 15 min) at the start and 4.02 / 7.17 / 8.52 at the end of the run, on 8 cores (`host.load_average_1_5_15min`).
  - Warm end-to-end median: 0.106 s per screenshot (IQR 0.101–0.115 s, n = 49), almost all of it detection (median 0.105 s; de-duplication, graph and code generation together under 1 ms).
  - A cold start takes about 1.2 s: imports 0.63 s (median of 5 fresh subprocesses), loading the model 0.46 s, and the first detection 0.14 s.
- A warm ui2compose call is about 200 times faster than a Claude call (medians 20.9 s and 0.106 s).

**Data leaving the machine**
- Claude: every screenshot (base64 PNG) and the prompt go to Anthropic's servers.
  - They travel with the CLI preamble, which holds an identity line, the date and the account e-mail.
  - Each call also sends a second request to `claude-haiku-4-5-20251001` for a session title (1,058 input tokens). Whether that request carries the image was not verified.
- ui2compose runs offline once the weights are installed.

**Quota and cost**
- Subscription login, so there is no per-token bill.
- All rate-limit events report status `allowed` in the five-hour window, with no overage.
- Per call: 3,721 prompt tokens and a mean of 3,509 output tokens (range 1,787–5,840).
- Thinking tokens are part of the output: mean 595 (range 0–1,505). The thinking itself is not returned.
- The CLI reports a notional USD 2.17 for the 20 calls (2.15 Sonnet, 0.02 Haiku). It marks the price basis for `claude-sonnet-5-5` as unknown, so this figure is not reliable.
- ui2compose has no per-call cost.

**Code size:** Claude files average 209 lines; ui2compose files average 161 (range 59–454).

## 5. Caveats

- Only ten screens, chosen as bundled examples rather than as a sample, so the findings are qualitative.
- Eight of the ten screens are training screens of the app-level split on which the default ui2compose weights were trained (battlesimulator is a validation and drivesafe a test screen), so the ui2compose detections on them are in-sample. Detection quality on unseen apps is measured in `results/eval_appsplit/`.
- The screens are public ReDraw images and may be in the model's training data.
- One model (`claude-sonnet-5-5`, not the newest Opus, which CLI 2.1.267 does not accept), one prompt, and two runs per screen. Sampling is at CLI defaults: temperature cannot be set, and extended thinking is on by default.
- The CLI adds a short fixed preamble (identity line, date, account metadata) that a subscription login cannot remove. Its size after setting `CLAUDE_CODE_ATTRIBUTION_HEADER=0` and `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1` was not measured.
- The renders are static layoutlib renders (Paparazzi, `NEXUS_7`), not device screenshots. Fonts differ from the device, and a `LazyColumn` shows only its first items.
- The comparison is not like for like: ui2compose outputs a skeleton with placeholders by design, while Claude is asked to reproduce the full screen.
- Claude was run on 20 October 2026, before the default ui2compose weights changed to the app-split weights. The ui2compose columns, the figures in this folder and the paper's Fig. 4 all use the current default output; the Claude outputs are the same files throughout.
- The text count relies on a manual transcription by one person.

## 6. Draft text (for the authors to edit)

### (a) Manuscript, Section 3.1 (119 words)

> As a qualitative baseline, each of the ten example screenshots was sent twice to claude-sonnet-5-5 through the Claude Code CLI 2.1.267 on 20 October 2026, with one fixed prompt, no tools and a short fixed CLI preamble (identity line, date, account metadata), and the returned Kotlin was compiled unchanged with our harness. 18 of the 20 files compiled and rendered, against 10 of 10 for ui2compose. The compiled files contained all 214 visible UI strings and reproduced colours, overlays and widget states, which ui2compose leaves as placeholders. No two runs gave the same code, four files contained invented strings, and a call took 21 s on average against 0.11 s for a warm ui2compose run on a laptop CPU.

### (b) Response to Reviewer 2, comment R2.11 (173 words)

> We added the requested comparison (Section 3.1, Fig. N; data and protocol in `results/mllm_claude/`). Each of the ten example screenshots was sent twice to Claude Sonnet 5.5 (`claude-sonnet-5-5`) through the Claude Code CLI 2.1.267, with a fixed prompt and no tools, and the Kotlin was compiled and rendered unchanged with the same harness. The LLM output is visually much closer to the input: it reproduced all visible strings in the compiled files, the colours, dialogs, menus and checkbox states, whereas ui2compose emits placeholders by design. ui2compose compiled 10 of 10 screens against 18 of 20 LLM files, gives the same output on every run (no two LLM runs were identical), takes about 0.11 s per screenshot on a laptop CPU (about 1.2 s cold) instead of about 21 s per remote call, and sends no data off the machine. The LLM also invented strings for hidden content in 4 of 20 files. The comparison is limited to ten public screens that may be in the model's training data, one model and one prompt.
