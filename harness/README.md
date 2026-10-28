# Kotlin compile + render harness

This harness checks that the Jetpack Compose code written by ui2compose
(`4_GeneratedScreen.kt`) compiles against a fixed Android toolchain, and
renders it off-device with [Paparazzi](https://github.com/cashapp/paparazzi).
It needs no emulator or device. Rendering composes, measures, lays out and
draws each screen. It does not measure how close the render is to the
original screenshot.

## Toolchain (versions actually used)

| Component | Version | Declared in |
|---|---|---|
| Gradle (wrapper) | 8.9, distribution SHA-256 pinned | `android/gradle/wrapper/gradle-wrapper.properties` |
| Android Gradle Plugin | 8.5.2 | `android/gradle/libs.versions.toml` |
| Kotlin (`org.jetbrains.kotlin.android`, `org.jetbrains.kotlin.plugin.compose`) | 2.0.21 | `android/gradle/libs.versions.toml` |
| Compose BOM | 2024.06.00. It resolves to compose ui, foundation and runtime 1.6.8 and material3 1.2.1 | `android/gradle/libs.versions.toml` |
| Paparazzi | 1.3.4 (layoutlib 2023.2.1-6c7316c, Android 14) | `android/gradle/libs.versions.toml` |
| JUnit | 4.13.2 | `android/gradle/libs.versions.toml` |
| JDK | 17 (17.0.13, Homebrew `openjdk@17`) | `JAVA_HOME` |
| compileSdk / minSdk | 34 / 24 | `android/screens/build.gradle.kts` |
| Snapshot device | Paparazzi `DeviceConfig.NEXUS_7`: 1200 x 1920 px, xhdpi (600 x 960 dp), the size of the ReDraw example screenshots | `compile_and_render.py` |

The resolved Compose versions and the JDK version are also written to the
report on every run (`toolchain.resolved`, `toolchain.jdk`). The emitter
(`src/ui2compose/infer/generate_compose_code.py`) targets this toolchain:
Kotlin 2.0.21 with Compose BOM 2024.06.00.

## Requirements

- JDK 17. The harness uses `JAVA_HOME`; if it is unset, it falls back to
  `/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home`, or you can
  pass `--java-home`.
- Android SDK with `platforms/android-34`. The harness uses `ANDROID_HOME` or
  `ANDROID_SDK_ROOT`, falling back to `~/Library/Android/sdk`, or you can
  pass `--sdk-dir`. It writes `android/local.properties` itself.
- Python 3.10 or newer, standard library only.
- Network access on the first run, to fetch the Gradle 8.9 distribution and
  the Maven artifacts from `google()` and `mavenCentral()`. Later runs can
  pass `--offline`.

## How to run

From the repository root:

```bash
# Published report: v1.2.0 example screens + codegen fixtures.
# Writes results/compile_report.json, results/v1.2.0_examples/<stem>/render.png
# and harness/renders/codegen__<fixture>.png
harness/compile_and_render.sh results/v1.2.0_examples tests/fixtures/codegen

# Regression reference: the v1.1.0 outputs, rendered outside the repository
harness/compile_and_render.sh --no-copy --render-dir /tmp/v110_renders \
    --report /tmp/compile_report_v110.json results/v1.1.0_examples
```

Each argument is a `.kt` file or a directory that is searched recursively for
`*.kt`. Options:

| Option | Effect |
|---|---|
| `--report FILE` | Report path (default `results/compile_report.json`). |
| `--render-dir DIR` | Where renders go when they cannot be placed next to their source (default `harness/renders`). |
| `--no-copy` | Never write `render.png` next to a source; send every render to `--render-dir`. |
| `--compile-only` | Skip Paparazzi. Every entry gets `rendered: false`, existing render files are left untouched, and the exit status depends on compilation only. |
| `--offline` | Pass `--offline` to Gradle. |
| `--java-home`, `--sdk-dir` | JDK and SDK locations. |

Exit status: 0 when every file compiled and (unless `--compile-only`) every
screen rendered, 1 when any failed, 2 for a usage or setup error. Gradle logs are kept in
`android/build/harness-logs/`.

## What it does

1. **Packages.** Each input file is assigned a package in sorted path order:
   `generated.s01`, `generated.s02`, and so on. The harness copies the file
   into `android/screens/src/main/java/generated/sNN/`. In the copy, it
   inserts a `package` line (or replaces an existing one), so the identical
   `GeneratedScreen()` functions do not clash. The source files are not
   modified.
2. **Snapshot tests.** For each file that defines `fun GeneratedScreen()`,
   the harness writes one Paparazzi test. The test renders
   `MaterialTheme { Surface(Modifier.fillMaxSize()) { GeneratedScreen() } }`,
   the default Material 3 light theme, because the generated code has no theme
   of its own.
3. **Compile.** The harness runs `:screens:compileDebugKotlin`. Compilation
   is non-incremental and forced with `--rerun`, so the diagnostics always
   cover every file.
   - Compiler errors and warnings are mapped back to the input files.
   - If the build fails, files with errors are left out and the rest are
     compiled again, for at most 5 rounds.
4. **Render.** The harness runs `:screens:recordPaparazziDebug`. A screen
   counts as rendered when its test passed and its PNG exists. Paparazzi
   stores snapshots scaled to at most 1000 px on the longer side, so the
   1200 x 1920 frame becomes 625 x 1000.
5. **Place renders.** A render becomes `render.png` next to its source when
   the source is the only `.kt` file in its directory (the pipeline layout
   `<stem>/4_GeneratedScreen.kt`). Otherwise it goes to
   `<render-dir>/<dir>__<stem>.png`. If a source does not render, any old
   render at its target path is deleted.
6. **Report.** The harness writes a JSON report with:
   - the toolchain;
   - the snapshot set-up;
   - a summary;
   - one entry per file: its source and SHA-256, package, compile result,
     compiler errors and warnings, and render result, path and size.

The module's generated sources (`android/screens/src/`), the Gradle state and
the build directories are ignored by git.

## Limits

- Paparazzi provides `LocalInspectionMode.current == true`. The WebView
  template therefore renders its placeholder branch: a box labelled
  "WebView". The `AndroidView(WebView)` branch is compiled but not rendered.
- Indeterminate progress indicators are captured at their first animation
  frame, so a `CircularProgressIndicator` shows as a small arc.
  `DropdownMenu` is captured closed.
- The render shows the window only: content below 960 dp is cut off, and
  `LazyColumn` content is cut off at its height bound.

## Results (19 October 2026)

| Inputs | Compiled | Rendered | Compiler warnings |
|---|---|---|---|
| `results/v1.2.0_examples/*/4_GeneratedScreen.kt` (10) | 10 | 10 | 0 |
| `tests/fixtures/codegen/*.kt` (6, including the all-templates fixture in three modes) | 6 | 6 | 0 |
| `results/v1.1.0_examples/*/4_GeneratedScreen.kt` (10, regression reference, renders not stored) | 10 | 10 | 0 |

No generator template had to be changed for the code to compile.

`tests/fixtures/codegen/all_templates_graph.json` uses every template: all 13
detector classes, an unknown class, circular and linear progress bars,
full-width templates inside a Row (`weight(1f)`), a `LazyColumn` with items
(a Row item and a containment Box item), a `LazyColumn` without items inside
a Row, and a containment Box. Its Kotlin is checked in three modes:
`all_templates_xycut.kt` (defaults), `all_templates_sized.kt`
(`size_from_boxes=True`, trace comments) and `all_templates_flat.kt`
(`layout='flat'`).

**Known layout issue visible in the renders (not fixed here).** A
containment Box that is a direct child of a Row gets no `weight(1f)`. When
the Box's inner layout fills the width (a Row or Column with
`fillMaxWidth()`), the Box takes the whole Row, and its weighted siblings
are measured with zero width. Two cases:
- In `renders/codegen__all_templates_xycut.png`, WebView#13 is not visible
  next to the ImageView#14 Box. In a scratch render, adding `weight(1f)` to
  that Box made it appear.
- In `com.luigiedeveloper.battlesimulator #3`, the list item with the
  ImageView#28 Box has the same structure: a Box with an inner Row, next to a
  weighted Column. That item lies below the visible part of the bounded
  `LazyColumn`, so the render does not show it.

## Tests

`tests/test_harness.py` runs in CI without Gradle. It tests:
- the driver logic, with Gradle replaced by fakes;
- the all-templates goldens;
- that `results/compile_report.json` is current. The report must list
  exactly the example screens and the codegen fixtures, with their current
  SHA-256, all compiled and rendered.

After you regenerate any of these `.kt` files, re-run the first command
above. `scripts/run_examples.py` replaces only the pipeline's own files in
`results/v1.2.0_examples/<stem>/` and keeps `render.png`; when it changes the
Kotlin of a screen it names the screen, and the render (and the report) must
then be refreshed with the first command. CI itself does not run Gradle.
