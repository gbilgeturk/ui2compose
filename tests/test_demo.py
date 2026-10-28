# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the Streamlit demo (ui2compose.webapp.demo) without Streamlit.

The helper functions are tested directly (they need only Pillow). The page
function main() is run once against a recording stand-in for the streamlit
module and a detector that replays the golden detections of an example; this checks
that main() runs from top to bottom and shows the pipeline's results, but not
Streamlit's rendering (the stand-in implements only the call surface the demo
uses). The JavaScript of the Compose preview is executed with Node.js when it
is installed.
"""

import copy
import inspect
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image

from ui2compose.infer.build_ui_graph import UIGraphBuilder
from ui2compose.infer.generate_compose_code import ComposeCodeGenerator, build_layout_tree
from ui2compose.infer.pipeline import run_pipeline
from ui2compose.webapp import demo

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / 'tests' / 'golden'
# A bundled example whose v1.2.0 output (default weights) has de-duplication removals, a
# containment Box and a LazyColumn.
SCREEN = 'com.luigiedeveloper.battlesimulator-screens_screenshot_3'


def _golden(stem: str, name: str) -> Any:
    """Reads a golden JSON file.

    Input:  stem — example stem; name — file name in tests/golden/<stem>/
    Output: parsed JSON
    """
    with open(GOLDEN_DIR / stem / name, 'r', encoding='utf-8') as f:
        return json.load(f)


class ReplayDetector:
    """Returns the golden raw detections of the screen named by the image path; writes a stub figure."""

    def __init__(self):
        """Initialises the call log.

        Input:  none
        Output: none
        """
        self.visualize_calls: List[Dict] = []

    def detect(self, image_path: str, **kwargs) -> List[Dict]:
        """Returns the golden raw detections of the screen.

        Input:  image_path — path of an example screenshot; kwargs — detector settings
        Output: deep copy of tests/golden/<stem>/1_detections_raw.json
        """
        return copy.deepcopy(_golden(Path(image_path).stem, '1_detections_raw.json'))

    def visualize(self, image_path: str, detections: List[Dict], output_path: str = None,
                  removed: List[Dict] = None):
        """Records the call and writes a small placeholder PNG to output_path.

        Input:  as ComponentDetector.visualize
        Output: None
        """
        self.visualize_calls.append({'kept': [d['id'] for d in detections],
                                     'removed': [d['id'] for d in removed or []]})
        if output_path:
            Image.new('RGB', (4, 4), 'white').save(output_path)


def _colours(image: Image.Image) -> set:
    """Returns the set of colours that occur in an image.

    Input:  image — PIL image
    Output: set of pixel values
    """
    return {colour for _, colour in image.getcolors(maxcolors=image.width * image.height)}


def _two_rows_boxes() -> List[Dict]:
    """Builds a plain box list: two boxes side by side above a wide box that contains a small one.

    Input:  none
    Output: list of box dictionaries (600 x 400 image)
    """
    return [
        {'id': 0, 'class_name': 'ImageView', 'bbox': [20, 20, 120, 120]},
        {'id': 1, 'class_name': 'TextView', 'bbox': [200, 40, 400, 100]},
        {'id': 2, 'class_name': 'ImageView', 'bbox': [20, 200, 580, 380]},
        {'id': 3, 'class_name': 'Button', 'bbox': [250, 250, 350, 300]},
    ]


# ── Import surface and defaults ─────────────────────────────


def test_demo_imports_without_streamlit_opencv_matplotlib_or_torch():
    """Importing the demo module loads none of streamlit, cv2, matplotlib, torch, ultralytics."""
    code = ("import sys, ui2compose.webapp.demo\n"
            "print([m for m in ('streamlit', 'cv2', 'matplotlib', 'torch', 'ultralytics') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]'


def test_demo_defaults_equal_pipeline_defaults():
    """The sidebar defaults are the defaults of run_pipeline, so the demo shows the CLI result."""
    params = inspect.signature(run_pipeline).parameters
    for key, value in demo.DEMO_DEFAULTS.items():
        assert params[key].default == value, key


def test_no_legacy_card_or_cardview_colours():
    """Only the four layout container types are coloured; the legacy Card/CardView entries are gone."""
    assert set(demo.LAYOUT_COLORS) == {'Column', 'Row', 'Box', 'LazyColumn'}
    source = inspect.getsource(demo)
    assert 'CardView' not in source and "'Card'" not in source and 'Toolbar' not in source


# ── Layout overlay ──────────────────────────────────────────


def test_layout_overlay_draws_the_layout_tree_containers():
    """The overlay draws a frame in the type colour for every container of the layout tree."""
    tree = build_layout_tree(_two_rows_boxes(), image_size=(600, 400))
    types_in_tree = {n['type'] for n, _ in demo.walk_layout(tree)}
    assert {'Column', 'Row'} <= types_in_tree
    image = Image.new('RGB', (600, 400), 'white')
    before = image.tobytes()
    out = demo.draw_layout_overlay(image, tree)

    assert out.size == image.size and out.mode == 'RGB'
    assert image.tobytes() == before                        # input not modified
    colours = _colours(out)
    for ctype in types_in_tree - {'leaf'}:
        assert demo.LAYOUT_COLORS[ctype] in colours, ctype   # opaque frame outline
    # the Row frame (members 0 and 1) is inflated by one step around their union box
    row = next(n for n, _ in demo.walk_layout(tree) if n['type'] == 'Row')
    step = max(2, round(demo.LAYOUT_PAD_STEP * 600 / demo.LAYOUT_REF_WIDTH))
    x = int(row['bbox'][0]) - step
    assert out.getpixel((x, int((row['bbox'][1] + row['bbox'][3]) / 2))) == demo.LAYOUT_COLORS['Row']


def test_layout_overlay_of_a_box_and_counts():
    """A container with a child becomes a Box frame; layout_container_counts counts the types."""
    graph = UIGraphBuilder().build_graph(_two_rows_boxes(), image_size=(600, 400))
    tree = build_layout_tree(graph)
    counts = demo.layout_container_counts(tree)
    assert counts.get('Box') == 1
    out = demo.draw_layout_overlay(Image.new('RGB', (600, 400), 'white'), tree)
    assert demo.LAYOUT_COLORS['Box'] in _colours(out)


# ── Graph views and text views ──────────────────────────────


def test_adjacency_matrix_maps_ids_with_gaps_to_positions():
    """With ids 0, 3, 7 (gaps after de-duplication) every edge lands in the right cell."""
    graph = {'nodes': [{'id': 0, 'class_name': 'TextView'}, {'id': 3, 'class_name': 'ImageView'},
                       {'id': 7, 'class_name': 'Button'}],
             'edges': [{'source': 7, 'target': 0, 'relation': 'above'},
                       {'source': 3, 'target': 7, 'relation': 'parent_child'},
                       {'source': 9, 'target': 0, 'relation': 'below'}]}
    matrix, labels = demo.adjacency_matrix(graph)
    assert labels == ['0:TextV', '3:Image', '7:Butto']
    assert matrix == [[0, 0, 0], [0, 0, demo.EDGE_CODES['parent_child']],
                      [demo.EDGE_CODES['above'], 0, 0]]


def test_graph_figures_are_built():
    """The node-edge and adjacency figures are created with matplotlib's object API."""
    pytest.importorskip('matplotlib')
    graph = UIGraphBuilder().build_graph(_two_rows_boxes(), image_size=(600, 400))
    fig = demo.plot_graph(graph, 600, 400, {'ImageView': '#D62728'})
    assert len(fig.axes[0].patches) == 4
    fig = demo.plot_adjacency(graph)
    assert fig.axes[0].images[0].get_array().shape == (4, 4)


def test_text_views_of_containment_and_layout():
    """The containment forest and the layout tree are rendered as indented text with their rules."""
    graph = UIGraphBuilder().build_graph(_two_rows_boxes(), image_size=(600, 400))
    containment = demo.format_containment(graph)
    assert 'ImageView#2 (1.00)' in containment and '  └─ Button#3 (1.00)' in containment
    assert 'Synthetic' not in containment
    layout = demo.format_layout_tree(build_layout_tree(graph))
    assert 'Box [rule=containment, container ImageView#2]' in layout
    assert 'Row [rule=gap, axis=x, 2 children]' in layout


def test_explain_options_label_kept_removed_and_filtered():
    """Every raw id is offered, labelled kept, removed by a de-duplication rule, or removed by class_conf."""
    raw = [{'id': i, 'class_name': 'TextView', 'confidence': c} for i, c in [(0, 0.9), (1, 0.5), (2, 0.2)]]
    kept = [raw[0]]
    removed = [dict(raw[1], dedup_rule='same_class_iou', removed_by=0, overlap=0.8)]
    assert demo.explain_options(raw, kept, removed) == [
        (0, '0 · TextView 0.90 · kept'),
        (1, '1 · TextView 0.50 · removed (same_class_iou)'),
        (2, '2 · TextView 0.20 · removed (class_conf)')]
    assert demo.removed_rows(removed) == [{'id': 1, 'class': 'TextView', 'confidence': 0.5,
                                          'rule': 'same_class_iou', 'overlap': 0.8, 'removed by id': 0}]


# ── HTML previews ───────────────────────────────────────────


def test_preview_has_a_case_for_every_code_template():
    """Each of the 13 template classes has its own preview, as does the 'Unknown' placeholder."""
    template = demo._PREVIEW_TEMPLATE
    for cls in ComposeCodeGenerator.COMPONENT_MAP:
        assert f"case '{cls}'" in template, cls
    assert "Unknown: " in template
    for ctype in ('Box', 'LazyColumn', 'Row'):
        assert f"node.type === '{ctype}'" in template


def test_preview_and_bbox_html_embed_escaped_data():
    """The data is embedded as JSON in which '</' cannot close the script element."""
    boxes = [{'id': 0, 'class_name': '</script><b>', 'bbox': [0, 0, 10, 10]}]
    tree = build_layout_tree(boxes, image_size=(100, 200))
    html = demo.build_layout_preview_html(tree, 100, 200)
    assert html.count('</script>') == 1 and '<\\/script>' in html
    html = demo.build_bbox_html(boxes, 100, 200, {'TextView': '#1F77B4'})
    assert html.count('</script>') == 1 and '"aspect": 0.5' in html


@pytest.mark.skipif(shutil.which('node') is None, reason="Node.js not installed")
def test_preview_javascript_renders_the_example_layout(tmp_path):
    """Executed with Node.js, the preview script renders the Box, the LazyColumn and every leaf of the example tree."""
    tree = _golden(SCREEN, '2_layout_tree.json')
    html = demo.build_layout_preview_html(tree, 1200, 1920)
    script = html.split('<script>')[1].split('</script>')[0]
    harness = ("const els = {};\n"
               "global.document = {getElementById: function(id){ if(!els[id]) els[id] = {style: {}, "
               "innerHTML: ''}; return els[id]; }};\n"
               + script + "\nprocess.stdout.write(els['screen'].innerHTML);\n")
    path = tmp_path / 'preview.js'
    path.write_text(harness, encoding='utf-8')
    out = subprocess.run(['node', str(path)], capture_output=True, text=True, check=True).stdout
    assert 'Box · ImageView#4' in out
    assert 'LazyColumn · ListView#3' in out
    n_leaves = sum(1 for n, _ in demo.walk_layout(tree) if n['type'] == 'leaf')
    assert out.count('class="cell"') == n_leaves
    assert out.count('Text Content') == 27 and out.count('Image placeholder') == 9


@pytest.mark.skipif(shutil.which('node') is None, reason="Node.js not installed")
def test_preview_box_in_a_row_grows_like_the_kotlin(tmp_path):
    """A containment Box in a Row: flex:1 in the preview as weight(1f) in the Kotlin; fixed size with sizes from boxes."""
    from ui2compose.infer.build_ui_graph import UIGraphBuilder
    boxes = [{'id': 0, 'class_name': 'ImageView', 'bbox': [0, 0, 500, 400]},
             {'id': 1, 'class_name': 'TextView', 'bbox': [20, 20, 200, 60]},
             {'id': 2, 'class_name': 'TextView', 'bbox': [20, 300, 200, 340]},
             {'id': 3, 'class_name': 'EditText', 'bbox': [600, 100, 1100, 200]}]
    tree = build_layout_tree(UIGraphBuilder().build_graph(boxes, image_size=(1200, 1920)))
    assert tree['type'] == 'Row' and [c['type'] for c in tree['children']] == ['Box', 'leaf']

    def render(size_from_boxes: bool) -> str:
        """Runs the preview script with Node.js and returns the rendered HTML.

        Input:  size_from_boxes — preview option
        Output: innerHTML of the screen element
        """
        html = demo.build_layout_preview_html(tree, 1200, 1920, size_from_boxes=size_from_boxes)
        script = html.split('<script>')[1].split('</script>')[0]
        harness = ("const els = {};\n"
                   "global.document = {getElementById: function(id){ if(!els[id]) els[id] = {style: {}, "
                   "innerHTML: ''}; return els[id]; }};\n"
                   + script + "\nprocess.stdout.write(els['screen'].innerHTML);\n")
        path = tmp_path / f'preview_{size_from_boxes}.js'
        path.write_text(harness, encoding='utf-8')
        return subprocess.run(['node', str(path)], capture_output=True, text=True, check=True).stdout

    box_div = re.compile(r'<div class="c box" style="([^"]*)">')
    assert box_div.findall(render(False)) == ['flex:1;min-width:0;']
    (style,) = box_div.findall(render(True))
    assert 'flex' not in style and style.startswith('width:') and 'height:' in style


# ── Pipeline call and page ──────────────────────────────────


def test_run_demo_pipeline_gives_the_command_line_result(tmp_path):
    """The demo runs run_pipeline: on the golden detections of the example it yields the golden Kotlin and figure call."""
    detector = ReplayDetector()
    res, log = demo.run_demo_pipeline(str(REPO_ROOT / 'examples' / f'{SCREEN}.png'), 'w.pt',
                                      str(REPO_ROOT / 'configs' / 'dataset.yaml'), detector,
                                      dict(demo.DEMO_DEFAULTS), str(tmp_path))
    assert res['code'] == (GOLDEN_DIR / SCREEN / '4_GeneratedScreen.kt').read_text(encoding='utf-8')
    assert detector.visualize_calls == [{'kept': [d['id'] for d in res['detections']],
                                         'removed': [6, 21, 37]}]
    assert (tmp_path / '1_detections_viz.png').exists()
    assert 'Pipeline Completed' in log


class _StopPage(Exception):
    """Raised by the stand-in for st.stop()."""


class _Recorder:
    """Records calls: any attribute is a callable that logs itself and returns a new recorder."""

    def __init__(self, log: List, name: str = 'st'):
        """Stores the shared call log.

        Input:  log — list receiving (name, args, kwargs); name — element name
        Output: none
        """
        self._log, self._name = log, name

    def __getattr__(self, name: str):
        """Returns a recording callable for any element or method name.

        Input:  name — attribute name
        Output: callable that logs (name, args, kwargs) and returns a _Recorder
        """
        if name.startswith('__'):
            raise AttributeError(name)

        def call(*args, **kwargs):
            """Logs the call.

            Input:  any arguments
            Output: a new _Recorder
            """
            self._log.append((name, args, kwargs))
            return _Recorder(self._log, name)
        return call

    def __enter__(self):
        """Context-manager entry (for `with st.sidebar:`, columns, tabs, status, expander).

        Input:  none
        Output: self
        """
        return self

    def __exit__(self, *exc):
        """Context-manager exit; does not suppress exceptions.

        Input:  exception info
        Output: False
        """
        return False


class _FakeStreamlit(types.ModuleType):
    """Stand-in for the streamlit module with the widgets' default answers."""

    def __init__(self, answers: Dict[str, Any]):
        """Prepares the call log, session state and widget answers.

        Input:  answers — widget label -> returned value (others return their default)
        Output: none
        """
        super().__init__('streamlit')
        self.log: List = []
        self.answers = answers
        self.session_state: Dict = {}
        self.sidebar = _Recorder(self.log, 'sidebar')

    def __getattr__(self, name: str):
        """Delegates every other element to a recorder.

        Input:  name — attribute name
        Output: recording callable
        """
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(_Recorder(self.log), name)

    def stop(self):
        """Stops the page like st.stop().

        Input:  none
        Output: never returns (raises _StopPage)
        """
        raise _StopPage()

    def cache_resource(self, func):
        """Returns the function unchanged (no caching).

        Input:  func — function to cache
        Output: func
        """
        return func

    def columns(self, spec):
        """Returns one recorder per column.

        Input:  spec — number of columns or list of widths
        Output: list of recorders
        """
        return [_Recorder(self.log, 'column') for _ in range(spec if isinstance(spec, int) else len(spec))]

    def tabs(self, labels):
        """Returns one recorder per tab.

        Input:  labels — tab labels
        Output: list of recorders
        """
        self.log.append(('tabs', (labels,), {}))
        return [_Recorder(self.log, 'tab') for _ in labels]

    def _answer(self, label, default):
        """Returns the configured answer of a widget or its default.

        Input:  label — widget label; default — the widget's default value
        Output: answer
        """
        value = self.answers.get(label, default)
        self.log.append(('widget', (label, value), {}))
        return value

    def selectbox(self, label, options, index=0, **kwargs):
        """Returns the configured or the default option.

        Input:  label, options, index — as st.selectbox
        Output: selected option
        """
        return self._answer(label, list(options)[index])

    def radio(self, label, options, index=0, **kwargs):
        """Returns the configured or the default option.

        Input:  label, options, index — as st.radio
        Output: selected option
        """
        return self._answer(label, list(options)[index])

    def slider(self, label, min_value, max_value, value, step=None, **kwargs):
        """Returns the configured or the default value.

        Input:  as st.slider
        Output: value
        """
        return self._answer(label, value)

    def checkbox(self, label, value=False, **kwargs):
        """Returns the configured or the default value.

        Input:  as st.checkbox
        Output: bool
        """
        return self._answer(label, value)

    def button(self, label, **kwargs):
        """Returns True (every button is pressed).

        Input:  as st.button
        Output: True
        """
        return self._answer(label, True)

    def file_uploader(self, label, **kwargs):
        """Returns None (nothing uploaded).

        Input:  as st.file_uploader
        Output: None
        """
        return self._answer(label, None)


def test_page_runs_end_to_end_with_a_stand_in_streamlit(monkeypatch):
    """main() runs top to bottom on a bundled example and shows counts, explain lines, code and previews."""
    pytest.importorskip('matplotlib')
    pytest.importorskip('cv2')              # class colours come from detect_components
    fake = _FakeStreamlit({'…or pick a bundled example (ReDraw screen)': f'{SCREEN}.png'})
    components_pkg = types.ModuleType('streamlit.components')
    components_v1 = types.ModuleType('streamlit.components.v1')
    html_calls: List = []
    components_v1.html = lambda html, height=None, **kw: html_calls.append((html, height))
    components_pkg.v1 = components_v1
    fake.components = components_pkg
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    monkeypatch.setitem(sys.modules, 'streamlit.components', components_pkg)
    monkeypatch.setitem(sys.modules, 'streamlit.components.v1', components_v1)
    monkeypatch.setattr(demo, 'find_models',
                        lambda: [demo.PROJECT_ROOT / 'runs' / 'oversample_5k' / 'weights' / 'best.pt',
                                 demo.PROJECT_ROOT / 'runs' / 'oversample_5k_appsplit' / 'weights' / 'best.pt'])
    monkeypatch.setattr(demo, '_load_detector', lambda model_path, dataset_yaml: ReplayDetector())

    demo.main()

    # the sidebar preselects the pipeline's default weights, not the first model found
    assert ('widget', ('Model', 1), {}) in fake.log
    metrics = {args[0]: args[1] for name, args, _ in fake.log if name == 'metric'}
    assert metrics == {'Raw detections': 41, 'Removed by de-duplication': 3,
                       'Kept (in the code)': 38, 'Kotlin lines': 454}
    codes = [args[0] for name, args, _ in fake.log if name == 'code']
    golden_code = (GOLDEN_DIR / SCREEN / '4_GeneratedScreen.kt').read_text(encoding='utf-8')
    assert golden_code in codes
    assert any(c.startswith('Explain trace for detection id 0') for c in codes)
    assert any('Box [rule=containment, container ImageView#4]' in c for c in codes)
    assert len(html_calls) == 2 and 'LazyColumn' in html_calls[0][0]
    tabs = next(args[0] for name, args, _ in fake.log if name == 'tabs')
    assert tabs == ["Detection", "Graph", "Hierarchy", "Explain", "Compose Code", "Compose Preview"]
    assert sum(1 for name, _, _ in fake.log if name == 'pyplot') == 2
