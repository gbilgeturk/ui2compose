# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests of the Kotlin emitter (ComposeCodeGenerator) and its command line.

Pure tests (no torch / ultralytics).  The golden files under
tests/fixtures/codegen/ were generated with the calls listed in GOLDEN and
checked by hand before they were frozen; to regenerate one after an
intentional change, write ``generate(...) + "\\n"`` for the same arguments to
the file and review the diff.
"""

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from _reference_data import require_reference_data
from ui2compose.infer.generate_compose_code import (
    ComposeCodeGenerator,
    build_layout_tree,
    main,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / 'tests' / 'fixtures' / 'codegen'
V110_EXAMPLES = REPO / 'results' / 'v1.1.0_examples'
IMG = (1200, 1920)

M3 = 'androidx.compose.material3.'
LAYOUT = 'androidx.compose.foundation.layout.'
RT = 'androidx.compose.runtime.'
STATE = [RT + 'getValue', RT + 'setValue', RT + 'remember']


# ── helpers ────────────────────────────────────────────────

def load(name: str) -> Dict:
    """Loads a JSON fixture graph.

    Input:  name — file name under tests/fixtures/codegen
    Output: graph dictionary
    """
    return json.loads((FIXTURES / name).read_text(encoding='utf-8'))


def node(i: int, bbox: List[float], cls: str = 'TextView', conf: Optional[float] = 0.9,
         children: Optional[List[Dict]] = None) -> Dict:
    """Builds a hierarchy node.

    Input:  i — id; bbox — [x1, y1, x2, y2]; cls — class; conf — confidence;
            children — child nodes
    Output: node dictionary
    """
    return {'id': i, 'class': cls, 'bbox': list(bbox), 'confidence': conf,
            'children': list(children or [])}


def graph(roots: List[Dict], **extra) -> Dict:
    """Wraps root nodes into a minimal UI graph.

    Input:  roots — root nodes; extra — more top-level keys
    Output: graph dictionary
    """
    g = {'nodes': [], 'edges': [], 'hierarchy': {'roots': roots}}
    g.update(extra)
    return g


def gen(g, trace: bool = False, **kwargs) -> str:
    """Generates Kotlin with a fresh generator (image size 1200x1920 unless given).

    Input:  g — graph or node list; trace — trace_comments flag; kwargs — generate() keywords
    Output: Kotlin source
    """
    kwargs.setdefault('image_size', IMG)
    return ComposeCodeGenerator(trace_comments=trace).generate(g, **kwargs)


def strip_literals(code: str) -> str:
    """Removes string literals and // comments, so brackets inside them do not count.

    Input:  code — Kotlin source
    Output: code without string literals and line comments
    """
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    return re.sub(r'//[^\n]*', '', code)


def assert_balanced(code: str):
    """Asserts that braces and parentheses are balanced and never close before they open.

    Input:  code — Kotlin source
    Output: none (assertion)
    """
    for open_c, close_c in (('{', '}'), ('(', ')')):
        depth = 0
        for ch in strip_literals(code):
            depth += (ch == open_c) - (ch == close_c)
            assert depth >= 0, f"unbalanced {close_c}"
        assert depth == 0, f"unbalanced {open_c}"


def imports_of(code: str) -> List[str]:
    """Returns the import paths of a Kotlin source.

    Input:  code — Kotlin source
    Output: list of import paths in file order
    """
    return re.findall(r'^import (\S+)$', code, flags=re.M)


def body_of(code: str) -> str:
    """Returns the code after the imports with literals and comments stripped.

    Input:  code — Kotlin source
    Output: function body text
    """
    return strip_literals('@Composable' + code.split('\n@Composable', 1)[1])


def traced_ids(code: str) -> Counter:
    """Counts the detection ids named by trace comments (leaves and list containers).

    Input:  code — Kotlin source generated with trace_comments=True
    Output: Counter of ids
    """
    ids = re.findall(r'^\s*// leaf \S+#(-?\d+) ', code, flags=re.M)
    ids += re.findall(r'^\s*// LazyColumn \(rule=list, list container \S+#(-?\d+) ', code, flags=re.M)
    return Counter(int(i) for i in ids)


def graph_ids(g: Dict) -> set:
    """Returns the ids of all detections listed in graph['nodes'].

    Input:  g — UI graph
    Output: set of ids
    """
    return {n['id'] for n in g['nodes']}


def assert_imports_exact(code: str):
    """Asserts no wildcard imports, every import used, and every used symbol imported.

    Input:  code — Kotlin source
    Output: none (assertions)
    """
    imports = imports_of(code)
    body = body_of(code)
    assert imports == sorted(set(imports)), "imports must be unique and sorted"
    assert not any(i.endswith('*') for i in imports)
    names = {i.rsplit('.', 1)[1]: i for i in imports}
    for name in names:
        if name in ('getValue', 'setValue'):           # operators of 'by' delegation
            assert ' by remember' in body
        else:
            assert re.search(r'\b' + name + r'\b', body), f"unused import {name}"
    assert 'Composable' in names
    # Every capitalised call / reference and every modifier call has an import
    for sym in set(re.findall(r'\b([A-Z]\w*)\b(?=\s*[(.])', body)) - {'android'}:
        if sym in ('WebView', 'GeneratedScreen'):       # fully qualified / declared here
            continue
        assert sym in names, f"{sym} used without import"
    member_calls = {'weight', 'spacedBy', 'item', 'LightGray', 'CenterVertically',
                    'CenterHorizontally', 'current', 'webkit', 'WebView', 'dp'}
    for call in set(re.findall(r'\.([a-zA-Z]\w*)\b', body)):
        if call in member_calls:
            continue
        assert call in names, f".{call} used without import"
    if ' by remember' in body:
        assert {'getValue', 'setValue', 'remember'} <= set(names)


def lazy_columns_are_bounded(code: str) -> bool:
    """Tells whether every LazyColumn( is followed by a heightIn(max = N.dp) modifier.

    Input:  code — Kotlin source
    Output: True if every LazyColumn has a bounded maximum height
    """
    lines = code.splitlines()
    starts = [i for i, l in enumerate(lines) if re.search(r'\bLazyColumn\($', l)]
    return all(any(re.search(r'\.heightIn\(max = \d+\.dp\)', l) for l in lines[i + 1:i + 4])
               for i in starts)


# ── Shazam-like fixture: no detection dropped or duplicated ──

def test_shazam_like_no_id_dropped_or_duplicated():
    """Two partly overlapping ImageView containers sharing two children: every id is emitted once."""
    g = load('shazam_like_graph.json')
    code = gen(g, trace=True)
    counts = traced_ids(code)
    assert set(counts) == graph_ids(g) and set(counts.values()) == {1}
    body = body_of(code)
    assert len(re.findall(r'\bImage\(', body)) == 5          # ImageViews 1, 3, 8, 9, 10
    assert code.count('text = "Text Content"') == 4          # TextViews 4, 7, 11, 12
    assert len(re.findall(r'\bButton\(', body)) == 1
    generator = ComposeCodeGenerator()
    assert generator.last_layout_tree is None                 # set only by generate()
    generator.generate(g, image_size=IMG)
    assert generator.last_layout_tree['members'] == sorted(graph_ids(g))


def test_shazam_like_container_becomes_box_with_its_children():
    """ImageView#8 is a Box holding its own Image and the Column of #10 and #11; #9 stays an Image leaf."""
    code = gen(load('shazam_like_graph.json'), trace=True)
    lines = code.splitlines()
    box = lines.index('            // Box (rule=containment, container ImageView#8 (conf 0.38))')
    assert lines[box + 1].strip() == 'Box('
    seq = [l.strip() for l in lines[box:] if l.strip().startswith('// leaf')]
    assert seq[:4] == ['// leaf ImageView#8 (conf 0.38)', '// leaf ImageView#10 (conf 0.51)',
                       '// leaf TextView#11 (conf 0.48)', '// leaf ImageView#9 (conf 0.31)']


def test_v110_example_graphs_emit_every_detection_once():
    """On the ten v1.1.0-format example graphs every detection id appears exactly once in the code."""
    require_reference_data(V110_EXAMPLES)
    graphs = sorted(V110_EXAMPLES.glob('*/3_ui_graph.json'))
    assert len(graphs) == 10
    for path in graphs:
        g = json.loads(path.read_text(encoding='utf-8'))
        code = gen(g, trace=True)
        counts = traced_ids(code)
        assert set(counts) == graph_ids(g), path.parent.name
        assert set(counts.values()) == {1}, path.parent.name
        assert 'Unknown:' not in code and lazy_columns_are_bounded(code)
        assert_balanced(code)
        assert_imports_exact(code)


# ── templates ──────────────────────────────────────────────

TEMPLATE_CASES = [
    ('Button', [0, 0, 400, 100], r'Button\(\n\s+onClick = \{ /\* TODO \*/ \},\n\s+modifier = Modifier\.',
     [M3 + 'Button', M3 + 'Text']),
    ('TextView', [0, 0, 400, 100], r'Text\(\n\s+text = "Text Content",', [M3 + 'Text']),
    ('EditText', [0, 0, 400, 100], r'var textValue0 by remember \{ mutableStateOf\(""\) \}\n\s*TextField\(',
     [M3 + 'TextField', RT + 'mutableStateOf'] + STATE),
    ('ImageView', [0, 0, 400, 400], r'Image\(\n\s+painter = ColorPainter\(Color\.LightGray\),',
     ['androidx.compose.foundation.Image', 'androidx.compose.ui.graphics.Color',
      'androidx.compose.ui.graphics.painter.ColorPainter']),
    ('CheckBox', [0, 0, 80, 80], r'var checked0 by remember \{ mutableStateOf\(false\) \}\n\s*Checkbox\(',
     [M3 + 'Checkbox', RT + 'mutableStateOf'] + STATE),
    ('Switch', [0, 0, 120, 80], r'var switched0 by remember \{ mutableStateOf\(false\) \}\n\s*Switch\(',
     [M3 + 'Switch', RT + 'mutableStateOf'] + STATE),
    ('RadioButton', [0, 0, 80, 80],
     r'var selected0 by remember \{ mutableStateOf\(false\) \}\n\s*RadioButton\(\n\s+selected = selected0,'
     r'\n\s+onClick = \{ selected0 = !selected0 \}',
     [M3 + 'RadioButton', RT + 'mutableStateOf'] + STATE),
    ('SeekBar', [0, 0, 1000, 60],
     r'var sliderValue0 by remember \{ mutableFloatStateOf\(0\.5f\) \}\n\s*Slider\(\n\s+value = sliderValue0,'
     r'\n\s+onValueChange = \{ sliderValue0 = it \},',
     [M3 + 'Slider', RT + 'mutableFloatStateOf'] + STATE),
    ('ProgressBar', [0, 0, 1000, 30], r'LinearProgressIndicator\(\n\s+modifier = ',
     [M3 + 'LinearProgressIndicator']),
    ('ProgressBar', [0, 0, 60, 60], r'CircularProgressIndicator\(\n\s+modifier = ',
     [M3 + 'CircularProgressIndicator']),
    ('Spinner', [0, 0, 400, 100],
     r'var expanded0 by remember \{ mutableStateOf\(false\) \}\n\s*Box\(\n.*\n\s*\) \{\n'
     r'\s*OutlinedButton\(onClick = \{ expanded0 = true \}\) \{\n\s*Text\("Select"\)\n\s*\}\n'
     r'\s*DropdownMenu\(\n\s*expanded = expanded0,\n\s*onDismissRequest = \{ expanded0 = false \}\n\s*\) \{\n'
     r'\s*DropdownMenuItem\(\n\s*text = \{ Text\("Option 1"\) \},\n\s*onClick = \{ expanded0 = false \}',
     [M3 + 'OutlinedButton', M3 + 'DropdownMenu', M3 + 'DropdownMenuItem', M3 + 'Text', LAYOUT + 'Box',
      RT + 'mutableStateOf'] + STATE),
    ('WebView', [0, 0, 1200, 600],
     r'if \(LocalInspectionMode\.current\) \{\n\s*Box\(\n.*\n\s*\) \{\n\s*Text\("WebView"\)\n\s*\}\n'
     r'\s*\} else \{\n\s*AndroidView\(\n\s*factory = \{ context -> android\.webkit\.WebView\(context\) \},',
     ['androidx.compose.ui.platform.LocalInspectionMode', 'androidx.compose.ui.viewinterop.AndroidView',
      LAYOUT + 'Box', M3 + 'Text']),
    ('ListView', [0, 0, 1200, 600],
     r'LazyColumn\(\n\s*modifier = Modifier\n\s*\.(fillMaxWidth\(\)|width\(411\.dp\))\n\s*\.heightIn\(max = 206\.dp\)\n\s*\) \{\n'
     r'\s*item \{\n\s*Text\("ListView: no items detected"\)',
     ['androidx.compose.foundation.lazy.LazyColumn', LAYOUT + 'heightIn', M3 + 'Text']),
    ('RecyclerView', [0, 0, 1200, 600], r'item \{\n\s*Text\("RecyclerView: no items detected"\)',
     ['androidx.compose.foundation.lazy.LazyColumn', LAYOUT + 'heightIn', M3 + 'Text']),
]


@pytest.mark.parametrize('size_from_boxes', [False, True])
@pytest.mark.parametrize('cls, bbox, pattern, needed', TEMPLATE_CASES,
                         ids=[f"{c[0]}-{c[1][2]}x{c[1][3]}" for c in TEMPLATE_CASES])
def test_every_template(cls, bbox, pattern, needed, size_from_boxes):
    """Each detector class gets its composable, its imports, balanced brackets and no placeholder."""
    code = gen(graph([node(0, bbox, cls)]), size_from_boxes=size_from_boxes)
    assert re.search(pattern, code), code
    if size_from_boxes:
        assert re.search(r'Modifier\.size\(width = \d+\.dp, height = \d+\.dp\)|\.width\(\d+\.dp\)', code)
    else:
        assert 'size(width =' not in code
    imports = set(imports_of(code))
    assert set(needed) <= imports
    if cls == 'WebView':                                # box-derived height in both modes
        assert (LAYOUT + 'size' if size_from_boxes else LAYOUT + 'height') in imports
    assert_balanced(code)
    assert_imports_exact(code)
    assert 'Unknown:' not in code
    assert 'Experimental' not in code and '@OptIn' not in code


def test_textview_imports_are_exactly_the_used_ones():
    """A single TextView needs exactly five imports."""
    code = gen(graph([node(0, [0, 0, 100, 40])]))
    assert imports_of(code) == ['androidx.compose.foundation.layout.padding', M3 + 'Text',
                                RT + 'Composable', 'androidx.compose.ui.Modifier',
                                'androidx.compose.ui.unit.dp']


def test_component_map_covers_every_dataset_class():
    """All 13 detector classes of configs/dataset.yaml have a template (none is a placeholder)."""
    yaml = pytest.importorskip('yaml')
    names = yaml.safe_load((REPO / 'configs' / 'dataset.yaml').read_text(encoding='utf-8'))['names']
    assert set(ComposeCodeGenerator.COMPONENT_MAP) == set(names)
    for cls in names:
        code = gen(graph([node(0, [0, 0, 400, 100], cls)]))
        assert 'Unknown:' not in code, cls


def test_unknown_class_gets_escaped_placeholder():
    """An unknown class name gets the explicit placeholder; the name is escaped for Kotlin."""
    code = gen(graph([node(0, [0, 0, 100, 40], 'My"$Widget')]))
    assert 'Text("Unknown: My\\"\\$Widget")' in code
    assert_balanced(code)


def test_progress_bar_square_threshold():
    """Aspect ratio <= 1.5 -> CircularProgressIndicator, otherwise LinearProgressIndicator."""
    assert 'CircularProgressIndicator(' in gen(graph([node(0, [0, 0, 150, 100], 'ProgressBar')]))
    assert 'CircularProgressIndicator(' in gen(graph([node(0, [0, 0, 100, 150], 'ProgressBar')]))
    assert 'LinearProgressIndicator(' in gen(graph([node(0, [0, 0, 151, 100], 'ProgressBar')]))
    assert 'LinearProgressIndicator(' in gen(graph([node(0, [0, 0, 100, 0], 'ProgressBar')]))


def test_state_variable_names_are_unique():
    """Several stateful widgets get distinct remembered state variables, each used by its widget."""
    classes = ['Switch', 'RadioButton', 'SeekBar', 'Spinner', 'EditText', 'CheckBox'] * 3
    roots = [node(i, [0, 100 * i, 300, 100 * i + 60], cls) for i, cls in enumerate(classes)]
    code = gen(graph(roots))
    names = re.findall(r'\bvar (\w+) by remember', code)
    assert len(names) == len(classes) == len(set(names))
    for name in names:
        assert len(re.findall(r'\b' + name + r'\b', code)) >= 3   # declaration + read + write
    assert_balanced(code)


# ── composites ─────────────────────────────────────────────

def test_containment_box_wraps_container_widget_then_children():
    """An ImageView containing a play icon becomes Box { Image(container); Image(icon) }."""
    g = graph([node(0, [0, 0, 1200, 800], 'ImageView', 0.6, [node(1, [500, 300, 700, 500], 'ImageView', 0.5)])])
    lines = [l.strip() for l in gen(g).splitlines()]
    i = lines.index('Box(')
    assert lines[i + 1] == 'modifier = Modifier.padding(4.dp)'
    assert [l for l in lines[i:] if l == 'Image('] == ['Image(', 'Image(']


def test_list_with_children_has_one_item_per_group_and_bounded_height():
    """A ListView with three stacked rows gives LazyColumn with three item { } and heightIn."""
    rows = []
    for k in range(3):
        y = 100 + 200 * k
        rows += [node(10 + 2 * k, [20, y, 200, y + 100], 'ImageView'), node(11 + 2 * k, [250, y, 1100, y + 100])]
    code = gen(graph([node(0, [0, 80, 1200, 720], 'ListView', 0.9, rows)]), trace=True)
    assert code.count('item {') == 3 and code.count('LazyColumn(') == 1
    assert '.heightIn(max = 219.dp)' in code                # 640 px * 411 / 1200 = 219.2
    assert traced_ids(code) == Counter({0: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1})


@pytest.mark.parametrize('layout', ['xycut', 'flat'])
@pytest.mark.parametrize('size_from_boxes', [False, True])
def test_lazy_column_is_always_bounded(layout, size_from_boxes):
    """Every emitted LazyColumn has heightIn(max = ...), also nested lists and lists inside a Row."""
    nested = node(0, [0, 0, 1200, 1500], 'ListView', 0.9, [
        node(1, [0, 100, 1200, 700], 'RecyclerView', 0.8, [node(2, [10, 120, 1100, 200])]),
        node(3, [0, 800, 1200, 1400], 'ListView', 0.8)])
    in_row = [node(10, [0, 1600, 500, 1900], 'ListView'), node(11, [600, 1600, 1200, 1900], 'RecyclerView')]
    for g in (graph([nested] + in_row), load('shazam_like_graph.json'), load('list_screen_graph.json')):
        code = gen(g, layout=layout, size_from_boxes=size_from_boxes)
        assert 'LazyColumn(' in code
        assert lazy_columns_are_bounded(code)
        assert_balanced(code)


def test_flat_layout_list_note_names_separately_emitted_children():
    """In the flat baseline a list with children says its children are emitted separately."""
    code = gen(load('list_screen_graph.json'), layout='flat')
    assert 'Text("ListView: 8 child detection(s) emitted separately (flat layout)")' in code
    generator = ComposeCodeGenerator()
    generator.generate(load('list_screen_graph.json'), layout='flat')
    assert all(c['type'] == 'leaf' for c in generator.last_layout_tree['children'])
    assert len(generator.last_layout_tree['children']) == 20      # ids 0..19, containment ignored


# ── sizing ─────────────────────────────────────────────────

def test_size_from_boxes_converts_px_to_dp():
    """dp = px * screen_width_dp / image width (half up), for leaves, Boxes and LazyColumns."""
    g = graph([node(0, [0, 0, 300, 60]),
               node(1, [0, 100, 600, 700], 'ImageView', 0.5, [node(2, [100, 200, 200, 300], 'ImageView')]),
               node(3, [0, 800, 1200, 1400], 'ListView')])
    code = gen(g, size_from_boxes=True)
    assert 'modifier = Modifier.size(width = 103.dp, height = 21.dp)' in code    # 102.75, 20.55
    assert 'Box(\n            modifier = Modifier.size(width = 206.dp, height = 206.dp)' in code
    assert '.width(411.dp)' in code and '.heightIn(max = 206.dp)' in code
    narrow = gen(g, size_from_boxes=True, screen_width_dp=360)
    assert 'Modifier.size(width = 90.dp, height = 18.dp)' in narrow
    default = gen(g)
    assert 'Modifier.size(width' not in default and '.heightIn(max = 206.dp)' in default
    assert 'text = "Text Content",\n            modifier = Modifier.padding(4.dp)' in default
    with pytest.raises(ValueError):
        gen(g, screen_width_dp=0)


def test_full_width_templates_use_weight_inside_a_row():
    """TextField, Slider, LinearProgressIndicator, WebView and lists use weight(1f) as Row children."""
    row = [node(0, [0, 0, 100, 100]), node(1, [150, 0, 300, 100], 'EditText'),
           node(2, [350, 0, 500, 100], 'SeekBar'), node(3, [550, 0, 700, 30], 'ProgressBar'),
           node(4, [750, 0, 900, 100], 'WebView'), node(5, [950, 0, 1100, 100], 'ListView')]
    code = gen(graph(row))
    body = body_of(code)
    assert code.count('modifier = Modifier.weight(1f).padding(8.dp)') == 3
    assert 'Modifier.weight(1f).height(34.dp).padding(8.dp)' in code       # 100 px * 411 / 1200
    assert '.weight(1f)\n                .heightIn(max = 34.dp)' in code
    assert 'fillMaxWidth().padding(8.dp)' not in body
    alone = gen(graph([node(1, [0, 0, 300, 100], 'EditText')]))
    assert 'modifier = Modifier.fillMaxWidth().padding(8.dp)' in alone


def _own_modifier(lines: List[str]) -> str:
    """Returns the first modifier expression in a block of Kotlin lines (one- or multi-line).

    Input:  lines — the lines of one emitted node, after its trace comment
    Output: e.g. 'Modifier.weight(1f).padding(4.dp)' or 'Modifier.weight(1f).heightIn(max = 9.dp)';
            '' when the block has no modifier
    """
    for k, line in enumerate(lines):
        m = re.search(r'modifier = (Modifier\S*)', line)
        if m:
            expr = m.group(1).rstrip(',')
            for cont in lines[k + 1:]:
                if not cont.strip().startswith('.'):
                    break
                expr += cont.strip().rstrip(',')
            return expr
    return ''


def starved_row_children(code: str) -> List[str]:
    """Lists the direct children of a Row that can take the Row's whole remaining width.

    Compose measures the children of a Row without weight first, with the
    remaining width as maximum; a child without weight(1f) and without a fixed
    width (size(width = ..) or width(..)) whose subtree uses fillMaxWidth()
    therefore takes all of that width and leaves its later siblings 0 dp
    wide.  The nesting is read from the trace comments: every emitted node
    starts with a '//' comment at its own indentation.

    Input:  code — Kotlin source generated with trace_comments=True
    Output: trace-comment texts of such children (empty if there are none)
    """
    lines = code.splitlines()
    comments = [(i, len(l) - len(l.lstrip()), l.strip()[3:]) for i, l in enumerate(lines)
                if i > 0 and l.lstrip().startswith('// ')]
    bad = []
    for k, (i, ind, text) in enumerate(comments):
        parent = next((c for c in reversed(comments[:k]) if c[1] < ind), None)
        if parent is None or not parent[2].startswith('Row ('):
            continue
        end = next((c[0] for c in comments[k + 1:] if c[1] <= ind), len(lines))
        block = lines[i + 1:end]
        own = _own_modifier(block)
        if 'weight(1f)' in own or 'size(width' in own or re.search(r'\.width\(', own):
            continue
        if any('fillMaxWidth()' in l for l in block):
            bad.append(text)
    return bad


def test_starved_row_children_detects_an_unweighted_full_width_child():
    """Negative control of the helper: an unweighted Box whose inner Column fills the width is reported."""
    code = "\n".join([
        "// header",
        "        // Row (rule=gap)",
        "        Row(",
        "            // Box (rule=containment, container ImageView#0 (conf 0.90))",
        "            Box(",
        "                modifier = Modifier.padding(4.dp)",
        "            ) {",
        "                // Column (rule=gap)",
        "                Column(",
        "                    modifier = Modifier",
        "                        .fillMaxWidth()",
        "                        .padding(4.dp),",
        "                ) {",
        "                }",
        "            }",
        "            // leaf EditText#3 (conf 0.90)",
        "            TextField(",
        "                modifier = Modifier.weight(1f).padding(8.dp)",
        "            )",
        "        }"])
    assert starved_row_children(code) == ['Box (rule=containment, container ImageView#0 (conf 0.90))']
    assert starved_row_children(code.replace('Modifier.padding(4.dp)', 'Modifier.weight(1f).padding(4.dp)')) == []


def test_containment_box_inside_a_row_gets_weight():
    """A containment Box beside a weighted sibling in a Row has weight(1f), so the sibling keeps its width."""
    box = node(0, [0, 0, 500, 400], 'ImageView', 0.9,
               [node(1, [20, 20, 200, 60]), node(2, [20, 300, 200, 340])])
    g = graph([box, node(3, [600, 100, 1100, 200], 'EditText')])
    code = gen(g, trace=True)
    assert 'Box(\n            modifier = Modifier.weight(1f).padding(4.dp)' in code
    assert 'modifier = Modifier.weight(1f).padding(8.dp)' in code          # the TextField
    assert starved_row_children(code) == []
    alone = gen(graph([box]))                                               # not in a Row: unchanged
    assert 'Box(\n        modifier = Modifier.padding(4.dp)' in alone
    sized = gen(g, size_from_boxes=True)                                    # fixed size, no weight
    assert 'Box(\n            modifier = Modifier.size(width = 171.dp, height = 137.dp)' in sized


def _golden_screen_graph(stem: str) -> Dict:
    """Builds the UI graph of a golden example as the pipeline does (default settings).

    Input:  stem — directory name under tests/golden
    Output: UI graph built from the de-duplicated golden raw detections
    """
    from ui2compose.infer.build_ui_graph import UIGraphBuilder
    from ui2compose.infer.postprocess import deduplicate_detections

    raw = json.loads((REPO / 'tests' / 'golden' / stem / '1_detections_raw.json').read_text(encoding='utf-8'))
    kept, _ = deduplicate_detections(raw)
    return UIGraphBuilder().build_graph(kept, image_size=IMG)


_GOLDEN_STEMS = sorted(p.name for p in (REPO / 'tests' / 'golden').iterdir() if p.is_dir())


@pytest.mark.parametrize('source', ['all_templates_graph.json', 'shazam_like_graph.json',
                                    'list_screen_graph.json'] + _GOLDEN_STEMS)
@pytest.mark.parametrize('size_from_boxes', [False, True])
@pytest.mark.parametrize('fallback, cut_tol_frac', [('y_sorted', 0.005), ('least_overlap', 0.0)])
def test_no_row_child_can_take_the_whole_row(source, size_from_boxes, fallback, cut_tol_frac):
    """In every fixture and example, no direct Row child without weight or fixed width uses fillMaxWidth()."""
    g = load(source) if source.endswith('.json') else _golden_screen_graph(source)
    code = gen(g, trace=True, size_from_boxes=size_from_boxes, fallback=fallback,
               cut_tol_frac=cut_tol_frac, image_size=None if source.endswith('.json') else IMG)
    assert starved_row_children(code) == []


# ── determinism, edges, options ────────────────────────────

def test_output_is_deterministic_and_input_not_mutated():
    """Same input -> same code; the input graph is left unchanged."""
    g = load('list_screen_graph.json')
    before = copy.deepcopy(g)
    a = gen(g, trace=True)
    b = gen(copy.deepcopy(g), trace=True)
    generator = ComposeCodeGenerator(trace_comments=True)
    c1 = generator.generate(g, image_size=IMG)
    c2 = generator.generate(g, image_size=IMG)
    assert a == b == c1 == c2
    assert g == before


def test_graph_edges_are_ignored():
    """The spatial edges do not influence the generated code."""
    g = load('shazam_like_graph.json')
    no_edges = copy.deepcopy(g)
    no_edges['edges'] = []
    junk = copy.deepcopy(g)
    junk['edges'] = [{'source': 1, 'target': 17, 'relation': 'left_of', 'confidence': 0.9},
                     {'source': 12, 'target': 3, 'relation': 'parent_child', 'confidence': 1.0}]
    assert gen(g) == gen(no_edges) == gen(junk)


def test_use_xy_cut_maps_to_layout_and_layout_wins():
    """use_xy_cut=False is the flat baseline; an explicit layout overrides use_xy_cut."""
    g = load('shazam_like_graph.json')
    assert gen(g, use_xy_cut=False) == gen(g, layout='flat')
    assert gen(g, use_xy_cut=False, layout='xycut') == gen(g)
    assert gen(g, use_xy_cut=True, layout='flat') == gen(g, layout='flat')


def test_last_layout_tree_is_the_emitted_tree():
    """generate() keeps the layout tree it emitted, equal to build_layout_tree with the same options."""
    g = load('list_screen_graph.json')
    generator = ComposeCodeGenerator()
    generator.generate(g, cut_tol_frac=0.01, fallback='least_overlap')
    assert generator.last_layout_tree == build_layout_tree(g, cut_tol_frac=0.01, fallback='least_overlap')


def test_header_comment_records_the_options():
    """The first line records layout, tolerance, fallback, sizing, image size (and source) and screen width."""
    code = gen(load('shazam_like_graph.json'))
    assert code.splitlines()[0] == (
        "// Generated by ui2compose (layout=xycut, cut_tol_frac=0, fallback=y_sorted, "
        "size_from_boxes=false, image_size=1200x1920 (argument), screen_width_dp=411)")
    flat = ComposeCodeGenerator().generate(load('list_screen_graph.json'), layout='flat')
    assert flat.splitlines()[0] == ("// Generated by ui2compose (layout=flat, size_from_boxes=false, "
                                    "image_size=1080x1920 (graph), screen_width_dp=411)")


def test_trace_comments_are_off_by_default():
    """Without trace_comments no '//' line appears below the header."""
    code = gen(load('list_screen_graph.json'))
    assert [l for l in code.splitlines()[1:] if l.strip().startswith('//')] == []


def test_empty_graph_gives_compilable_empty_column():
    """A graph without detections yields an empty Column inside GeneratedScreen."""
    code = gen(graph([]))
    assert 'fun GeneratedScreen() {' in code and 'Column(' in code
    assert_balanced(code)
    assert_imports_exact(code)


def test_legacy_emitter_is_removed():
    """The unreachable v1.1.0 recursive emitter (Card/Toolbar/CardView branches) and the demo shims are gone."""
    for name in ('_is_horizontal_layout', '_generate_component', '_generate_card',
                 '_generate_container', '_generate_leaf_component', '_collect_leaves', '_xy_cut'):
        assert not hasattr(ComposeCodeGenerator, name)
    for g in (load('shazam_like_graph.json'), load('list_screen_graph.json')):
        body = body_of(gen(g))
        assert not re.search(r'\b(Card|TopAppBar)\(', body)


# ── golden files ───────────────────────────────────────────

GOLDEN = [
    ('shazam_like_graph.json', {}, {'image_size': IMG}, 'shazam_like_xycut.kt'),
    ('shazam_like_graph.json', {}, {'image_size': IMG, 'layout': 'flat'}, 'shazam_like_flat.kt'),
    ('list_screen_graph.json', {'trace_comments': True}, {'size_from_boxes': True}, 'list_screen_sized.kt'),
]


@pytest.mark.parametrize('graph_file, ctor, kwargs, golden', GOLDEN, ids=[g[3] for g in GOLDEN])
def test_golden_files(graph_file, ctor, kwargs, golden):
    """The generator output matches the hand-checked golden Kotlin files."""
    code = ComposeCodeGenerator(**ctor).generate(load(graph_file), **kwargs)
    assert code + "\n" == (FIXTURES / golden).read_text(encoding='utf-8')
    assert_balanced(code)
    assert_imports_exact(code)


# ── command line ───────────────────────────────────────────

def test_cli_writes_code_tree_and_explanations(tmp_path, capsys):
    """main() writes the .kt and the layout JSON and prints explain_path for the requested ids."""
    out, tree_json = tmp_path / 'Screen.kt', tmp_path / 'tree.json'
    graph_path = str(FIXTURES / 'shazam_like_graph.json')
    rc = main([graph_path, '-o', str(out), '--image-size', '1200', '1920', '--layout-json',
               str(tree_json), '--explain', '11', '9'])
    assert rc == 0
    assert out.read_text(encoding='utf-8') == gen(load('shazam_like_graph.json'))
    printed = capsys.readouterr().out
    assert 'Detection 11:' in printed and 'Detection 9:' in printed
    assert 'leaf TextView#11 (conf 0.48, bbox [700,862,774,936])' in printed
    assert 'Box (rule=containment, container ImageView#8 (conf 0.38)' in printed
    assert json.loads(tree_json.read_text(encoding='utf-8'))['meta']['image_size'] == [1200, 1920]


def test_cli_prints_code_without_output_and_rejects_unknown_ids(capsys):
    """Without -o the code goes to stdout; an unknown --explain id exits with status 2."""
    graph_path = str(FIXTURES / 'list_screen_graph.json')
    assert main([graph_path, '--layout', 'flat', '--size-from-boxes', '--trace-comments']) == 0
    assert capsys.readouterr().out.strip() == ComposeCodeGenerator(trace_comments=True).generate(
        load('list_screen_graph.json'), layout='flat', size_from_boxes=True)
    assert main([graph_path, '--explain', '99']) == 2
    assert 'detection id 99 is not in the layout tree' in capsys.readouterr().err
