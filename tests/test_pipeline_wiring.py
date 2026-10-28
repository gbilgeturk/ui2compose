# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Tests for the pipeline wiring (ui2compose.infer.pipeline and app/run_pipeline.py).

A fake detector replaces ComponentDetector, so the whole pipeline runs end to
end without torch or ultralytics: detection -> class_conf -> de-duplication ->
graph -> code generation, including the written files, the removed list, the
explain trace, the configuration file and the command line.
"""

import copy
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

from _reference_data import require_reference_data
from ui2compose.infer import pipeline
from ui2compose.infer.generate_compose_code import explain_path
from ui2compose.infer.postprocess import deduplicate_detections
from ui2compose.infer.pipeline import (CLI_DEFAULTS, config_to_kwargs, load_config,
                                       resolve_cli_settings, run_pipeline)

REPO_ROOT = Path(__file__).resolve().parents[1]
IMG_W, IMG_H = 600, 960

EXPECTED_FILES = {
    '0_run_config.json', '1_detections_raw.json', '1_detections.json',
    '1_dedup_removed.json', '1_detections_viz.png', '2_layout_tree.json',
    '3_ui_graph.json', '4_GeneratedScreen.kt',
}


def det(det_id: int, class_name: str, confidence: float, bbox: List[float],
        with_size: bool = True) -> Dict:
    """Builds a detection in the format of ComponentDetector.detect.

    Input:  det_id — id; class_name — class; confidence — score; bbox — [x1, y1, x2, y2];
            with_size — if True, add 'img_w'/'img_h' as the detector does
    Output: detection dictionary
    """
    x1, y1, x2, y2 = bbox
    d = {'id': det_id, 'class_name': class_name, 'class_id': 0, 'confidence': confidence,
         'bbox': [float(v) for v in bbox],
         'bbox_norm': [(x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H,
                       (x2 - x1) / IMG_W, (y2 - y1) / IMG_H],
         'center': [(x1 + x2) / 2, (y1 + y2) / 2], 'width': x2 - x1, 'height': y2 - y1}
    if with_size:
        d.update(img_w=IMG_W, img_h=IMG_H)
    return d


def fake_detections(with_size: bool = True) -> List[Dict]:
    """Returns a small screen: title row, an image card with a text inside, a duplicate of that text, a button.

    Input:  with_size — passed to det()
    Output: list of detections (id 4 duplicates id 3 with lower confidence)
    """
    return [det(0, 'TextView', 0.90, [20, 20, 280, 70], with_size),
            det(1, 'ImageView', 0.80, [300, 20, 580, 70], with_size),
            det(2, 'ImageView', 0.70, [40, 200, 560, 700], with_size),
            det(3, 'TextView', 0.60, [100, 300, 400, 360], with_size),
            det(4, 'TextView', 0.30, [101, 301, 399, 359], with_size),
            det(5, 'Button', 0.85, [200, 800, 400, 880], with_size)]


class FakeDetector:
    """Stand-in for ComponentDetector with the same detect()/visualize() interface."""

    def __init__(self, detections: List[Dict]):
        """Stores the detections to return and prepares the call logs.

        Input:  detections — list returned (as a deep copy) by every detect() call
        Output: none
        """
        self.detections = detections
        self.detect_calls: List[Dict] = []
        self.visualize_calls: List[Dict] = []

    def detect(self, image_path: str, conf_threshold: float = 0.3, iou_threshold: float = 0.7,
               imgsz: Optional[int] = None, agnostic_nms: bool = False,
               max_det: int = 300) -> List[Dict]:
        """Records the detection settings and returns the stored detections.

        Input:  same parameters as ComponentDetector.detect
        Output: deep copy of the stored detections
        """
        self.detect_calls.append({'conf_threshold': conf_threshold, 'iou_threshold': iou_threshold,
                                  'imgsz': imgsz, 'agnostic_nms': agnostic_nms,
                                  'max_det': max_det})
        return copy.deepcopy(self.detections)

    def visualize(self, image_path: str, detections: List[Dict], output_path: Optional[str] = None,
                  removed: Optional[List[Dict]] = None):
        """Records the call and writes a placeholder image file.

        Input:  same parameters as ComponentDetector.visualize
        Output: none (side effect of writing output_path)
        """
        self.visualize_calls.append({'ids': [d['id'] for d in detections],
                                     'removed_ids': [d['id'] for d in (removed or [])]})
        if output_path:
            Path(output_path).write_bytes(b"placeholder")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    """Writes a blank IMG_W x IMG_H PNG screenshot.

    Input:  tmp_path — pytest temporary directory
    Output: path of the PNG file
    """
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "screen.png"
    pil.new("RGB", (IMG_W, IMG_H), "white").save(path)
    return path


def run(image: Path, out: Path, detector: FakeDetector, **kwargs) -> Dict:
    """Runs the pipeline with the fake detector.

    Input:  image — screenshot path; out — output directory; detector — fake detector;
            kwargs — further run_pipeline keywords
    Output: run_pipeline result dictionary
    """
    return run_pipeline(str(image), "unused.pt", "unused.yaml", output_dir=str(out),
                        detector=detector, **kwargs)


def read_json(path: Path):
    """Reads a JSON file.

    Input:  path — file path
    Output: parsed JSON value
    """
    return json.loads(path.read_text(encoding='utf-8'))


def roundtrip(obj):
    """Passes an object through JSON, as it is written to disk.

    Input:  obj — JSON-serialisable object
    Output: the object after json.dumps / json.loads
    """
    return json.loads(json.dumps(obj))


# ── run_pipeline end to end ────────────────────────────────


def test_run_pipeline_writes_all_files(tmp_path, image):
    """Every stage writes its file, and the files hold exactly the returned results."""
    fake = FakeDetector(fake_detections())
    out = tmp_path / "out"
    res = run(image, out, fake, explain_id=3)

    assert {p.name for p in out.iterdir()} == EXPECTED_FILES | {'2_explain_3.txt'}
    assert set(res) == {'raw_detections', 'detections', 'removed', 'graph', 'layout_tree',
                        'code', 'explain', 'output_dir'}
    assert read_json(out / '1_detections_raw.json') == fake_detections()
    assert read_json(out / '1_detections_raw.json') == roundtrip(res['raw_detections'])
    assert read_json(out / '1_detections.json') == roundtrip(res['detections'])
    assert read_json(out / '1_dedup_removed.json') == roundtrip(res['removed'])
    assert read_json(out / '3_ui_graph.json') == roundtrip(res['graph'])
    assert read_json(out / '2_layout_tree.json') == roundtrip(res['layout_tree'])
    assert res['layout_tree'] is not None
    assert (out / '4_GeneratedScreen.kt').read_text(encoding='utf-8') == res['code']
    assert '@Composable' in res['code']
    assert res['output_dir'] == str(out)

    config = read_json(out / '0_run_config.json')
    assert config['iou_threshold'] == 0.7 and config['explain_id'] == 3
    assert config['containment_ratio'] == 0.9 and config['layout'] == 'xycut'


def test_removed_list_is_honoured(tmp_path, image):
    """The duplicate (id 4) is removed, reported with its rule, absent from graph and code input, drawn as removed."""
    fake = FakeDetector(fake_detections())
    res = run(image, tmp_path / "out", fake)

    assert [d['id'] for d in res['removed']] == [4]
    assert res['removed'][0]['dedup_rule'] == 'same_class_iou'
    assert res['removed'][0]['removed_by'] == 3
    assert [d['id'] for d in res['detections']] == [0, 1, 2, 3, 5]
    assert [n['id'] for n in res['graph']['nodes']] == [0, 1, 2, 3, 5]
    assert '"id": 4,' not in json.dumps(res['graph']['hierarchy'])
    assert fake.visualize_calls == [{'ids': [0, 1, 2, 3, 5], 'removed_ids': [4]}]
    # ids are not renumbered, so they can be followed across files
    assert [d['id'] for d in res['raw_detections']] == [0, 1, 2, 3, 4, 5]


def test_detector_defaults_reproduce_v110_settings(tmp_path, image):
    """Defaults: conf 0.3, NMS IoU 0.7, the checkpoint's imgsz, class-aware NMS, max_det 300 (as v1.1.0)."""
    fake = FakeDetector(fake_detections())
    run(image, tmp_path / "a", fake)
    assert fake.detect_calls == [{'conf_threshold': 0.3, 'iou_threshold': 0.7, 'imgsz': None,
                                  'agnostic_nms': False, 'max_det': 300}]

    fake = FakeDetector(fake_detections())
    run(image, tmp_path / "b", fake, conf_threshold=0.15, iou_threshold=0.5, imgsz=320,
        agnostic_nms=True, max_det=50)
    assert fake.detect_calls == [{'conf_threshold': 0.15, 'iou_threshold': 0.5, 'imgsz': 320,
                                  'agnostic_nms': True, 'max_det': 50}]


def test_no_dedup_keeps_every_detection(tmp_path, image):
    """dedup=False passes all detections on and writes an empty removed list."""
    out = tmp_path / "out"
    res = run(image, out, FakeDetector(fake_detections()), dedup=False)
    assert res['removed'] == [] and read_json(out / '1_dedup_removed.json') == []
    assert [d['id'] for d in res['detections']] == [0, 1, 2, 3, 4, 5]


def test_class_conf_and_dedup_params_are_applied(tmp_path, image):
    """class_conf filters before de-duplication; dedup_params reach deduplicate_detections."""
    res = run(image, tmp_path / "out", FakeDetector(fake_detections()),
              class_conf={'Button': 0.9},
              dedup_params={'same_class_iou': None, 'same_class_iomin': None})
    assert [d['id'] for d in res['detections']] == [0, 1, 2, 3, 4]   # 5 filtered, 4 kept
    assert res['removed'] == []                                       # class_conf is not dedup


def test_run_config_records_the_effective_values(tmp_path, image):
    """0_run_config.json resolves None defaults: dedup thresholds, cut tolerance and imgsz actually used."""
    from ui2compose.infer.generate_compose_code import DEFAULT_CUT_TOL_FRAC
    dedup_defaults = {name: p.default for name, p in
                      inspect.signature(deduplicate_detections).parameters.items()
                      if name != 'detections'}
    out = tmp_path / "a"
    run(image, out, FakeDetector(fake_detections()))
    config = read_json(out / '0_run_config.json')
    assert config['dedup_params'] == dedup_defaults == {'same_class_iou': 0.4,
                                                        'same_class_iomin': 0.85,
                                                        'cross_class_iou': 0.5}
    assert config['cut_tol_frac'] == DEFAULT_CUT_TOL_FRAC == read_json(out / '2_layout_tree.json')['meta']['cut_tol_frac']
    assert config['imgsz'] is None and config['imgsz_effective'] is None   # fake: no checkpoint

    class WithCheckpoint(FakeDetector):
        """Fake detector that reports a checkpoint imgsz like ComponentDetector.effective_imgsz."""

        def effective_imgsz(self, imgsz=None):
            """Returns imgsz, or the 'checkpoint' value 416.

            Input:  imgsz — requested size or None
            Output: effective size
            """
            return 416 if imgsz is None else imgsz

    out = tmp_path / "b"
    run(image, out, WithCheckpoint(fake_detections()), dedup_params={'cross_class_iou': None},
        cut_tol_frac=0.0)
    config = read_json(out / '0_run_config.json')
    assert config['dedup_params'] == {**dedup_defaults, 'cross_class_iou': None}
    assert config['cut_tol_frac'] == 0.0 and config['imgsz_effective'] == 416
    out = tmp_path / "c"
    run(image, out, WithCheckpoint(fake_detections()), dedup=False, imgsz=320)
    config = read_json(out / '0_run_config.json')
    assert config['dedup_params'] is None and config['imgsz_effective'] == 320
    with pytest.raises(ValueError, match='unknown de-duplication parameter'):
        run(image, tmp_path / "d", FakeDetector(fake_detections()), dedup_params={'iou': 0.5})


def test_unknown_class_conf_name_stops_before_detection(tmp_path, image):
    """A misspelt class name in class_conf raises before the detector runs (it would filter nothing)."""
    fake = FakeDetector(fake_detections())
    with pytest.raises(ValueError, match="unknown class 'Buton'.*did you mean 'Button'"):
        run(image, tmp_path / "out", fake, class_conf={'Buton': 0.9})
    assert fake.detect_calls == []
    fake.class_names = ['Button', 'Card']           # a detector with other classes
    res = run(image, tmp_path / "ok", fake, class_conf={'Card': 0.9, 'Button': 0.9})
    assert 5 not in [d['id'] for d in res['detections']]


def test_layout_from_boxes_equals_the_pipeline_layout(tmp_path, image):
    """Plain boxes laid out with layout_from_boxes give exactly the pipeline's layout tree (containment included)."""
    from ui2compose.infer.generate_compose_code import build_layout_tree
    from ui2compose.infer.pipeline import layout_from_boxes
    for kwargs in ({}, {'cut_tol_frac': 0.0, 'fallback': 'least_overlap'}, {'layout': 'flat'}):
        res = run(image, tmp_path / f"o{len(kwargs)}", FakeDetector(fake_detections()),
                  dedup=False, **kwargs)
        boxes = [{'bbox': d['bbox'], 'class_name': d['class_name'], 'id': d['id'],
                  'confidence': d['confidence']} for d in fake_detections()]
        assert layout_from_boxes(boxes, image_size=(IMG_W, IMG_H), **kwargs) == res['layout_tree']
    # Ground-truth-like boxes (no id, no confidence) work; containment produces a Box, which
    # build_layout_tree on the plain list alone would not.
    gt = [{'bbox': [0, 0, 600, 400], 'class_name': 'ImageView'},
          {'bbox': [100, 100, 300, 200], 'class_name': 'TextView'},
          {'bbox': [0, 500, 600, 600], 'class_name': 'Button'}]
    tree = layout_from_boxes(gt, image_size=(IMG_W, IMG_H))
    assert any(c['type'] == 'Box' for c in tree['children'])
    plain = build_layout_tree([dict(b, id=i) for i, b in enumerate(gt)], image_size=(IMG_W, IMG_H))
    assert not any(c['type'] == 'Box' for c in plain['children'])


def test_graph_receives_image_size_and_thresholds(tmp_path, image):
    """The graph is built at the image's size with the given containment ratio and tolerance."""
    res = run(image, tmp_path / "out", FakeDetector(fake_detections()),
              containment_ratio=0.9, align_tol_frac=0.05)
    assert res['graph']['image_size'] == [IMG_W, IMG_H]
    assert res['graph']['params']['image_size_source'] == 'argument'
    assert res['graph']['params']['containment_ratio'] == 0.9
    assert res['graph']['params']['tolerance_px'] == pytest.approx(0.05 * IMG_W)


def test_image_size_falls_back_to_the_image_file(tmp_path, image):
    """Without img_w/img_h in the detections the size is read from the image (not the ReDraw default)."""
    res = run(image, tmp_path / "out", FakeDetector(fake_detections(with_size=False)))
    assert res['graph']['image_size'] == [IMG_W, IMG_H]
    res = run(image, tmp_path / "empty", FakeDetector([]))
    assert res['graph']['image_size'] == [IMG_W, IMG_H]
    assert res['detections'] == [] and '@Composable' in res['code']


def test_generator_receives_layout_options(tmp_path, image, monkeypatch):
    """layout / size_from_boxes / fallback / image_size reach generate(); cut_tol_frac only when set."""
    calls: List[Dict] = []

    class Recording(pipeline.ComposeCodeGenerator):
        """Code generator that records the keywords of generate()."""

        def generate(self, graph, **kwargs):
            """Records the keywords and delegates to the real generator.

            Input:  graph — UI graph; kwargs — generate() keywords
            Output: generated code
            """
            calls.append(kwargs)
            return super().generate(graph, **kwargs)

    monkeypatch.setattr(pipeline, 'ComposeCodeGenerator', Recording)
    run(image, tmp_path / "a", FakeDetector(fake_detections()))
    assert calls[-1] == {'image_size': (IMG_W, IMG_H), 'layout': 'xycut',
                         'size_from_boxes': False, 'fallback': 'y_sorted'}
    res = run(image, tmp_path / "b", FakeDetector(fake_detections()), layout='flat',
              size_from_boxes=True, cut_tol_frac=0.01)
    assert calls[-1] == {'image_size': (IMG_W, IMG_H), 'layout': 'flat',
                         'size_from_boxes': True, 'fallback': 'y_sorted', 'cut_tol_frac': 0.01}
    assert '@Composable' in res['code']


# ── explain trace ──────────────────────────────────────────


def test_explain_file_traces_a_kept_detection(tmp_path, image, capsys):
    """--explain writes containment (graph) and layout lines for a kept detection, and prints them."""
    out = tmp_path / "out"
    res = run(image, out, FakeDetector(fake_detections()), explain_id=3)
    text = (out / '2_explain_3.txt').read_text(encoding='utf-8')
    assert text.splitlines() == res['explain']
    assert "[post-processing] kept" in text
    assert "[graph] containment:" in text and "parent: 2 ImageView" in text
    layout_lines = text.split("[layout] path in the layout tree:\n")[1].splitlines()
    assert layout_lines == ["  " + line for line in explain_path(res['layout_tree'], 3)]
    assert layout_lines
    assert "parent: 2 ImageView" in capsys.readouterr().out


def test_explain_file_for_removed_filtered_and_unknown_ids(tmp_path, image):
    """The trace stops at the stage that removed the detection, naming the rule."""
    out = tmp_path / "a"
    run(image, out, FakeDetector(fake_detections()), explain_id=4)
    text = (out / '2_explain_4.txt').read_text(encoding='utf-8')
    assert "removed by de-duplication: rule 'same_class_iou'" in text
    assert "with kept detection 3" in text and "[graph]" not in text

    out = tmp_path / "b"
    run(image, out, FakeDetector(fake_detections()), explain_id=5, class_conf={'Button': 0.9})
    assert "per-class confidence" in (out / '2_explain_5.txt').read_text(encoding='utf-8')

    out = tmp_path / "c"
    run(image, out, FakeDetector(fake_detections()), explain_id=99)
    assert "no raw detection has id 99" in (out / '2_explain_99.txt').read_text(encoding='utf-8')


# ── configuration file and command line ────────────────────


def test_config_file_values_equal_code_defaults():
    """configs/pipeline_config.yaml lists every setting, with the same values as the code defaults."""
    kwargs = config_to_kwargs(load_config(str(REPO_ROOT / "configs/pipeline_config.yaml")))
    params = inspect.signature(run_pipeline).parameters
    expected_keys = set(params) - {'explain_id', 'detector'}
    assert set(kwargs) == expected_keys

    assert kwargs['model_path'] == CLI_DEFAULTS['model_path']
    assert kwargs['dataset_yaml'] == CLI_DEFAULTS['dataset_yaml']
    assert kwargs['output_dir'] == params['output_dir'].default == CLI_DEFAULTS['output_dir']
    dedup_defaults = {name: p.default for name, p in
                      inspect.signature(deduplicate_detections).parameters.items()
                      if name != 'detections'}
    assert kwargs['dedup_params'] == dedup_defaults
    for key in expected_keys - {'image_path', 'model_path', 'dataset_yaml', 'dedup_params'}:
        assert kwargs[key] == params[key].default, key
    assert kwargs['iou_threshold'] == 0.7 and kwargs['imgsz'] is None


def test_config_rejects_unknown_keys():
    """Misspelt sections or keys raise ValueError instead of being ignored."""
    with pytest.raises(ValueError):
        config_to_kwargs({'detection': {'iou': 0.5}})
    with pytest.raises(ValueError):
        config_to_kwargs({'detektion': {}})
    with pytest.raises(ValueError):
        config_to_kwargs({'dedup': {'same_class_iuo': 0.5}})
    with pytest.raises(ValueError):
        config_to_kwargs({'graph': 0.95})
    assert config_to_kwargs({'dedup': {'enabled': False, 'cross_class_iou': None}, 'output': None}) == \
        {'dedup': False, 'dedup_params': {'cross_class_iou': None}}
    assert config_to_kwargs({}) == {}


def test_load_config_handles_empty_and_invalid_files(tmp_path):
    """An empty YAML file is an empty config; a non-mapping top level is an error."""
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_config(str(empty)) == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        load_config(str(bad))


def test_cli_flags_override_config_values(tmp_path):
    """Precedence: explicit flags > --config values > defaults."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        'input': {'image': 'from_config.png'},
        'detection': {'iou_threshold': 0.5, 'imgsz': 320},
        'dedup': {'enabled': False, 'same_class_iou': 0.4},
        'layout': {'mode': 'flat'},
    }))
    s = resolve_cli_settings(['--config', str(cfg), '--iou', '0.6'])
    assert s['iou_threshold'] == 0.6                      # flag wins
    assert s['imgsz'] == 320 and s['layout'] == 'flat'    # from the file
    assert s['dedup'] is False and s['dedup_params'] == {'same_class_iou': 0.4}
    assert s['image_path'] == 'from_config.png'
    assert s['model_path'] == CLI_DEFAULTS['model_path']  # default

    s = resolve_cli_settings(['given.png', '--config', str(cfg), '--layout', 'xycut', '--imgsz', '640'])
    assert s['image_path'] == 'given.png' and s['layout'] == 'xycut' and s['imgsz'] == 640

    s = resolve_cli_settings(['img.png'])
    assert s == {'image_path': 'img.png', **CLI_DEFAULTS}  # everything else: run_pipeline defaults

    s = resolve_cli_settings(['img.png', '--model', 'm.pt', '--dataset-yaml', 'd.yaml',
                              '--output', 'o', '--conf', '0.2', '--iou', '0.65', '--imgsz', '480',
                              '--agnostic-nms', '--max-det', '100', '--no-dedup', '--layout', 'flat',
                              '--size-from-boxes', '--explain', '7', '--no-viz'])
    assert s == {'image_path': 'img.png', 'model_path': 'm.pt', 'dataset_yaml': 'd.yaml',
                 'output_dir': 'o', 'conf_threshold': 0.2, 'iou_threshold': 0.65, 'imgsz': 480,
                 'agnostic_nms': True, 'max_det': 100, 'dedup': False, 'layout': 'flat',
                 'size_from_boxes': True, 'explain_id': 7, 'visualize': False}
    assert set(s) - {'model_path'} <= set(inspect.signature(run_pipeline).parameters)

    cfg.write_text(yaml.safe_dump({'detection': {'agnostic_nms': True}}))
    assert resolve_cli_settings(['x.png', '--config', str(cfg),
                                 '--no-agnostic-nms'])['agnostic_nms'] is False

    with pytest.raises(SystemExit):
        resolve_cli_settings([])                          # no image anywhere
    cfg.write_text(yaml.safe_dump({'detection': {'iou': 0.5}}))
    with pytest.raises(SystemExit):
        resolve_cli_settings(['x.png', '--config', str(cfg)])
    with pytest.raises(SystemExit):
        resolve_cli_settings(['x.png', '--layout', 'grid'])


def test_cli_main_passes_settings_to_run_pipeline(tmp_path, image, monkeypatch):
    """The ui2compose command calls run_pipeline with the resolved settings."""
    model = tmp_path / "model.pt"
    model.write_bytes(b"")
    received: List[Dict] = []

    def fake_run_pipeline(**kwargs):
        """Records the settings instead of running the pipeline.

        Input:  kwargs — run_pipeline keywords
        Output: minimal result dictionary
        """
        received.append(kwargs)
        return {'output_dir': kwargs['output_dir']}

    monkeypatch.setattr(pipeline, 'run_pipeline', fake_run_pipeline)
    pipeline.main([str(image), '--model', str(model), '--output', str(tmp_path / 'o'),
                   '--iou', '0.6', '--explain', '2'])
    assert received == [{'image_path': str(image), 'model_path': str(model),
                         'dataset_yaml': CLI_DEFAULTS['dataset_yaml'],
                         'output_dir': str(tmp_path / 'o'), 'iou_threshold': 0.6,
                         'explain_id': 2}]


def _load_app_module():
    """Imports app/run_pipeline.py (a script, not part of the package) as a module.

    Input:  none
    Output: the loaded module
    """
    spec = importlib.util.spec_from_file_location("app_run_pipeline", REPO_ROOT / "app/run_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_run_pipeline_passes_the_whole_config_through(monkeypatch):
    """app/run_pipeline.py hands every config value to run_pipeline (output in a timestamped folder)."""
    app = _load_app_module()
    config = load_config(str(REPO_ROOT / "configs/pipeline_config.yaml"))
    kwargs = app.build_run_kwargs(config, "TS")
    expected = config_to_kwargs(config)
    expected['output_dir'] = f"{expected['output_dir']}/TS"
    assert kwargs == expected
    assert app.build_run_kwargs({}, "TS") == {
        'image_path': app.DEFAULT_CONFIG['input']['image'],
        'model_path': app.DEFAULT_CONFIG['input']['model'],
        'dataset_yaml': app.DEFAULT_CONFIG['input']['dataset_yaml'],
        'output_dir': 'output/TS'}

    require_reference_data(*(REPO_ROOT / rel for rel in
                             (expected['image_path'], expected['model_path'], expected['dataset_yaml'])))
    received: List[Dict] = []

    def fake_run_pipeline(**kw):
        """Records the settings instead of running the pipeline.

        Input:  kw — run_pipeline keywords
        Output: minimal result dictionary
        """
        received.append(kw)
        return {'output_dir': '/nonexistent', 'detections': [], 'raw_detections': [],
                'removed': [], 'code': ''}

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(app, 'run_pipeline', fake_run_pipeline)
    monkeypatch.setattr('builtins.input', lambda prompt='': '')
    app.main()
    assert len(received) == 1
    got = dict(received[0])
    assert got.pop('output_dir').startswith(f"{config['output']['directory']}/")
    assert got == {k: v for k, v in expected.items() if k != 'output_dir'}


def test_pipeline_module_imports_without_torch():
    """Importing the pipeline (and running its CLI helpers) must not import torch or ultralytics."""
    code = ("import sys; import ui2compose.infer.pipeline as p; "
            "p.resolve_cli_settings(['x.png']); "
            "bad = [m for m in ('torch', 'ultralytics') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
