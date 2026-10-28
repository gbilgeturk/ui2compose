# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Consistency checks of the release metadata, entry points, CI workflow and README commands.

The checks read the files as text (and Python sources with ast), so they need
none of the heavy dependencies; modules that import ultralytics at module
level (the v1.1.0 metric scripts) are checked without being imported.
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict

import yaml

import ui2compose

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')


def _console_scripts() -> Dict[str, str]:
    """Reads the [project.scripts] table of pyproject.toml.

    Input:  none
    Output: dictionary command name -> 'module:function'
    """
    block = PYPROJECT.split('[project.scripts]', 1)[1].split('\n[', 1)[0]
    return dict(re.findall(r'^([\w-]+)\s*=\s*"([\w.:]+)"', block, re.MULTILINE))


def test_version_is_the_same_everywhere():
    """pyproject.toml, ui2compose.__version__, .zenodo.json and the newest CHANGELOG entry agree."""
    version = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT, re.MULTILINE).group(1)
    assert version == ui2compose.__version__ == '1.2.0'
    zenodo = json.loads((REPO_ROOT / '.zenodo.json').read_text(encoding='utf-8'))
    assert zenodo['version'] == version
    changelog = (REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    assert re.search(r'^## \[([^\]]+)\]', changelog, re.MULTILINE).group(1) == version


def test_every_console_script_points_to_a_main_function():
    """Every entry point names an existing module with a top-level function of that name."""
    scripts = _console_scripts()
    for name in ('ui2compose', 'ui2compose-codegen', 'ui2compose-app-split', 'ui2compose-train',
                 'ui2compose-evaluate-detector', 'ui2compose-benchmark'):
        assert name in scripts, name
    for name, target in scripts.items():
        module, func = target.split(':')
        base = REPO_ROOT / 'src' / Path(*module.split('.'))
        path = base.with_suffix('.py') if base.with_suffix('.py').exists() else base / '__init__.py'
        assert path.exists(), f"{name}: {path} missing"
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert func in functions, f"{name}: {module} has no function {func}"


def test_readme_commands_exist():
    """Every ui2compose-* command used in the README is a declared entry point."""
    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    used = set(re.findall(r'\b(ui2compose-[a-z-]+[a-z])\b', readme))
    assert used and used <= set(_console_scripts())
    assert 'actions/workflows/ci.yml/badge.svg' in readme


def test_ci_workflow_runs_pytest_without_torch_on_three_pythons():
    """The CI workflow parses, runs on push and pull requests, and tests Python 3.10-3.12 with the light install."""
    workflow = yaml.safe_load((REPO_ROOT / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8'))
    triggers = workflow.get('on', workflow.get(True))   # YAML 1.1 reads the key `on` as True
    assert set(triggers) == {'push', 'pull_request'}
    job = workflow['jobs']['test']
    assert job['runs-on'] == 'ubuntu-latest'
    assert job['strategy']['matrix']['python-version'] == ['3.10', '3.11', '3.12']
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'pip install numpy pillow pyyaml pytest' in commands
    assert 'pip install -e . --no-deps' in commands
    assert 'pytest -q' in commands
    assert 'torch' not in commands and 'ultralytics' not in commands


def test_ci_requires_the_reference_data():
    """CI sets UI2COMPOSE_REQUIRE_REFERENCE_DATA=1, so missing reference data fails instead of skipping."""
    workflow = yaml.safe_load((REPO_ROOT / '.github' / 'workflows' / 'ci.yml').read_text(encoding='utf-8'))
    run_steps = [step for step in workflow['jobs']['test']['steps'] if 'pytest -q' in step.get('run', '')]
    assert len(run_steps) == 1
    assert run_steps[0]['env']['UI2COMPOSE_REQUIRE_REFERENCE_DATA'] == '1'


def test_missing_reference_data_skips_locally_and_fails_when_required(tmp_path, monkeypatch):
    """require_reference_data skips without the variable and fails with UI2COMPOSE_REQUIRE_REFERENCE_DATA=1."""
    import pytest

    from _reference_data import REQUIRE_ENV, require_reference_data
    missing = tmp_path / 'absent.json'
    empty = tmp_path / 'empty_dir'
    empty.mkdir()
    present = tmp_path / 'present.json'
    present.write_text('{}')
    monkeypatch.delenv(REQUIRE_ENV, raising=False)
    require_reference_data(present)                                  # nothing happens
    with pytest.raises(pytest.skip.Exception, match='absent.json'):
        require_reference_data(present, missing)
    with pytest.raises(pytest.skip.Exception, match='empty_dir'):
        require_reference_data(empty)
    monkeypatch.setenv(REQUIRE_ENV, '1')
    with pytest.raises(pytest.fail.Exception, match='absent.json'):
        require_reference_data(missing)
    require_reference_data(present)


def test_no_unresolved_placeholders_in_configs_and_docs():
    """No 'AUTHORS TO CONFIRM' placeholder (or 'believed to be') remains in configs, README or CHANGELOG."""
    files = sorted((REPO_ROOT / 'configs').glob('*.yaml')) + [REPO_ROOT / 'README.md', REPO_ROOT / 'CHANGELOG.md']
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert 'AUTHORS TO CONFIRM' not in text and 'believed to be' not in text, path.name


def test_license_statements_cover_the_bundled_material():
    """LICENSE is MIT, sources carry the two SPDX holders, and .zenodo.json names the third-party licenses.

    The holders named on the LICENSE copyright line are the authors' decision and are not asserted here.
    """
    license_text = (REPO_ROOT / 'LICENSE').read_text(encoding='utf-8')
    assert license_text.startswith('MIT License')
    holders = set()
    for path in (REPO_ROOT / 'src').rglob('*.py'):
        holders |= set(re.findall(r'SPDX-FileCopyrightText: \d{4} ([^<]+?) <', path.read_text(encoding='utf-8')))
    assert holders == {'Murat Saran', 'Göktürk Bilgetürk'}
    notes = json.loads((REPO_ROOT / '.zenodo.json').read_text(encoding='utf-8'))['notes']
    for needle in ('MIT', 'AGPL-3.0', 'CC BY 4.0', 'Apache-2.0', 'gradle-wrapper.jar'):
        assert needle in notes, needle
    readme = (REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    license_section = readme.split('## License', 1)[1]
    for needle in ('MIT License', 'AGPL-3.0', 'CC BY 4.0', 'Apache License 2.0', 'harness/android/gradlew'):
        assert needle in license_section, needle
    for wrapper in ('gradlew', 'gradlew.bat'):
        assert 'SPDX-License-Identifier: Apache-2.0' in (REPO_ROOT / 'harness' / 'android' / wrapper).read_text(encoding='utf-8')
