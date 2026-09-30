"""What `pip install kidpack[...]` may pull in."""
from pathlib import Path

import pytest

tomllib = pytest.importorskip('tomllib')  # Python 3.11+

PYPROJECT = Path(__file__).resolve().parents[1] / 'pyproject.toml'


def all_requirements():
    project = tomllib.loads(PYPROJECT.read_text())['project']
    requirements = list(project['dependencies'])
    for extra in project.get('optional-dependencies', {}).values():
        requirements += extra
    return requirements


def test_nirfsg_is_never_installed_by_kidpack():
    # The DAQ PC already has NI's nirfsg; kidpack must not install, pin or upgrade it.
    names = [r.split('[')[0].split('>')[0].split('<')[0].split('=')[0].split(';')[0].strip().lower()
             for r in all_requirements()]
    assert 'nirfsg' not in names


def test_the_console_scripts_are_registered():
    scripts = tomllib.loads(PYPROJECT.read_text())['project']['scripts']
    assert set(scripts) == {'kidpack-daq', 'kidpack-monitor', 'kidpack-iqscan', 'kidpack-iqplot'}
