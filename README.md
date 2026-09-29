# KIDPACK

Analysis tools for Kinetic Inductance Detectors (KID).

Independent project living under `KID/`; it has its own git repository.

## Install (development)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Layout

```
src/kidpack/   package source
tests/         pytest tests
```
