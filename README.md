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

## DAQ

`kidpack-daq` (or `python -m kidpack.daq`) acquires waveforms from the NI-SCOPE
digitizer. Every setting that was hard-coded in the old `kid.py` is an option
(`kidpack-daq --help`); the defaults reproduce the old values.

```bash
pip install -e ".[daq]"     # on the DAQ PC (adds the niscope bindings)

kidpack-daq --run-number 12 --events-per-file 1000 --num-files 5 \
    --condition "temp 5.5K (pid); lna 1.9V 13.7mA; sg 5.490GHz -12dBm"
```

- `--run-number`, `--events-per-file` and `--num-files` are required. Ctrl-C
  stops early and keeps the data acquired so far (exit code 130).
- Output: `run_XX/{data/runXX-YY.npz, config/runXX-YY.yaml, logs/daq.log}`
  under `--output-dir`. An existing run number is never reused.
- One line per run (run number, start/stop time, DAQ rate, condition) is
  appended to `run_summary.txt` (tab-separated) in `--output-dir`.
- Signal generator: give `--sg-frequency` and `--sg-power` to let the DAQ set
  and start it; without them it is not touched. It needs the site-specific
  `nirfsg` module (`PXIe_5654`); do not `pip install nirfsg`.
- `--backend simulator` runs without any hardware.

## Layout

```
src/kidpack/       package source (daq/: config, runner, writer, backends/)
scripts/           stand-alone measurement scripts (VacuumGauge/read.py)
tests/             pytest tests
```
