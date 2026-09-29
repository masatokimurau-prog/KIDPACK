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
- Output: `run_XX/{data/runXX-YY.npz, config/runXX-YY.yaml}` under
  `--output-dir` (default `KIDPACK/data/`, git-ignored, wherever the command is
  run from; `./data` if not run from a source checkout). No log file is
  written: progress and errors go to the console, errors also into the YAML.
  An existing run number is never reused.
- One line per run (run number, start/stop time, DAQ rate, condition) is
  appended to `run_summary.txt` (tab-separated) in `--output-dir`
  (i.e. `KIDPACK/data/run_summary.txt`, also git-ignored).
- Signal generator: give `--sg-frequency` and `--sg-power` to let the DAQ set
  and start it; without them it is not touched. It needs the site-specific
  `nirfsg` module (`PXIe_5654`); do not `pip install nirfsg`.
- Random trigger: `--random-trigger` makes the DAQ issue a software trigger
  itself, once per `--random-trigger-interval` seconds (default 1), instead of
  waiting for the edge trigger (`--trigger-*` options cannot be combined with
  it). Use it for unbiased noise/baseline waveforms. The interval is fixed and
  drift-free; `timestamp_unix_ns` is the time the trigger was sent.
- `--backend simulator` runs without any hardware.

## Online check

`kidpack-monitor` (or `python -m kidpack.monitor`) writes `pc1.png` (IQ plane)
and `pc2.png` (I, Q and the projected waveform) for 16 events in a 4x4 grid,
the same figures as `main_singlefile` of `analysis/ana_forJPS.py`. The data is
only read, never modified.

```bash
kidpack-monitor                          # newest file under KIDPACK/data/, PNGs in the current directory
kidpack-monitor path/to/run12-03.npz --output-dir /tmp/check
kidpack-monitor --rebin 4 --alpha --stride 2
```

- Events shown: the first 16 of the file (`--stride K`: every K-th; the old
  macro used 2).
- Pedestal: mean over the pre-trigger samples per event (`--alpha`: the first
  100 ns). `--rebin N` averages N samples before the peak is located.

## Layout

```
src/kidpack/       package source (daq/: config, runner, writer, backends/;
                   monitor/: online check; rawdata.py: raw-file reader)
scripts/           stand-alone measurement scripts (VacuumGauge/read.py)
tests/             pytest tests
data/              DAQ output (created by kidpack-daq, not tracked by git)
```
