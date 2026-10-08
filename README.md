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
(`kidpack-daq --help`); the defaults are the old values, except the trigger
coupling, which is `dc` (it was `LF_REJECT`).

```bash
pip install -e ".[daq]"     # on the DAQ PC (adds only the niscope binding; see below)

kidpack-daq --events-per-file 1000 --num-files 5 \
    --condition "temp 5.5K (pid); lna 1.9V 13.7mA; sg 5.490GHz -12dBm"
kidpack-daq --run-number 12 --events-per-file 1000 --num-files 5   # a number of your own
kidpack-daq --run-number test --events-per-file 100 --num-files 1  # a scratch run, replaced next time
```

- `--events-per-file` and `--num-files` are required. Ctrl-C stops early and
  keeps the data acquired so far (exit code 130).
- Run number: without `--run-number` the next free number is used (one more than
  the highest `run_NN` directory or run-summary entry; 1 for the first run). An
  existing number is refused. `--run-number test` is the exception: it writes
  to `run_test/` and replaces the previous test run (only DAQ output is ever
  deleted, and only after the instruments are ready); test runs do not count
  as numbers, and the summary keeps one line per test run.
- Output: `run_XX/{data/runXX-YY.npz, config/runXX-YY.yaml}` under
  `--output-dir` (default `KIDPACK/data/`, git-ignored, wherever the command is
  run from; `./data` if not run from a source checkout). No log file is
  written: progress and errors go to the console, errors also into the YAML.
  An existing run number is never reused.
- One line per run (run number, start/stop time, DAQ rate, condition) is
  appended to `run_summary.txt` (tab-separated) in `--output-dir`
  (i.e. `KIDPACK/data/run_summary.txt`, also git-ignored).
- Signal generator: give `--sg-frequency` and `--sg-power` to let the DAQ set
  and start it (through NI's `nirfsg`; resource `--sg-resource`, default
  `PXI2Slot3`); without them it is not touched.
- Random trigger: `--random-trigger` makes the DAQ issue a software trigger
  itself, once per `--random-trigger-interval` seconds (default 1), instead of
  waiting for the edge trigger (`--trigger-*` options cannot be combined with
  it). Use it for unbiased noise/baseline waveforms. The interval is fixed and
  drift-free; `timestamp_unix_ns` is the time the trigger was sent.
- `--backend simulator` runs without any hardware.
- `nirfsg` (NI's package, plus the NI-RFSG driver) is **not** installed by
  kidpack: it must already be in the Python environment, so that the installed
  version is never installed over or upgraded. `pip install -e ".[daq]"` only adds
  `niscope`; if that is installed too, plain `pip install -e .` is enough.

## Online check

`kidpack-monitor` (or `python -m kidpack.monitor`) makes `pc1` (IQ plane) and
`pc2` (I, Q and the projected waveform) for 16 events in a 4x4 grid, the same
figures as `main_singlefile` of `analysis/ana_forJPS.py`. It saves them as
`pc1.png` / `pc2.png` and opens them in windows (`plt.show()`; close the
windows to exit; `--no-show` to only save, e.g. without a display). The data is
only read, never modified.

```bash
kidpack-monitor                          # newest file under KIDPACK/data/, PNGs in the current directory
kidpack-monitor --run-number 12 --file-number 3   # run_12/data/run12-03.npz, no path needed
kidpack-monitor --run-number 12          # the file with the highest file number of run 12
kidpack-monitor --run-number test --file-number 0
kidpack-monitor --data-dir sample_data --run-number 0831_155253   # a run named after its start (converted old data)
kidpack-monitor path/to/run12-03.npz --output-dir /tmp/check --no-show
kidpack-monitor --rebin 1 --alpha --stride 2   # no smoothing before the peak search
```

- Events shown: the first 16 of the file (`--stride K`: every K-th; the old
  macro used 2).
- Pedestal: mean over the pre-trigger samples per event and channel
  (`--alpha`: the first 100 ns). It is subtracted from ch0 and ch1, the result
  is rebinned (`--rebin N`, default 5 samples), and the peak is the bin where
  |(ch0 - ped0) + i (ch1 - ped1)| is largest (`proj max`).
- pc1 points: with `--rebin` 5 or more (default 5) they are exactly the rebinned
  samples the peak is searched in, so the red star is one of the drawn points;
  with a smaller `--rebin` they are averaged further to at least 5 samples per
  point (`--rebin 1` gives the 5-sample averages of the old macro), so the star
  can lie outside the cloud. The red stars mark the peak: in pc1 at (I, Q); in
  pc2 as the ch0 and ch1 values at the peak time, drawn on the (raw, pedestal
  not subtracted) I and Q curves. The green Proj curve, the I/Q rotated onto the
  peak phase, peaks at `proj max`.

## IQ scan

`kidpack-iqscan` (or `python -m kidpack.iqscan`) measures the transmission versus
frequency: at each frequency the signal generator is set and started, the
digitizer acquires `--num-records` records immediately (no trigger), the
generator is stopped, and the mean I (ch0) and Q (ch1) over all samples are
stored. The measurement and the defaults (-2 dBm, 1 s records at 10 kS/s, 2
records, +-1 V range, 1 Mohm input, SG `PXI2Slot3`) are those of the macro
`iq_scan_kimura20260703.py`; every value hard-coded there is an option
(`kidpack-iqscan --help`). Only the frequency range is required.

```bash
kidpack-iqscan --f-start 5.213e9 --f-stop 5.313e9 --num-points 101 --power -2 \
    --condition "temp 5.5K (pid); lna 1.9V"
kidpack-iqscan --f-start 4.414e9 --f-stop 4.514e9 --num-points 101 --power -30 --name T5.5K_-30dBm
```

- Output in `--output-dir` (default `KIDPACK/data/iqscan/`, git-ignored):
  `iqscan_YYYYMMDDHHMM.npz` (or `--name`), a `.yaml` with the settings, times
  and status, and one line per scan in `iqscan_summary.txt`. An existing file is
  never overwritten.
- `dd` is the old layout, so `plot_iq_scan.py` and the scripts in `analysis/`
  read the files unchanged: columns frequency [Hz], mean ch0 [V], mean ch1 [V].
  Extra keys: `ch0_std`/`ch1_std`, `ch0_stderr`/`ch1_stderr` (std / sqrt(n),
  independent samples assumed), `n_samples`, per-point `timestamp_unix_ns`,
  `power_dbm`, ... Values are as measured: no calibration or normalisation.
- Ctrl-C stops the scan, keeps the points measured so far (`status:
  interrupted`) and always stops the generator. `--settle-time S` waits after
  starting the generator at each point (the old script had that commented out).
- The generator is driven through NI's `nirfsg` exactly as in the macro `iq_scan_kimura20260703.py`: for every frequency a session is
  opened with `id_query=True, reset_device=True`, set to CW, started, and
  aborted and closed after the acquisition; the driver session's lock is
  replaced by a no-op, as the macro does. `--backend simulator` runs without
  hardware (a notch resonator in the middle of the scan range).

### Quick look at a scan

`kidpack-iqplot` (or `python -m kidpack.iqscan.plot`) shows three plots of a
scan in a window: ch0 vs ch1, frequency vs |S21| and frequency vs arg S21. It
reads the files of `kidpack-iqscan` and the old `iq_scan.py` files and never
modifies them.

```bash
kidpack-iqplot                                   # newest file under KIDPACK/data/iqscan/
kidpack-iqplot path/to/scan.npz --ref path/to/through.npz --db
kidpack-iqplot scan.npz --no-show --save scan.png    # no window, only a PNG
```

- Without `--ref` the stored values are shown as they are (mV) and labelled
  *uncalibrated*: they are S21 times the unknown gain and cable phase.
  `--ref` divides by a reference scan at the same frequencies (S21 = scan /
  ref, as `plot_iq_scan.py` does; the two must have the same frequency grid).
- The phase is shown continuously (`--wrap` folds it into -pi..pi); `--db`
  shows the magnitude in dB; `--save PNG` also writes the figure.

## Examples for students

`examples/` has short, stand-alone macros (one file each, only numpy and
matplotlib; no kidpack needed) to hand out; see `examples/README.md`.
`01_kid_response_toymc.py` computes the response of a KID from the ideal S21 of a
notch resonator and compares it for different parameters; `02_pulse_analysis.py`
analyses a waveform file (pedestal, peak, half-max times, integral; rebin 5) and
histograms the pedestals; `03_plot_waveforms.py` draws the figures of `kidpack-monitor`
(IQ plane and waveforms of 16 events) and `04_plot_iqscan.py` those of `kidpack-iqplot`
(an IQ scan: IQ plane, |S21| and phase). All of them also run in a Jupyter notebook.

## Sample data

`sample_data/` (git-ignored, 800 MB) holds 20 runs of 2026-08-31 taken with the old DAQ macro and
converted to the current raw-data format (`kidpack.legacy`): the z-scans at x = 4.0 mm (9 points)
and x = 3.5 mm (7 points) and the x-scan at z = 6.30 mm (5 points; its x = 3.50 mm file is the
z = 6.30 mm file of the other z-scan, so it is one run). It is made from the old files, which are
only read:

```bash
python scripts/make_sample_data.py    # ../data/Aug31st/ -> sample_data/ (--old-data-dir, --output-dir)
kidpack-monitor --data-dir sample_data --run-number 0831_153858
```

A converted run is named after the start in the old file name, so that it can be traced back:
`wf_260831_155253_49.55Hz.npz` becomes `run_0831_155253/data/run0831_155253-00.npz` (run name
`MMDD_HHMMSS`, no year). The DAQ itself still writes numbers and `test`; the monitor takes all three.
The start of each run is known only from the old file name (to the second, taken as UTC+9), so
absolute times are good to +-1 s; times between events are exact. `sample_data/README.md` (written by
the script) has the table of runs and conditions and the list of changes.

## All options

[`docs/OPTIONS.md`](docs/OPTIONS.md) lists every option of `kidpack-daq`, `kidpack-monitor`,
`kidpack-iqscan` and `kidpack-iqplot` with its allowed values, default and meaning, plus
the rules between options. It is generated from the real argument parsers
(`python -m kidpack.optionsdoc --output docs/OPTIONS.md`) and a test fails if it is out
of date or if an allowed value is wrong, so it can be trusted; `<command> --help` shows the
same information in the terminal.

## Layout

```
src/kidpack/       package source (daq/: pulse DAQ, iqscan/: IQ scan,
                   monitor/: online check; rawdata.py: raw-file reader;
                   legacy.py: converter of old DAQ-macro files)
scripts/           stand-alone scripts (VacuumGauge/read.py, make_sample_data.py)
examples/          sample macros for students (stand-alone, see examples/README.md)
sample_data/       old z-scan data in the current format (made by scripts/make_sample_data.py, not tracked)
docs/              OPTIONS.md: all command-line options (generated)
tests/             pytest tests
data/              DAQ output (created by kidpack-daq, not tracked by git)
```
