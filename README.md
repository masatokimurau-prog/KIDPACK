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
  and start it (NI's `nirfsg` package, installed with `pip install -e ".[daq]"`;
  resource `--sg-resource`, default `PXI2Slot3`); without them it is not touched.
- Random trigger: `--random-trigger` makes the DAQ issue a software trigger
  itself, once per `--random-trigger-interval` seconds (default 1), instead of
  waiting for the edge trigger (`--trigger-*` options cannot be combined with
  it). Use it for unbiased noise/baseline waveforms. The interval is fixed and
  drift-free; `timestamp_unix_ns` is the time the trigger was sent.
- `--backend simulator` runs without any hardware.

## Online check

`kidpack-monitor` (or `python -m kidpack.monitor`) makes `pc1` (IQ plane) and
`pc2` (I, Q and the projected waveform) for 16 events in a 4x4 grid, the same
figures as `main_singlefile` of `analysis/ana_forJPS.py`. It saves them as
`pc1.png` / `pc2.png` and opens them in windows (`plt.show()`; close the
windows to exit; `--no-show` to only save, e.g. without a display). The data is
only read, never modified.

```bash
kidpack-monitor                          # newest file under KIDPACK/data/, PNGs in the current directory
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
- The generator is driven through NI's `nirfsg` (`pip install -e ".[daq]"`) exactly
  as in the macro `iq_scan_kimura20260703.py`: for every frequency a session is
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

## Layout

```
src/kidpack/       package source (daq/: pulse DAQ, iqscan/: IQ scan,
                   monitor/: online check; rawdata.py: raw-file reader)
scripts/           stand-alone measurement scripts (VacuumGauge/read.py)
tests/             pytest tests
data/              DAQ output (created by kidpack-daq, not tracked by git)
```
