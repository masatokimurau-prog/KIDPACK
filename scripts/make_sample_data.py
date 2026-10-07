"""Make sample_data/ from the z-scan files (Aug 31) of the old DAQ macro.

The nine old files (wf_260831_*.npz of ``analysis/ana_forJPS.py::main_compare_distance``,
"z-scan at x = 4.0 mm") are converted to the current raw-data format, one run each, in the
order of their start times. The old files are only read.

    python scripts/make_sample_data.py                     # ../data/Aug31st -> sample_data/
    python scripts/make_sample_data.py --old-data-dir /path/to/data --output-dir /tmp/sample

Needs kidpack to be installed (``pip install -e .``). Runs that already exist in the
output directory are never overwritten: remove sample_data/ first to make it again.
"""
import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

from kidpack.legacy import JST, convert_old_files

ROOT = Path(__file__).resolve().parents[1]

# (glob pattern under --old-data-dir, z position [mm]): the list of main_compare_distance
SCANS = [
    ('Aug31st/wf_260831_15554?_*.npz', 6.90),
    ('Aug31st/wf_260831_15562?_*.npz', 6.70),
    ('Aug31st/wf_260831_15570?_*.npz', 6.50),
    ('Aug31st/wf_260831_15575?_*.npz', 6.30),
    ('Aug31st/wf_260831_15413?_*.npz', 6.15),
    ('Aug31st/wf_260831_15421?_*.npz', 6.00),
    ('Aug31st/wf_260831_15425?_*.npz', 5.85),
    ('Aug31st/wf_260831_15434?_*.npz', 5.70),
    ('Aug31st/wf_260831_16212?_*.npz', 5.50),
]


def condition_text(z_mm):
    return f'z-scan at x = 4.0 mm, z = {z_mm:.2f} mm'


def find_files(old_data_dir):
    """(path, z) of every scan; each pattern must match exactly one file."""
    items = []
    for pattern, z_mm in SCANS:
        matches = sorted(glob.glob(os.path.join(old_data_dir, pattern)))
        if len(matches) != 1:
            raise SystemExit(f'error: {pattern!r} (z = {z_mm:.2f} mm) matches {len(matches)} files '
                             f'under {old_data_dir}, expected exactly 1')
        items.append((matches[0], z_mm))
    return items


def readme_text(converted, z_of):
    rows = []
    for run in converted:
        start = datetime.fromtimestamp(run.start_ns / 1e9, JST).strftime('%H:%M:%S')
        rows.append(f'| {run.run_number} | {z_of[run.run_number]:.2f} | {start} | {run.nevents} | '
                    f'`{run.source}` |')
    table = '\n'.join(rows)
    first = converted[0].run_number
    return f"""# sample_data

2026-08-31 に古い DAQ マクロ (`kid.py`) で取った **z スキャン** (x = 4.0 mm 固定、z を 6.90 mm から
5.50 mm まで変えた 9 点) を、現在の DAQ (`kidpack-daq`) と同じ形式に直したものです。
解析や `kidpack-monitor` の練習に使えます。元のファイルは変更していません。
`scripts/make_sample_data.py` で作りました。

| run | z [mm] | 開始時刻 (JST) | 事象数 | 元のファイル (`data/Aug31st/` 内) |
|---|---|---|---|---|
{table}

run 番号は開始時刻の順です (z の順ではありません)。1 run = 1 ファイル (`runXX-00.npz`)、
1 ファイル 1000 事象 × 5000 点 (2.5 GS/s、トリガー位置は記録の 20 %)、1 ファイル約 40 MB です。

```
sample_data/
  run_summary.txt                 1 run が 1 行 (run 番号、開始・終了時刻、DAQ レート、condition)
  run_01/data/run01-00.npz        波形
  run_01/config/run01-00.yaml     そのファイルの設定・時刻 (わかっていることだけ)
  ...
```

## 使い方

```bash
kidpack-monitor --data-dir sample_data --run-number {first}              # pc1 / pc2 の図
python examples/02_pulse_analysis.py sample_data/run_{first:02d}/data/run{first:02d}-00.npz
```

```python
from kidpack.rawdata import load_raw
raw = load_raw('sample_data/run_{first:02d}/data/run{first:02d}-00.npz')
raw.ch0, raw.ch1, raw.timestamp_unix_ns
```

## 古い形式からの変更点

- `ch0`, `ch1` (V), `npts`, `sample_rate` は元のまま。`ref_position` は `float64` になった。
- `event_id` (0 から) と `run_start_unix_ns` を追加した。
- `timestamp_unix_ns` は、元の `deltat` (DAQ 開始からの経過時間) を、ファイル名の開始時刻に足して作った。
  波形を取り出した直後の PC の時刻、という意味は現在の DAQ (エッジトリガー) と同じ。
- `deltat` と `daq_rate` は npz には入れていない (DAQ レートは YAML と `run_summary.txt` にある)。

## 時刻について (注意)

- 元のファイルには DAQ 開始時刻がファイル名にしか残っていません。しかも秒までです (小数部は捨てられている)。
  また、ファイル名は DAQ PC の現地時刻なので、**日本時間 (UTC+9) だと仮定**して UTC にしました。
  よって、**絶対時刻は ±1 秒程度の不確かさ**があります。事象どうしの時間差は正確です (マイクロ秒)。
- トリガーやチャンネルの設定 (レンジ、結合など) は元のファイルに記録がないため、YAML では `unknown` にしてあります。
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--old-data-dir', default=str(ROOT.parent / 'data'),
                        help='directory that contains Aug31st/ (default: %(default)s)')
    parser.add_argument('--output-dir', default=str(ROOT / 'sample_data'),
                        help='where the runs are written (default: %(default)s)')
    args = parser.parse_args(argv)

    items = find_files(args.old_data_dir)
    z_of_file = {os.path.basename(path): z_mm for path, z_mm in items}
    converted = convert_old_files([(path, condition_text(z_mm)) for path, z_mm in items],
                                  args.output_dir, first_run_number=1,
                                  source_root=os.path.join(args.old_data_dir, 'Aug31st'))
    z_of = {run.run_number: z_of_file[os.path.basename(run.source)] for run in converted}

    readme = os.path.join(args.output_dir, 'README.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(readme_text(converted, z_of))
    for run in converted:
        print(f'run {run.run_number:2d}  z = {z_of[run.run_number]:.2f} mm  {run.nevents} events  '
              f'{run.source}  ->  {os.path.relpath(run.npz_path)}')
    print(f'wrote {readme}')


if __name__ == '__main__':
    sys.exit(main())
