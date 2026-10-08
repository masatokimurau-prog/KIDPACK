"""Make sample_data/ from the position scans (Aug 31) of the old DAQ macro.

The old files (wf_260831_*.npz of ``analysis/ana_forJPS.py::main_compare_distance``: the
z-scans at x = 4.0 and 3.5 mm and the x-scan at z = 6.30 mm) are converted to the current
raw-data format, one run per file. A run is named after the start in the file name:
wf_260831_155253_49.55Hz.npz -> run_0831_155253. A file that is in two scans (see SCANS) is
converted once and gets both in its condition. The old files are only read.

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

# (name, axis that is scanned, [(glob pattern under --old-data-dir, position [mm]), ...])
SCANS = [
    ('z-scan at x = 4.0 mm', 'z', [
        ('Aug31st/wf_260831_15554?_*.npz', 6.90),
        ('Aug31st/wf_260831_15562?_*.npz', 6.70),
        ('Aug31st/wf_260831_15570?_*.npz', 6.50),
        ('Aug31st/wf_260831_15575?_*.npz', 6.30),
        ('Aug31st/wf_260831_15413?_*.npz', 6.15),
        ('Aug31st/wf_260831_15421?_*.npz', 6.00),
        ('Aug31st/wf_260831_15425?_*.npz', 5.85),
        ('Aug31st/wf_260831_15434?_*.npz', 5.70),
        ('Aug31st/wf_260831_16212?_*.npz', 5.50),
    ]),
    ('z-scan at x = 3.5 mm', 'z', [
        ('Aug31st/wf_260831_15474?_*.npz', 6.90),
        ('Aug31st/wf_260831_15482?_*.npz', 6.70),
        ('Aug31st/wf_260831_15490?_*.npz', 6.50),
        ('Aug31st/wf_260831_15494?_*.npz', 6.30),
        ('Aug31st/wf_260831_15502?_*.npz', 6.00),
        ('Aug31st/wf_260831_15514?_*.npz', 5.70),
        ('Aug31st/wf_260831_16203?_*.npz', 5.50),
    ]),
    # Not in ana_forJPS.py with a heading. It is an x-scan at z = 6.30 mm because its 3.50 mm
    # file (15494?) is the 6.30 mm file of the z-scan at x = 3.5 mm.
    ('x-scan at z = 6.30 mm', 'x', [
        ('Aug31st/wf_260831_15525?_*.npz', 3.20),
        ('Aug31st/wf_260831_15494?_*.npz', 3.50),
        ('Aug31st/wf_260831_15454?_*.npz', 3.65),
        ('Aug31st/wf_260831_15385?_*.npz', 4.00),
        ('Aug31st/wf_260831_15542?_*.npz', 4.50),
    ]),
]
def condition_text(name, axis, position):
    return f'{name}, {axis} = {position:.2f} mm'


def find_files(old_data_dir):
    """[(path, condition), ...]: every file of the scans once; each pattern must match exactly one file."""
    conditions = {}  # path -> its condition for each scan it is in
    for name, axis, entries in SCANS:
        for pattern, position in entries:
            matches = sorted(glob.glob(os.path.join(old_data_dir, pattern)))
            if len(matches) != 1:
                raise SystemExit(f'error: {pattern!r} ({name}, {axis} = {position:.2f} mm) matches '
                                 f'{len(matches)} files under {old_data_dir}, expected exactly 1')
            conditions.setdefault(matches[0], []).append(condition_text(name, axis, position))
    return [(path, '; '.join(texts)) for path, texts in conditions.items()]


def readme_text(converted):
    rows = []
    for run in converted:
        start = datetime.fromtimestamp(run.start_ns / 1e9, JST).strftime('%H:%M:%S')
        rows.append(f'| {run.label} | {run.condition} | {start} | {run.nevents} | `{run.source}` |')
    table = '\n'.join(rows)
    first = converted[0].label
    megabytes = sum(os.path.getsize(run.npz_path) for run in converted) / 1e6
    scans = '\n'.join(
        f'- {name}: {axis} = {min(p for _, p in entries):.2f} ... {max(p for _, p in entries):.2f} mm, '
        f'{len(entries)} 点' for name, axis, entries in SCANS)
    return f"""# sample_data

2026-08-31 に古い DAQ マクロ (`kid.py`) で取った、検出器の位置 (x, z) を変えた測定 {len(converted)} 点を、
現在の DAQ (`kidpack-daq`) と同じ形式に直したものです。
解析や `kidpack-monitor` の練習に使えます。元のファイルは変更していません。
`scripts/make_sample_data.py` で作りました。

{scans}

x = 3.5 mm の z スキャンの z = 6.30 mm と、z = 6.30 mm の x スキャンの x = 3.50 mm は同じファイルです
(run が 1 つで、条件が 2 つ書いてあります)。

| run | 測定条件 | 開始時刻 (JST) | 事象数 | 元のファイル (`data/Aug31st/` 内) |
|---|---|---|---|---|
{table}

run の名前は、元のファイル名にある開始時刻 (月日_時分秒) です。
`wf_260831_155253_49.55Hz.npz` は `run_0831_155253/data/run0831_155253-00.npz` になります (年は付けていません)。
表は開始時刻の順です (位置の順ではありません)。
1 run = 1 ファイル (`run月日_時分秒-00.npz`)、
1 ファイル 1000 事象 × 5000 点 (2.5 GS/s、トリガー位置は記録の 20 %)、1 ファイル約 40 MB、全部で約 {megabytes:.0f} MB です。

```
sample_data/
  run_summary.txt                 1 run が 1 行 (run の名前、開始・終了時刻、DAQ レート、condition)
  run_0831_153858/data/run0831_153858-00.npz        波形
  run_0831_153858/config/run0831_153858-00.yaml     そのファイルの設定・時刻 (わかっていることだけ)
  ...
```

## 使い方

```bash
kidpack-monitor --data-dir sample_data --run-number {first}     # pc1 / pc2 の図
python examples/02_pulse_analysis.py sample_data/run_{first}/data/run{first}-00.npz
```

```python
from kidpack.rawdata import load_raw
raw = load_raw('sample_data/run_{first}/data/run{first}-00.npz')
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

    converted = convert_old_files(find_files(args.old_data_dir), args.output_dir,
                                  source_root=os.path.join(args.old_data_dir, 'Aug31st'))

    readme = os.path.join(args.output_dir, 'README.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(readme_text(converted))
    for run in converted:
        print(f'run {run.label}  {run.nevents} events  {run.source}  ->  '
              f'{os.path.relpath(run.npz_path)}   [{run.condition}]')
    print(f'wrote {readme}')


if __name__ == '__main__':
    sys.exit(main())
