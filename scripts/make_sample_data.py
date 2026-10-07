"""Make sample_data/ from the position scans (Aug 31) of the old DAQ macro.

The old files (wf_260831_*.npz of ``analysis/ana_forJPS.py::main_compare_distance``: the
z-scans at x = 4.0 and 3.5 mm and the x-scan at z = 6.30 mm) are converted to the current
raw-data format, one run per file, in the order of their start times. A file that is in two
scans (see SCANS) is converted once and gets both in its condition. The old files are only read.

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
from kidpack.runs import run_dir_name

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
# Indices into SCANS: the runs of one batch are numbered in the order of the start times, the
# batches one after the other. (The first batch is the first nine runs of this dataset.)
BATCHES = [[0], [1, 2]]


def condition_text(name, axis, position):
    return f'{name}, {axis} = {position:.2f} mm'


def find_batches(old_data_dir):
    """[(path, condition), ...] for every batch; each pattern must match exactly one file."""
    batches, batch_of = [], {}
    for number, indices in enumerate(BATCHES):
        conditions = {}  # path -> its condition for each scan it is in
        for i in indices:
            name, axis, entries = SCANS[i]
            for pattern, position in entries:
                matches = sorted(glob.glob(os.path.join(old_data_dir, pattern)))
                if len(matches) != 1:
                    raise SystemExit(f'error: {pattern!r} ({name}, {axis} = {position:.2f} mm) matches '
                                     f'{len(matches)} files under {old_data_dir}, expected exactly 1')
                path = matches[0]
                if batch_of.setdefault(path, number) != number:
                    raise SystemExit(f'error: {path} is in scans of two batches')
                conditions.setdefault(path, []).append(condition_text(name, axis, position))
        batches.append([(path, '; '.join(texts)) for path, texts in conditions.items()])
    return batches


def run_ranges(batch_sizes):
    """'run 1-9: z-scan at x = 4.0 mm; run 10-20: ...': which scans the runs of each batch are in."""
    parts, first = [], 1
    for indices, size in zip(BATCHES, batch_sizes):
        parts.append(f"run {first}-{first + size - 1}: {', '.join(SCANS[i][0] for i in indices)}")
        first += size
    return '; '.join(parts)


def readme_text(converted, batch_sizes):
    rows = []
    for run in converted:
        start = datetime.fromtimestamp(run.start_ns / 1e9, JST).strftime('%H:%M:%S')
        rows.append(f'| {run.run_number} | {run.condition} | {start} | {run.nevents} | `{run.source}` |')
    table = '\n'.join(rows)
    first = converted[0].run_number
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
(run 1 つで、条件が 2 つ書いてあります)。

| run | 測定条件 | 開始時刻 (JST) | 事象数 | 元のファイル (`data/Aug31st/` 内) |
|---|---|---|---|---|
{table}

run 番号は、{run_ranges(batch_sizes)} の順に、それぞれ開始時刻の順に付けてあります (位置の順ではありません)。
1 run = 1 ファイル (`runXX-00.npz`)、
1 ファイル 1000 事象 × 5000 点 (2.5 GS/s、トリガー位置は記録の 20 %)、1 ファイル約 40 MB、全部で約 {megabytes:.0f} MB です。

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

    batches = find_batches(args.old_data_dir)
    total = sum(len(items) for items in batches)
    for number in range(1, total + 1):  # check everything before writing anything
        directory = os.path.join(args.output_dir, run_dir_name(number))
        if os.path.exists(directory):
            raise FileExistsError(f'run directory already exists: {directory}')

    converted = []
    for items in batches:
        converted += convert_old_files(items, args.output_dir, first_run_number=len(converted) + 1,
                                      source_root=os.path.join(args.old_data_dir, 'Aug31st'))

    readme = os.path.join(args.output_dir, 'README.md')
    with open(readme, 'w', encoding='utf-8') as f:
        f.write(readme_text(converted, [len(items) for items in batches]))
    for run in converted:
        print(f'run {run.run_number:2d}  {run.nevents} events  {run.source}  ->  '
              f'{os.path.relpath(run.npz_path)}   [{run.condition}]')
    print(f'wrote {readme}')


if __name__ == '__main__':
    sys.exit(main())
