"""IQ スキャンのプロット (サンプルマクロ 4)

IQ スキャン (周波数を振って、ch0 と ch1 の平均を測ったもの) のファイル (.npz) から、3 つのプロットを描きます。
kidpack-iqplot (スキャンをすぐに確認するコマンド) と同じ図です。

    python 04_plot_iqscan.py  iqscan_202607221241.npz
    python 04_plot_iqscan.py  iqscan_202607221241.npz  through.npz      # 2 つ目は、基準にするスキャン (下を見る)

kidpack-iqscan が保存したファイルも、古い iq_scan.py が保存したファイルも読めます。
このファイル 1 つだけで動きます (numpy と matplotlib が必要)。
関数は使わずに、上から順に読めるように書いてあります。

Jupyter ノートブックで使うとき
------------------------------
* このファイルの中身を 1 つのセルに貼って、下の FILENAME にスキャンのファイルの名前を書いて実行する
  (基準にするスキャンを使うなら、REF_FILENAME にも)。
* または、セルに  %run 04_plot_iqscan.py  スキャン.npz  と書く。


出てくる図
----------
左   ch0 (I) と ch1 (Q) の平面。共振器の S21 は、この平面で円を描く。赤い点が最初の周波数
中   周波数と |S21|。共振のところで谷になる
右   周波数と arg S21 (位相)。共振のところで位相が大きく動く

基準のスキャンについて
----------------------
ファイルに入っているのは、ch0 と ch1 の平均 (電圧) そのままです。S21 の大きさと位相に、
ケーブルやアンプの利得・位相がかかっているので、そのままだと S21 ではありません (uncalibrated)。
基準のスキャン (同じ周波数で、共振器を通さずに測ったもの) を 2 つ目に渡すと、
S21 = スキャン / 基準 (複素数の割り算) にして描きます。


流れ
----
1. スキャンを読む
2. 基準のスキャンがあれば、割る
3. 大きさと位相を求める
4. 描く
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DB = False       # True にすると、大きさを dB (20 log10) で描く
WRAP = False     # True にすると、位相を -pi ~ pi に折りたたんで描く (False なら、つながった位相)
SAVE_PNG = None  # 'scan.png' のようにファイル名を書くと、図をそのファイルにも保存する

# ノートブックでは、ここにファイルの名前を書く (コマンドラインでは、引数で渡すので、None のまま)
FILENAME = None       # 例: 'iqscan_202607221241.npz'
REF_FILENAME = None   # 基準のスキャン。使わなければ None のまま

args = sys.argv[1:]
if args[:1] == ['-f']:       # ノートブックのセルでは、sys.argv は Jupyter の ['-f', '...json']。これはこのマクロの引数ではない
    args = []
if len(args) > 2 or (len(args) == 0 and FILENAME is None):
    print('使い方: python 04_plot_iqscan.py  スキャン.npz  [基準のスキャン.npz]')
    print('        (ノートブックでは、上の FILENAME に、基準のスキャンは REF_FILENAME に、ファイルの名前を書く)')
    sys.exit(1)
filename = args[0] if args else FILENAME
ref_filename = args[1] if len(args) == 2 else REF_FILENAME
for name_ in (filename, ref_filename):
    if name_ is not None and not os.path.exists(name_):
        sys.exit(f'ファイルが見つかりません: {name_}')


# =============================================================================
# 1. スキャンを読む
# =============================================================================
data = np.load(filename)
if 'dd' not in data.files:
    sys.exit(f'{filename}: "dd" がありません (IQ スキャンのファイルではありません)')
dd = data['dd']                             # 形は (周波数の数, 3)
if dd.ndim != 2 or dd.shape[1] != 3:
    sys.exit(f'{filename}: "dd" の形は (周波数の数, 3) のはずですが、{dd.shape} です')
frequency = dd[:, 0]                        # 周波数 [Hz]
iq = dd[:, 1] + 1j * dd[:, 2]               # ch0 (I) + i ch1 (Q)。周波数ごとの平均 [V]
power_dbm = float(data['power_dbm']) if 'power_dbm' in data.files else None   # 古いファイルにはない
name = os.path.splitext(os.path.basename(filename))[0]
print(f'{filename}: {len(frequency)} points, {frequency[0] / 1e9:.4f} - {frequency[-1] / 1e9:.4f} GHz')


# =============================================================================
# 2. 基準のスキャンがあれば、割る: S21 = スキャン / 基準
# =============================================================================
calibrated = ref_filename is not None
if calibrated:
    ref = np.load(ref_filename)
    if 'dd' not in ref.files or ref['dd'].ndim != 2 or ref['dd'].shape[1] != 3:
        sys.exit(f'{ref_filename}: IQ スキャンのファイルではありません ("dd" がないか、形が違います)')
    ref_dd = ref['dd']
    if len(ref_dd) != len(frequency) or not np.allclose(ref_dd[:, 0], frequency, rtol=1e-9, atol=0):
        sys.exit(f'基準のスキャン {ref_filename} は、{filename} と周波数が違います')
    iq = iq / (ref_dd[:, 1] + 1j * ref_dd[:, 2])
    ref_name = os.path.splitext(os.path.basename(ref_filename))[0]


# =============================================================================
# 3. 大きさと位相を求める
# =============================================================================
magnitude = np.abs(iq)                      # |S21|
phase = np.angle(iq)                        # arg S21 [rad]。-pi ~ pi に折りたたまれている
if not WRAP:
    phase = np.unwrap(phase)                # 2 pi のとびをなくして、つなげる
freq_ghz = frequency / 1e9

# 基準がないときは、電圧 [V] を mV にして描く。基準があれば、S21 (単位なし) をそのまま描く
scale, unit = (1.0, '') if calibrated else (1e3, ' [mV]')


# =============================================================================
# 4. 描く
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), layout='constrained')

# 左: IQ 平面
ax[0].plot(iq.real * scale, iq.imag * scale, 'o-', ms=3)
ax[0].plot(iq.real[0] * scale, iq.imag[0] * scale, 'o', ms=7, color='r', label='first point')
ax[0].set_xlabel(('Re S21' if calibrated else 'ch0 (I)') + unit)
ax[0].set_ylabel(('Im S21' if calibrated else 'ch1 (Q)') + unit)
ax[0].set_aspect('equal', adjustable='datalim')
ax[0].legend(loc='best')

# 中: |S21|
if DB:
    ax[1].plot(freq_ghz, 20 * np.log10(magnitude), 'o-', ms=3)
    ax[1].set_ylabel('|S21| [dB]' if calibrated else '20 log10 |IQ| [dB re 1 V]')
else:
    ax[1].plot(freq_ghz, magnitude * scale, 'o-', ms=3)
    ax[1].set_ylabel('|S21|' if calibrated else '|S21| (uncalibrated)' + unit)

# 右: arg S21
ax[2].plot(freq_ghz, phase, 'o-', ms=3)
ax[2].set_ylabel('arg S21 [rad]' + ('' if calibrated else ' (uncalibrated)'))

for a in ax[1:]:
    a.set_xlabel('Frequency [GHz]')
for a in ax:
    a.grid(True)

title = f'{name}: {len(freq_ghz)} points, {freq_ghz[0]:.4f}-{freq_ghz[-1]:.4f} GHz'
if power_dbm is not None:
    title += f', {power_dbm:g} dBm'
title += f' / ref {ref_name}' if calibrated else ' (uncalibrated: S21 x gain x cable phase)'
fig.suptitle(title)

if SAVE_PNG:
    fig.savefig(SAVE_PNG, dpi=100)
    print(f'saved: {SAVE_PNG}')
plt.show()
