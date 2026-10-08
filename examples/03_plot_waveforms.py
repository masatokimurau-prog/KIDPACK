"""波形のプロット (サンプルマクロ 3)

DAQ が保存した波形ファイル (.npz) から、最初の 16 イベントを 4 x 4 のマスに並べて、2 種類の図を描きます。
kidpack-monitor (測定中の確認用のコマンド) と同じ図です。

    python 03_plot_waveforms.py  run01-00.npz

このファイル 1 つだけで動きます (numpy と matplotlib が必要)。
関数は使わずに、上から順に読めるように書いてあります。

Jupyter ノートブックで使うとき
------------------------------
* このファイルの中身を 1 つのセルに貼って、下の FILENAME に波形ファイルの名前を書いて実行する。
* または、セルに  %run 03_plot_waveforms.py  波形ファイル.npz  と書く。


出てくる図
----------
図 1 (pc1)  IQ 平面。マスが 1 イベント。I (ch0) と Q (ch1) の点を、時間で色をつけて描く。
            赤い x = ペデスタル (トリガーの前の平均)、赤い * = ピーク (proj max)
図 2 (pc2)  波形。I と Q (どちらもペデスタルを引く前) と、Proj (ペデスタルを引いて、
            ピークの向きを実軸にそろえた波形)。赤い * = ピークのときの I と Q

ピークは、|(I - I のペデスタル) + i (Q - Q のペデスタル)| が最大のところです。
図 1 の赤い * は、図 2 の 2 つの赤い * と同じ点です。


流れ
----
1. 波形を読む
2. 描くイベントを選ぶ              最初の 16 個 (STRIDE を大きくすると、とびとびに選ぶ)
3. ペデスタルを求める              トリガーより前の平均 (I と Q それぞれ)
4. Rebin                           REBIN サンプルずつ平均する
5. ピークを探す                    ペデスタルを引いた |I + iQ| が最大のところ
6. 向きをそろえる (proj)           ピークの向きが実軸になるように回す
7. 図 1: IQ 平面
8. 図 2: 波形
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REBIN = 5        # 5 サンプルずつ平均する (ピークを探すのも、図に描く点も、この平均)
STRIDE = 1       # 何イベントおきに描くか。1 なら最初の 16 個、2 なら 0, 2, 4, ... 番目の 16 個
SAVE = False     # True にすると、図を pc1.png と pc2.png として、今いる場所にも保存する
NROWS, NCOLS = 4, 4      # マスの数 (16 イベント)

# ノートブックでは、ここに波形ファイルの名前を書く (コマンドラインでは、引数で渡すので、None のまま)
FILENAME = None   # 例: 'run01-00.npz'

args = sys.argv[1:]
if args[:1] == ['-f']:       # ノートブックのセルでは、sys.argv は Jupyter の ['-f', '...json']。これはこのマクロの引数ではない
    args = []
if len(args) > 1 or (len(args) == 0 and FILENAME is None):
    print('使い方: python 03_plot_waveforms.py  波形ファイル.npz')
    print('        (ノートブックでは、上の FILENAME に波形ファイルの名前を書く)')
    sys.exit(1)
filename = args[0] if args else FILENAME
if not os.path.exists(filename):
    sys.exit(f'ファイルが見つかりません: {filename}')
if REBIN < 1 or STRIDE < 1:
    sys.exit('REBIN と STRIDE は 1 以上にしてください')


# =============================================================================
# 1. 波形を読む
# =============================================================================
data = np.load(filename)
v0_all = data['ch0']                        # ch0 (I) の波形 [V]。形は (イベント数, サンプル数)
v1_all = data['ch1']                        # ch1 (Q) の波形 [V]
npts = int(data['npts'])                    # 1 イベントのサンプル数
sample_rate = float(data['sample_rate'])    # サンプリングレート [Hz]
ref_position = float(data['ref_position'])  # トリガーの位置 [%] (波形の先頭からどれだけ進んだところか)
if v0_all.shape[0] == 0:
    sys.exit('イベントがありません')
if 'event_id' in data.files:                # DAQ が付けたイベント番号 (古いファイルにはない)
    event_id_all = data['event_id']
else:
    event_id_all = np.arange(v0_all.shape[0])
print(f'{filename}: {v0_all.shape[0]} events, {npts} samples, {sample_rate / 1e9:g} GS/s')


# =============================================================================
# 2. 描くイベントを選ぶ
# =============================================================================
indices = np.arange(0, v0_all.shape[0], STRIDE)[:NROWS * NCOLS]   # 選んだイベントの番号
nplot = len(indices)                        # 描くイベントの数 (16 より少ないこともある)
v0 = v0_all[indices]                        # 選んだイベントだけ取り出す
v1 = v1_all[indices]
event_id = event_id_all[indices]
print(f'events {event_id[0]} ... {event_id[-1]} ({nplot} events), rebin = {REBIN}')


# =============================================================================
# 3. ペデスタルを求める
# =============================================================================
ped_end = int(npts * ref_position / 100)    # トリガーの位置 = トリガーより前のサンプル数
if ped_end < 1:
    sys.exit('トリガーより前のサンプルがないので、ペデスタルが計算できません (ref_position = 0)')

ped0 = v0[:, :ped_end].mean(axis=1)         # 波形の先頭からトリガーまでの平均 (イベントごと)
ped1 = v1[:, :ped_end].mean(axis=1)


# =============================================================================
# 4. Rebin: REBIN サンプルずつ平均する
# =============================================================================
nbin = npts // REBIN                        # 平均したあとのサンプル数 (bin 数)
i_bin = v0[:, :nbin * REBIN].reshape(nplot, nbin, REBIN).mean(axis=2)   # I (ペデスタルを引く前)
q_bin = v1[:, :nbin * REBIN].reshape(nplot, nbin, REBIN).mean(axis=2)   # Q (ペデスタルを引く前)

# 各 bin の時刻 [us] (トリガーが 0)
time_us = (np.arange(nbin) - nbin * ref_position / 100.) / (sample_rate / REBIN) * 1e6


# =============================================================================
# 5. ピークを探す: ペデスタルを引いた |I + iQ| が最大のところ
# =============================================================================
vc = (i_bin - ped0.reshape(-1, 1)) + 1j * (q_bin - ped1.reshape(-1, 1))   # ペデスタルを引いた I + iQ
peak_bin = np.argmax(np.abs(vc), axis=1)    # ピークがある bin の番号 (イベントごと)


# =============================================================================
# 6. 向きをそろえる (proj): ピークの向きが実軸 (プラス) になるように回す
# =============================================================================
theta = np.angle(vc[np.arange(nplot), peak_bin])          # ピークのときの I + iQ の向き [rad]
proj = np.real(vc * np.exp(-1j * theta.reshape(-1, 1)))   # 回した波形の実部 (実数)


# =============================================================================
# 7. 図 1: IQ 平面
# =============================================================================
fig1, ax1 = plt.subplots(NROWS, NCOLS, figsize=(14, 12), sharex=True, sharey=True,
                         squeeze=False, layout='constrained')
for k in range(NROWS * NCOLS):
    a = ax1.flat[k]
    if k >= nplot:
        a.axis('off')                       # イベントが 16 個ない: 余ったマスは空にする
        continue
    p = peak_bin[k]                         # このイベントのピークの bin
    image = a.scatter(i_bin[k] * 1e3, q_bin[k] * 1e3, c=time_us, cmap='viridis', s=8)   # V を mV にして描く
    a.plot(ped0[k] * 1e3, ped1[k] * 1e3, 'x', ms=9, color='r', label='ped')
    a.plot(i_bin[k, p] * 1e3, q_bin[k, p] * 1e3, '*', ms=10, color='r', label='proj max')
    if k + NCOLS >= nplot:                  # 下にマスがない列だけ、x 軸の名前をつける
        a.set_xlabel('I [mV]')
    if k % NCOLS == 0:                      # 左端のマスだけ、y 軸の名前をつける
        a.set_ylabel('Q [mV]')
    a.set_title(f'Event #{event_id[k]}')
    a.grid(True)
ax1.flat[0].legend(loc='best')
fig1.colorbar(image, ax=ax1, label='time [µs]', shrink=0.6)


# =============================================================================
# 8. 図 2: 波形
# =============================================================================
fig2, ax2 = plt.subplots(NROWS, NCOLS, figsize=(14, 12), sharex=True, sharey=True,
                         squeeze=False, layout='constrained')
for k in range(NROWS * NCOLS):
    a = ax2.flat[k]
    if k >= nplot:
        a.axis('off')
        continue
    p = peak_bin[k]
    a.plot(time_us, i_bin[k] * 1e3, label='I')
    a.plot(time_us, q_bin[k] * 1e3, label='Q')
    a.plot(time_us, proj[k] * 1e3, label='Proj')
    a.plot(time_us[p], i_bin[k, p] * 1e3, '*', ms=10, color='r', label='proj max')
    a.plot(time_us[p], q_bin[k, p] * 1e3, '*', ms=10, color='r', label='_nolegend_')   # 凡例には 1 つだけ
    if k + NCOLS >= nplot:
        a.set_xlabel('time [µs]')
    if k % NCOLS == 0:
        a.set_ylabel('voltage [mV]')
    a.set_title(f'Event #{event_id[k]}')
    a.grid(True)
ax2.flat[0].legend(loc='best')

if SAVE:
    fig1.savefig('pc1.png', dpi=100)
    fig2.savefig('pc2.png', dpi=100)
    print('saved: pc1.png, pc2.png')
plt.show()
