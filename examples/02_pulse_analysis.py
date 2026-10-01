"""パルス解析のサンプルマクロ (サンプルマクロ 2)

DAQ が保存した波形ファイル (.npz) を読んで、イベントごとに下の量を計算し、
最後に ch0_ped と ch1_ped のヒストグラムを描きます。

    python 02_pulse_analysis.py  wf_260901_173154_49.70Hz.npz

このファイル 1 つだけで動きます (numpy と matplotlib が必要)。
関数は使わずに、上から順に読めるように書いてあります。


計算する量 (イベントごとに 1 つ)
---------------------------------
ch0_ped, ch1_ped     ペデスタル。トリガーより前の部分の平均 [V]
proj_max             ピークの大きさ [V]。ペデスタルを引いた I + iQ の絶対値の最大値
proj_max_t           ピークの時刻 [us]。トリガーの時刻が 0
proj_half_max_left   ピークから、立ち上がりで半分の高さになるところまでの時間 [us]
proj_half_max_right  ピークから、減衰で半分の高さになるところまでの時間 [us] (なければ nan)
proj_integ           波形 (proj) の、トリガーから 1 us 分の和 [V x サンプル数]

"proj" とは: ピークのときの I + iQ の向きを実軸 (プラス) に合わせるように回した波形です。
パルスの大きさだけが残り、パルスが IQ 平面のどの向きに出ても、同じように扱えます。


流れ
----
1. 波形を読む
2. ペデスタルを引く         I + iQ を作って、トリガーより前の平均を引く
3. Rebin                   5 サンプルずつ平均して、ノイズを減らす
4. ピークを探す             |I + iQ| が最大のところ
5. 向きをそろえる (proj)    ピークの向きが実軸になるように回す
6. 半値幅と和を求める
7. 表を保存して、ヒストグラムを描く
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

REBIN = 5  # 5 サンプルずつ平均する (この値で固定)

if len(sys.argv) != 2:
    print('使い方: python 02_pulse_analysis.py  波形ファイル.npz')
    sys.exit(1)
filename = sys.argv[1]
if not os.path.exists(filename):
    sys.exit(f'ファイルが見つかりません: {filename}')


# =============================================================================
# 1. 波形を読む
# =============================================================================
data = np.load(filename)
v0 = data['ch0']                            # ch0 (I) の波形 [V]。形は (イベント数, サンプル数)
v1 = data['ch1']                            # ch1 (Q) の波形 [V]
npts = int(data['npts'])                    # 1 イベントのサンプル数
sample_rate = float(data['sample_rate'])    # サンプリングレート [Hz]
ref_position = float(data['ref_position'])  # トリガーの位置 [%] (波形の先頭からどれだけ進んだところか)
nwf = v0.shape[0]                           # イベント数
print(f'{filename}: {nwf} events, {npts} samples, {sample_rate / 1e9:g} GS/s')


# =============================================================================
# 2. ペデスタルを引く
# =============================================================================
ped_end = int(npts * ref_position / 100)    # トリガーの位置 = トリガーより前のサンプル数
if ped_end < 1:
    sys.exit('トリガーより前のサンプルがないので、ペデスタルが計算できません (ref_position = 0)')

ch0_ped = v0[:, :ped_end].mean(axis=1)      # 波形の先頭からトリガーまでの平均 (イベントごと)
ch1_ped = v1[:, :ped_end].mean(axis=1)

vc = v0 + 1j * v1                           # I + iQ (複素数の波形)
vc = vc - ch0_ped.reshape(-1, 1) - 1j * ch1_ped.reshape(-1, 1)   # ペデスタルを引く


# =============================================================================
# 3. Rebin: REBIN サンプルずつ平均する
# =============================================================================
nbin = npts // REBIN                        # 平均したあとのサンプル数 (bin 数)
vc = vc[:, :nbin * REBIN].reshape(nwf, nbin, REBIN).mean(axis=2)

sample_rate_bin = sample_rate / REBIN       # 平均したあとの、1 秒あたりの bin 数
# 各 bin の時刻 [us] (トリガーが 0)
time_us = (np.arange(nbin) - nbin * ref_position / 100.) / sample_rate_bin * 1e6


# =============================================================================
# 4. ピークを探す: |I + iQ| が最大のところ
# =============================================================================
peak_bin = np.argmax(np.abs(vc), axis=1)    # ピークがある bin の番号 (イベントごと)
proj_max = np.max(np.abs(vc), axis=1)       # ピークの大きさ [V]
proj_max_t = time_us[peak_bin]              # ピークの時刻 [us]


# =============================================================================
# 5. 向きをそろえる (proj): ピークの向きが実軸 (プラス) になるように回す
# =============================================================================
theta = np.angle(vc[np.arange(nwf), peak_bin])       # ピークのときの I + iQ の向き [rad]
proj = np.real(vc * np.exp(-1j * theta.reshape(-1, 1)))   # 回した波形の実部 (実数)


# =============================================================================
# 6. 半値幅と和を求める
# =============================================================================
# ピークの半分の高さを横切るところを、イベントごとに探す。
# bin と bin のあいだは直線でつないで (線形補間)、bin の間の値まで求める。
proj_half_max_left = np.full(nwf, np.nan)   # 見つからなければ nan のまま
proj_half_max_right = np.full(nwf, np.nan)

for i in range(nwf):
    y = proj[i]                             # このイベントの波形
    k = peak_bin[i]                         # ピークの bin
    half = proj_max[i] / 2                  # 半分の高さ

    # --- 左 (立ち上がり): 波形の先頭から見て、最初に half 以上になるところ ---
    above = np.where(y[:k + 1] >= half)[0]
    j = above[0]                            # half 以上になった最初の bin
    if j == 0:
        crossing = 0.0
    elif y[j] == y[j - 1]:
        crossing = float(j)
    else:                                   # bin j-1 と j の間の、ちょうど half になるところ
        frac = (half - y[j - 1]) / (y[j] - y[j - 1])
        crossing = (j - 1) + frac
    proj_half_max_left[i] = k - crossing    # ピークまでの距離 [bin]

    # --- 右 (減衰): ピークから後ろで、最初に half より小さくなるところ ---
    below = np.where(y[k:] < half)[0]
    if len(below) == 0:
        continue                            # 最後まで half を下回らない: nan のまま
    j = below[0]                            # half より小さくなった最初の bin (ピークから数えた番号)
    if j == 0:
        crossing = 0.0
    elif y[k + j] == y[k + j - 1]:
        crossing = float(j)
    else:
        frac = (half - y[k + j - 1]) / (y[k + j] - y[k + j - 1])
        crossing = (j - 1) + frac
    proj_half_max_right[i] = crossing       # ピークからの距離 [bin]

# bin の数を時間 [us] に直す
proj_half_max_left = proj_half_max_left / sample_rate_bin * 1e6
proj_half_max_right = proj_half_max_right / sample_rate_bin * 1e6

# proj の和: トリガーから 1 us 後まで。bin の和に REBIN をかけて、元のサンプルでの和にする
# (こうすると REBIN を変えても、値がほとんど変わらない)
bin_1us = ped_end + int(1000e-9 * sample_rate)       # トリガーから 1 us 後のサンプル番号
proj_integ = proj[:, ped_end // REBIN:bin_1us // REBIN].sum(axis=1) * REBIN


# =============================================================================
# 7. 表を保存して、ヒストグラムを描く
# =============================================================================
names = ['proj_half_max_left', 'proj_half_max_right', 'proj_max', 'proj_max_t',
         'proj_integ', 'ch0_ped', 'ch1_ped']
table = np.column_stack([proj_half_max_left, proj_half_max_right, proj_max, proj_max_t,
                         proj_integ, ch0_ped, ch1_ped])

csv_name = os.path.splitext(os.path.basename(filename))[0] + '_ana.csv'   # 今いる場所に保存する
np.savetxt(csv_name, table, delimiter=',', header=','.join(names), comments='', fmt='%.8g')
print(f'saved: {csv_name}')

print(''.join(f'{name:>22}' for name in names))       # 最初の 5 イベントを表示
for row in table[:5]:
    print(''.join(f'{value:22.6g}' for value in row))

print(f'ch0_ped: mean = {ch0_ped.mean() * 1e3:.4f} mV, std = {ch0_ped.std() * 1e3:.4f} mV')
print(f'ch1_ped: mean = {ch1_ped.mean() * 1e3:.4f} mV, std = {ch1_ped.std() * 1e3:.4f} mV')

fig, ax = plt.subplots(figsize=(10, 4), ncols=2)
ax[0].hist(ch0_ped * 1e3, bins=50)          # V を mV にして描く
ax[0].set_xlabel('ch0_ped [mV]')
ax[1].hist(ch1_ped * 1e3, bins=50)
ax[1].set_xlabel('ch1_ped [mV]')
for a in ax:
    a.set_ylabel('Events')
    a.grid(True)
fig.tight_layout()
plt.show()
