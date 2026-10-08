"""KID の応答 toy MC (サンプルマクロ 1)

温度パルスで共振周波数が動くと、固定した読み出し周波数での S21 が時間とともに
どう変わるかを、理想的な S21 の式から計算して、プロットします。
パラメータを変えて、結果を並べて比べることもできます。

    python 01_kid_response_toymc.py              # 下の SETTINGS のまま実行
    python 01_kid_response_toymc.py 6.0 5.5      # パルスの温度 T_hot と定常温度 T_base を指定
                                                 # (パルスの大きさは delta_T = T_hot - T_base になる)

このファイル 1 つだけで動きます (numpy と matplotlib が必要)。
使い方: 下の 1. の値を書き換えて、もう一度実行してください。

Jupyter ノートブックで使うとき
------------------------------
* このファイルの中身を 1 つのセルに貼って実行する (図が出る)。または、セルに
      %run 01_kid_response_toymc.py
  と書く。値を変えるには、上と同じように 1. の SETTINGS を書き換えて、もう一度実行する。
* T_hot と T_base は、別のセルから main(6.0, 5.5) のように渡してもよい (コマンドラインの引数と同じ)。


やっていること
--------------
1. 温度パルス T(t) を作る:     T_base から立ち上がり (tau_rise)、元に戻る (tau_decay)
2. 共振周波数 fr(T) を決める:  温度が上がると fr は下がる
3. S21(t) を計算する:         読み出し周波数 f_readout での、理想的な notch 共振器の S21

出てくる図
----------
図 1  共振曲線と、S21 が IQ 平面で描く軌跡
図 2  温度・共振周波数・S21 の時間変化
図 3〜  COMPARISONS で指定したパラメータを変えて、応答を重ねて比較したもの
        (標準では、図 3 = 読み出し周波数、図 4 = 定常温度 T_base)


見どころ
--------
* 読み出し周波数が共振のどこにあるかで、応答の向きと大きさが変わる (図 1 と、f_readout の比較)。
* 定常温度 T_base が違うと、同じ大きさのパルス (delta_T) でも、応答が違う
  (図 4。共振の幅 fr/Ql や、共振の深さ Ql/Qc が、温度で変わるため)。
"""
import sys

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. ここを書き換えて遊ぶ
# =============================================================================
SETTINGS = {
    # --- 温度パルス ---
    'T_base': 5.5,         # [K]  パルスが来る前 (と、十分たった後) の温度
    'delta_T': 0.5,        # [K]  パルスの大きさ。T_hot = T_base + delta_T (実際のピークはこれより低い)
    'tau_rise': 50.0,      # [ns] 立ち上がりの時定数
    'tau_decay': 200.0,    # [ns] 減衰の時定数

    # --- 読み出し ---
    'f_readout': None,     # [Hz] 読み出し周波数。None なら、T_base での共振周波数に合わせる
                           #      (T_base = 5.5 K なら 5.3803 GHz)。数値を入れると、その周波数に固定する

    # --- 共振器: S21 の式のパラメータ ---
    'Ql': None,            # 負荷 Q。None なら T_base での値 Ql_T(T_base) を使う
    'Qc': None,            # 結合 Q。None なら T_base での値 Qc_T(T_base) を使う
    'phi': 0.0,            # [rad] 共振曲線の非対称性
    'a': 1.0,              # 全体の振幅
    'alpha': 0.0,          # [rad] 全体の位相
    'cable_delay': 0.0,    # [s]   ケーブルの遅延 (周波数に比例して位相が回る)

    # --- True にすると、温度で Ql, Qc も変わる (上の Ql, Qc の指定は無視される) ---
    'q_changes_with_T': False,
}

# 比べたいパラメータを (名前, 値のリスト) で書く。1 つにつき 1 枚の図ができる。
# 名前は SETTINGS のキーのどれでもよい。行を足したり消したりしてよい。
COMPARISONS = [
    ('f_readout', [5.374e9, 5.377e9, 5.380e9, 5.383e9, 5.386e9]),  # 読み出し周波数
    ('T_base', [4.5, 5.0, 5.5, 6.0, 6.5]),   # 定常温度。delta_T は同じまま、T_hot = T_base + delta_T が動く
    # ('delta_T', [0.1, 0.3, 0.5, 1.0, 1.5]),                       # パルスの大きさ
    # ('q_changes_with_T', [False, True]),                         # Q も動くと?
    # ('phi', [-0.5, 0.0, 0.5]),                                   # 共振曲線が非対称だと?
]

# 素子の性質 (温度依存性のモデル)
FR0 = 5.4e9           # [Hz] 温度が 0 K のときの共振周波数
T_CRITICAL = 9.2      # [K]  超伝導転移温度
ALPHA_KINETIC = 0.05  # 運動インダクタンス比 (大きいほど、温度で fr が大きく動く)

N_POINTS = 1000       # 時間の点の数 (1 点 = 1 ns)


# =============================================================================
# 2. 理想的な S21 と、温度依存性のモデル
# =============================================================================
def s21_notch(f, fr, Ql, Qc, phi=0.0, a=1.0, alpha=0.0, tau=0.0):
    """notch 型共振器の理想的な S21 (複素数)。f は配列でもよい。

        S21 = a e^{i alpha} e^{-2 pi i f tau} * (1 - (Ql/|Qc|) e^{i phi} / (1 + 2 i Ql (f/fr - 1)))

    f: 周波数 [Hz]   fr: 共振周波数 [Hz]   Ql: 負荷 Q   Qc: 結合 Q
    phi: 非対称性 [rad]   a: 振幅   alpha: 位相 [rad]   tau: ケーブル遅延 [s]
    """
    envelope = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * tau)
    resonance = (Ql / np.abs(Qc) * np.exp(1j * phi)) / (1 + 2j * Ql * (f / fr - 1))
    return envelope * (1 - resonance)


def fr_T(T):
    """共振周波数 [Hz]。温度 T [K] が上がると下がる。"""
    return FR0 / np.sqrt(1 - ALPHA_KINETIC + ALPHA_KINETIC / (1 - (T / T_CRITICAL) ** 4))


def Ql_T(T):
    """負荷 Q の温度依存 (この素子の測定に合わせた式)。1 より小さくはならない。"""
    value = 2.06e3 - 1.28e2 * T - 1.47e1 * T ** 2
    return np.where(value > 1.0, value, 1.0)


def Qc_T(T):
    """結合 Q の温度依存 (この素子の測定に合わせた式)。"""
    return 1.6e3 + 3.5e2 * np.log(1 + np.exp((T - 6.2) / 0.35))


def Qi_from(Ql, Qc):
    """内部 Q。 1/Ql = 1/Qi + 1/Qc より。"""
    return 1 / (1 / Ql - 1 / Qc)


# =============================================================================
# 3. シミュレーション
# =============================================================================
def simulate(T_base, delta_T, tau_rise, tau_decay, f_readout, Ql, Qc, phi, a, alpha,
             cable_delay, q_changes_with_T):
    """温度パルスに対する S21 の時間変化を計算する。結果は dict で返す。"""
    t = np.arange(N_POINTS, dtype=float)  # 時間 [ns]

    # 温度パルス: t = 0 で T_base から始まり、少し上がって、T_base に戻る
    T_hot = T_base + delta_T
    T = T_base + delta_T * (np.exp(-t / tau_decay) - np.exp(-t / tau_rise))
    if T.max() >= T_CRITICAL:
        raise ValueError(f'温度が転移温度 {T_CRITICAL} K に達します。T_base か delta_T を下げてください。')
    fr = fr_T(T)

    # 読み出し周波数: 指定がなければ、定常状態 (T_base) の共振周波数に合わせる
    if f_readout is None:
        f_readout = float(fr_T(T_base))

    # Ql, Qc: 一定にするか、温度で変えるか
    if q_changes_with_T:
        Ql_t, Qc_t = Ql_T(T), Qc_T(T)
    else:
        Ql_base = Ql_T(T_base) if Ql is None else Ql
        Qc_base = Qc_T(T_base) if Qc is None else Qc
        Ql_t, Qc_t = np.full(N_POINTS, Ql_base), np.full(N_POINTS, Qc_base)

    # 読み出し周波数での S21 (時間の配列)
    s21 = s21_notch(f_readout, fr, Ql_t, Qc_t, phi, a, alpha, cable_delay)

    return {'settings': dict(T_base=T_base, delta_T=delta_T, T_hot=T_hot, tau_rise=tau_rise,
                             tau_decay=tau_decay, f_readout=f_readout, Ql=Ql, Qc=Qc, phi=phi,
                             a=a, alpha=alpha, cable_delay=cable_delay,
                             q_changes_with_T=q_changes_with_T),
            't': t, 'T': T, 'fr': fr, 'Ql': Ql_t, 'Qc': Qc_t, 's21': s21}


def peak_response(result):
    """S21 が定常状態 (t = 0) から最も大きく動いた量。"""
    return np.max(np.abs(result['s21'] - result['s21'][0]))


# =============================================================================
# 4. プロット
# =============================================================================
def resonance_curve(fr, Ql, Qc, s, n=2000):
    """1 つの共振器の S21 を、周波数を振って計算する。 (周波数, S21) を返す。"""
    f = fr * (1 + np.linspace(-10, 10, n) / Ql)  # 共振の幅 (fr/Ql) の +-10 倍の範囲
    return f, s21_notch(f, fr, Ql, Qc, s['phi'], s['a'], s['alpha'], s['cable_delay'])


def plot_resonance_and_trajectory(result):
    """図 1: 左 = IQ 平面 (共振の円と、S21 の軌跡)、中と右 = 周波数を振った |S21| と arg S21。"""
    s, T, fr = result['settings'], result['T'], result['fr']
    i_peak = np.argmax(np.abs(T - s['T_base']))  # 温度が最も動いた時刻

    fig, ax = plt.subplots(figsize=(15, 4.8), ncols=3)

    # 共振曲線: 定常状態 (実線) と、パルスのピーク (破線)
    curves = [('T = %.2f K (base)' % T[0], 0, '-', 'tab:blue'),
              ('T = %.2f K (peak)' % T[i_peak], i_peak, '--', 'tab:red')]
    for label, i, style, color in curves:
        f, values = resonance_curve(fr[i], result['Ql'][i], result['Qc'][i], s)
        ax[0].plot(values.real, values.imag, style, color=color, alpha=0.6)
        ax[1].plot(f / 1e9, np.abs(values), style, color=color, label=label)
        ax[2].plot(f / 1e9, np.angle(values), style, color=color, label=label)
    for a_ in ax[1:]:
        a_.axvline(s['f_readout'] / 1e9, color='gray', ls=':', label='readout')

    # S21(t) の軌跡: 10 点おきに、時間で色をつける
    sc = ax[0].scatter(result['s21'][::10].real, result['s21'][::10].imag,
                       c=result['t'][::10], cmap='viridis', s=14)
    fig.colorbar(sc, ax=ax[0], label='Time [ns]')

    ax[0].set(xlabel='Re S21', ylabel='Im S21', title='IQ plane')
    ax[0].set_aspect('equal', adjustable='datalim')
    ax[1].set(xlabel='Frequency [GHz]', ylabel='|S21|', title='resonance curve')
    ax[2].set(xlabel='Frequency [GHz]', ylabel='arg S21 [rad]', title='resonance curve')
    for a_ in ax:
        a_.grid(True)
    ax[1].legend(fontsize='small')
    fig.tight_layout()
    return fig


def plot_time_response(result):
    """図 2: 温度、共振周波数、Q、S21 (実部・虚部・絶対値・位相) の時間変化。"""
    s, t, s21 = result['settings'], result['t'], result['s21']

    fig, ax = plt.subplots(figsize=(9, 10), ncols=2, nrows=4, sharex=True)
    ax[0, 0].plot(t, result['T'])
    ax[0, 0].set_ylabel('Temperature [K]')
    ax[0, 1].plot(t, result['fr'] / 1e9)
    ax[0, 1].set_ylabel('Resonant freq [GHz]')
    ax[1, 0].plot(t, result['Ql'], label='Ql')
    ax[1, 0].plot(t, result['Qc'], label='Qc')
    ax[1, 0].set_ylabel('Q')
    ax[1, 1].plot(t, Qi_from(result['Ql'], result['Qc']), label='Qi')
    ax[1, 1].set_ylabel('Qi')
    ax[2, 0].plot(t, s21.real)
    ax[2, 0].set_ylabel('S21 Real')
    ax[2, 1].plot(t, s21.imag)
    ax[2, 1].set_ylabel('S21 Imag')
    ax[3, 0].plot(t, np.abs(s21))
    ax[3, 0].set_ylabel('|S21|')
    ax[3, 1].plot(t, np.angle(s21))
    ax[3, 1].set_ylabel('arg S21 [rad]')

    for axis in ax.flat:
        axis.grid(True)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize='small')
    ax[3, 0].set_xlabel('Time [ns]')
    ax[3, 1].set_xlabel('Time [ns]')
    ax[0, 0].set_title(f"f_readout = {s['f_readout'] / 1e9:.4f} GHz")
    ax[0, 1].set_title(f"T_base = {s['T_base']:g} K, delta_T = {s['delta_T']:g} K")
    fig.tight_layout()
    return fig


def format_value(name, value):
    """図や表に出す値の文字列 (周波数は GHz にして読みやすくする)。"""
    return f'{value / 1e9:.3f} GHz' if name == 'f_readout' else f'{value:g}'


def compare(name, values, settings=None):
    """SETTINGS (settings を渡せば、それ) のうち name だけを values の各値に変えて、応答を重ねて比較する。

    name = 'T_base' のときは delta_T が SETTINGS のまま変わらないので、定常温度だけを変えた
    比較になる (T_hot = T_base + delta_T は、いっしょに動く)。
    """
    base = SETTINGS if settings is None else settings
    results = [simulate(**{**base, name: value}) for value in values]
    labels = [f'{name} = {format_value(name, value)}' for value in values]

    fig, ax = plt.subplots(figsize=(14, 8), ncols=3, nrows=2)
    ax_re, ax_im, ax_iq = ax[0]
    ax_abs, ax_arg, ax_peak = ax[1]
    for result, label in zip(results, labels):
        t, s21 = result['t'], result['s21']
        ax_re.plot(t, s21.real, label=label)
        ax_im.plot(t, s21.imag, label=label)
        ax_iq.plot(s21.real, s21.imag, label=label)
        ax_abs.plot(t, np.abs(s21), label=label)
        ax_arg.plot(t, np.angle(s21), label=label)
    ax_peak.plot(range(len(values)), [peak_response(r) for r in results], 'o-')
    ax_peak.set_xticks(range(len(values)))
    ax_peak.set_xticklabels([format_value(name, v) for v in values], rotation=30,
                            fontsize='small')

    ax_re.set(xlabel='Time [ns]', ylabel='S21 Real')
    ax_im.set(xlabel='Time [ns]', ylabel='S21 Imag')
    ax_iq.set(xlabel='Re S21', ylabel='Im S21', title='IQ plane')
    ax_iq.set_aspect('equal', adjustable='datalim')
    ax_abs.set(xlabel='Time [ns]', ylabel='|S21|')
    ax_arg.set(xlabel='Time [ns]', ylabel='arg S21 [rad]')
    ax_peak.set(xlabel=name, ylabel='max |S21(t) - S21(0)|', title='size of the response')
    for axis in ax.flat:
        axis.grid(True)
    ax_re.legend(fontsize='small')

    # 比べていない (そのまま) の設定のうち、パルスに関わるものをタイトルに書く
    fixed = [f'{key} = {base[key]:g}' for key in ('T_base', 'delta_T') if key != name]
    fig.suptitle(f'comparison of {name}' + (f'   ({", ".join(fixed)} fixed)' if fixed else ''))
    fig.tight_layout()

    # 応答の大きさを表にして表示する
    print(f'\n--- {name}: size of the response, max |S21(t) - S21(0)| ---')
    for value, result in zip(values, results):
        s = result['settings']
        print(f'{name} = {format_value(name, value):<10} T_hot = {s["T_hot"]:.2f} K   '
              f'f_readout = {s["f_readout"] / 1e9:.4f} GHz   max|dS21| = {peak_response(result):.3f}')
    return fig


# =============================================================================
# 5. 実行
# =============================================================================
def command_line_numbers():
    """コマンドラインの T_hot と T_base (python 01_kid_response_toymc.py 6.0 5.5)。数のリストで返す。

    ノートブックのセルでは、sys.argv に Jupyter 自身の引数 ['-f', '...json'] が入っている。
    それはこのマクロの引数ではないので、読まない。
    """
    args = sys.argv[1:3]
    if args[:1] == ['-f']:
        return []
    return [float(a) for a in args]


def main(T_hot=None, T_base=None):
    """図を描く。T_hot と T_base を渡すと、SETTINGS の T_base と delta_T の代わりにそれを使う。

    delta_T = T_hot - T_base (T_base を渡さなければ SETTINGS の T_base)。SETTINGS は書き換えない。
    """
    settings = dict(SETTINGS)
    if T_base is not None:
        settings['T_base'] = T_base
    if T_hot is not None:
        settings['delta_T'] = T_hot - settings['T_base']  # 比較 (compare) でも同じ値を使う

    result = simulate(**settings)
    plot_resonance_and_trajectory(result)
    plot_time_response(result)
    for name, values in COMPARISONS:
        compare(name, values, settings)
    plt.show()


if __name__ == '__main__':
    main(*command_line_numbers())
