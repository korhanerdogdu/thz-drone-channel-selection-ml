"""PRESENTATION-ONLY re-export of paper Figs. 2 and 3 (TAES revision, October 2026).

Input : fig_data.npz -- the exact arrays plotted by notebook cells 4818a43f
        (all_methods_comparison.png) and t0comb01 (ml_mobility_comparison.png).
        It was dumped from a scratch re-execution of thz_d3qn_v4.ipynb whose printed
        outputs matched the committed notebook in all 56 code cells, and whose regenerated
        PNGs were pixel-identical to the committed figures.
Output: fig_capacity_main.pdf/.png and fig_capacity_mobility.pdf/.png (new files; the
        original PNGs are untouched).

Nothing here trains, predicts, simulates or evaluates anything. Relative to the notebook
cells, only layout changes: the figure is authored at its printed size, panel tags move
to the empty upper-right corner, the elevation x-label is shortened so it is not clipped,
and the legend uses the manuscript's model names ("k-NN"). The data series, the display
clip min(prediction, MaxActive), the PCHIP log-space smoothing, the colours, the markers
and the line styles are those of the notebook.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(HERE, 'fig_data.npz'))


# ---- verbatim copies of the notebook's display helpers (cell smooth001) ----------------
def _smooth_log(x, y, n=250):
    x = np.asarray(x, float); y = np.maximum(np.asarray(y, float), 1e-12)
    o = np.argsort(x); x, y = x[o], y[o]
    keep = np.concatenate([[True], np.diff(x) > 0])
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return x, y
    xi = np.linspace(x[0], x[-1], n)
    yi = 10 ** PchipInterpolator(x, np.log10(y))(xi)
    return xi, yi


def _mk_ls(fmt):
    for ls in ('-.', '--', ':', '-'):
        if ls in fmt:
            return (fmt.replace(ls, '')[:1] or 'o'), ls
    return (fmt[:1] or 'o'), '-'


def slog(ax, x, y, color, marker='o', ls='-', lw=1.6, ms=5, label=None, alpha=1.0, zorder=3):
    y = np.maximum(np.asarray(y, float), 1e-12)
    xi, yi = _smooth_log(np.asarray(x, float), y)
    ax.semilogy(xi, yi, ls=ls, color=color, lw=lw, alpha=alpha, zorder=zorder, label=label)
    ax.semilogy(x, y, ls='none', marker=marker, color=color, ms=ms, alpha=alpha, zorder=zorder + 0.1)


# ---- notebook STYLE entries for the plotted models (cell a0528fa9) ----------------------
STYLE = {
    'Ridge':      ('#4878D0', 'P--'),
    'Huber':      ('#EE854A', '<-'),
    'KNN':        ('#D65F5F', 'v:'),
    'MLP':        ('#956CB4', 'h-.'),
    'GaussProc':  ('#7f7f7f', 'd-'),
    'ExtraTrees': ('#bcbd22', 's:'),
    'D3QN':       ('#17becf', '*-'),
}
FIG_MODELS = ['Ridge', 'Huber', 'KNN', 'MLP', 'GaussProc', 'ExtraTrees']
DISPLAY = {'KNN': '$k$-NN'}                       # manuscript naming only

# ---- printed-size typography (TAES: 8-10 pt figure text) --------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0, 'legend.fontsize': 8.0,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42,
})
LW_REF, MS_REF, LW_M, MS_M, LW_D3, MS_D3 = 2.0, 3.6, 1.15, 2.8, 1.3, 4.6
FS_TAG = 8.5


def _decades(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', ls='--', lw=0.4, alpha=0.45)


def _tag(ax, s):
    ax.text(0.97, 0.94, s, transform=ax.transAxes, fontsize=FS_TAG, fontweight='bold',
            va='top', ha='right',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))


def _legend_handles():
    h = [Line2D([], [], color='k', ls='-', marker='o', lw=LW_REF, ms=MS_REF,
                label='MaxActive (reference)')]
    for n in FIG_MODELS:
        c, mk = STYLE[n]; mkc, lsc = _mk_ls(mk)
        h.append(Line2D([], [], color=c, ls=lsc, marker=mkc, lw=LW_M, ms=MS_M,
                        label=DISPLAY.get(n, n)))
    h.append(Line2D([], [], color=STYLE['D3QN'][0], ls='-', marker='*', lw=LW_D3,
                    ms=MS_D3, label='D3QN'))
    return h


def _plot_models(ax, x, y_ma, preds, y_d3=None):
    slog(ax, x, np.maximum(y_ma, 1e-12), color='k', marker='o', ls='-',
         lw=LW_REF, ms=MS_REF, zorder=6)
    for m in FIG_MODELS:
        y_pred = np.minimum(preds[m], y_ma)            # display clip, as in the notebook
        c, mk = STYLE[m]; mkc, lsc = _mk_ls(mk)
        slog(ax, x, np.maximum(y_pred, 1e-12), color=c, marker=mkc, ls=lsc,
             lw=LW_M, ms=MS_M, alpha=0.9)
    if y_d3 is not None:
        y_d3 = np.minimum(y_d3, y_ma)
        slog(ax, x, np.maximum(y_d3, 1e-12), color=STYLE['D3QN'][0], marker='*', ls='-',
             lw=LW_D3, ms=MS_D3, alpha=0.95, zorder=7)


def _save(fig, stem):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'{stem}.{ext}'), dpi=600)
    plt.close(fig)


# ---- Fig. 2: aligned / diagonal scenarios, WF and EP ------------------------------------
def fig_main():
    x = D['dist_axis']
    fig, axes = plt.subplots(3, 2, figsize=(4.9, 5.35))
    tags = [['(a)', '(b)'], ['(c)', '(d)'], ['(e)', '(f)']]
    for r, sc in enumerate(['aligned', 'diag_noAdj', 'diag_Adj']):
        for c, alloc in enumerate(['WF', 'EP']):
            ax = axes[r, c]
            y_ma = D[f'main/{sc}/MA_{alloc}']
            preds = {m: D[f'main/{sc}/{m}/{alloc}'] for m in FIG_MODELS}
            _plot_models(ax, x, y_ma, preds, D[f'main/{sc}/D3QN'] if alloc == 'EP' else None)
            ax.set_xticks(x[::2])
            ax.set_xlim(x[0] - 3, x[-1] + 3)
            _decades(ax)
            if r == 2:
                ax.set_xlabel('Distance $d$ [m]', labelpad=1.0)
            else:
                ax.tick_params(labelbottom=True)
            if c == 0:
                ax.set_ylabel('Capacity [Gbps]', labelpad=1.5)
            ax.tick_params(pad=1.5)
            _tag(ax, tags[r][c])
    fig.legend(handles=_legend_handles(), loc='lower center', ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=2.3,
               handletextpad=0.4, borderpad=0.35, labelspacing=0.3, fancybox=False,
               edgecolor='0.7')
    fig.tight_layout(rect=[0.0, 0.085, 1.0, 1.0], h_pad=0.6, w_pad=0.9)
    _save(fig, 'fig_capacity_main')


# ---- Fig. 3: elevation and real-trace mobility, EP ---------------------------------------
def fig_mobility():
    fig, axes = plt.subplots(2, 2, figsize=(4.9, 3.75))
    comb = [[('elev_noAdj', '(a)'), ('elev_Adj', '(b)')],
            [('real_noAdj', '(c)'), ('real_Adj', '(d)')]]
    xlabels = [r'Zenith angle $\phi$ [deg]', 'Time $t$ [s]']
    for r, row in enumerate(comb):
        for c, (sc, tag) in enumerate(row):
            ax = axes[r, c]
            x = D[f'mob/{sc}/axis']
            preds = {m: D[f'mob/{sc}/{m}/EP'] for m in FIG_MODELS}
            _plot_models(ax, x, D[f'mob/{sc}/MA_EP'], preds, D[f'mob/{sc}/D3QN'])
            if r == 0:
                ax.set_xticks(np.arange(0, 181, 45))
            _decades(ax)
            ax.set_xlabel(xlabels[r], labelpad=1.0)
            if c == 0:
                ax.set_ylabel('Capacity [Gbps]', labelpad=1.5)
            ax.tick_params(pad=1.5)
            _tag(ax, tag)
    fig.legend(handles=_legend_handles(), loc='lower center', ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0, handlelength=2.3,
               handletextpad=0.4, borderpad=0.35, labelspacing=0.3, fancybox=False,
               edgecolor='0.7')
    fig.tight_layout(rect=[0.0, 0.115, 1.0, 1.0], h_pad=0.6, w_pad=0.9)
    _save(fig, 'fig_capacity_mobility')


if __name__ == '__main__':
    fig_main()
    fig_mobility()
    print('written: fig_capacity_main.{pdf,png}, fig_capacity_mobility.{pdf,png}')
