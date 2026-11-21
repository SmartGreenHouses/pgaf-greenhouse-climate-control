"""Shared publication style for ETIFA figures.

Mirrors the conventions of the reference figures
(results/partitions/ETIFA_V1_V2_V3_temporal_partitions and
results/exclusion_2021/temporal_coverage): DejaVu Sans, no titles (the
caption carries them), bold (a)/(b) panel letters, no top/right spines,
very light grid, muted pastel-derived palette, vector + 320-dpi raster output.
"""
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Pastel fills used in the partitions figure (Train / Validation / Test).
PASTEL = {'blue': '#B3D4E8', 'amber': '#F6CA65', 'green': '#B2E0B0'}
# Darker tones of the same hues, legible as lines/markers on white.
STRONG = {'blue': '#3F7FAE', 'amber': '#D99A1E', 'green': '#4E9A4B'}
INK = '#14232B'        # text inside/near data
AXIS_INK = '#31383E'   # tick labels
MUTED = '#58616B'      # annotations
GRID = '#ECEFF2'
REFLINE = '#8A9098'

FONT = {'tick': 13, 'label': 14, 'legend': 12.5, 'panel': 16, 'annot': 12}


def apply_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12.0,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': '#666666', 'axes.linewidth': 0.6,
        'axes.labelcolor': INK, 'axes.labelsize': FONT['label'],
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True,
        'xtick.color': AXIS_INK, 'ytick.color': AXIS_INK,
        'xtick.labelsize': FONT['tick'], 'ytick.labelsize': FONT['tick'],
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'legend.frameon': False, 'legend.fontsize': FONT['legend'],
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })


def style_axes(ax, grid_axis='y'):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.tick_params(pad=5)


def panel_label(ax, letter, x=-0.09, y=1.0):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=FONT['panel'],
            fontweight='bold', va='top', ha='right', color='black')


def save(fig, folder: Path, stem: str, formats=('pdf', 'svg', 'png')):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        dest = folder / f'{stem}.{ext}'
        fig.savefig(dest, dpi=320, bbox_inches='tight', pad_inches=0.14)
        print(dest)
    plt.close(fig)
