#!/usr/bin/env python3
"""Publication figures for the five-period interpolation sensitivity study.

Reads the CSVs written by interpolation_sensitivity.py (no
recomputation) and draws them in the shared ETIFA style (figure_style.py).
Figures carry no titles; state scenario, units and averaging in the caption:
  - *_comparison: MAE averaged with equal weight over the five periods;
    shaded band = range (min-max) across periods; dotted line = 60-min
    production interpolation limit.
  - period_comparison_60min: per-period MAE at 60-min gaps, mean over seeds.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figure_style import (apply_style, style_axes, panel_label, save,
                          STRONG, PASTEL, MUTED, REFLINE, FONT)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'results' / 'interpolation_sensitivity_5periods'

METHOD_LABEL = {
    'magnus_observed_aux': 'Magnus (observed TBS, HR)',
    'magnus_interpolated_aux': 'Magnus (interpolated TBS, HR)',
    'circular': 'Circular interpolation',
    'linear_dh': 'Linear interpolation',
    'arithmetic_linear': 'Linear interpolation',
    'persistence_dh': 'Persistence',
    'persistence': 'Persistence',
}
# Proposed method = blue, linear baseline = amber, persistence = green.
METHOD_COLOR = {
    'magnus_observed_aux': 'blue', 'magnus_interpolated_aux': 'blue', 'circular': 'blue',
    'linear_dh': 'amber', 'arithmetic_linear': 'amber',
    'persistence_dh': 'green', 'persistence': 'green',
}
METHOD_MARKER = {'blue': 'o', 'amber': 's', 'green': '^'}

CASES = [  # stem, scenario, variable, primary, linear baseline, y label
    ('dh_only_comparison', 'dh_only', 'humidity_deficit',
     'magnus_observed_aux', 'linear_dh', 'MAE (g m$^{-3}$)'),
    ('dh_joint_comparison', 'dh_joint', 'humidity_deficit',
     'magnus_interpolated_aux', 'linear_dh', 'MAE (g m$^{-3}$)'),
    ('wind_direction_comparison', 'wind_direction', 'wind_direction',
     'circular', 'arithmetic_linear', 'MAE (°)'),
]
PRODUCTION_LIMIT_MIN = 60


def gap_plot(ax, pooled, scenario, variable, ylabel):
    sub = pooled[(pooled.scenario == scenario) & (pooled.variable == variable)]
    gaps = sorted(sub.gap_min.unique())
    for method, s in sub.groupby('method', sort=False):
        s = s.sort_values('gap_min')
        c = METHOD_COLOR[method]
        ax.fill_between(s.gap_min, s.mae_min_period, s.mae_max_period,
                        color=PASTEL[c], alpha=0.35, lw=0, zorder=1)
        ax.plot(s.gap_min, s.mae_mean_of_periods, color=STRONG[c], lw=1.8,
                marker=METHOD_MARKER[c], ms=6, mec='white', mew=0.8,
                label=METHOD_LABEL[method], zorder=3)
    ax.axvline(PRODUCTION_LIMIT_MIN, color=REFLINE, lw=0.9, ls='--', zorder=2)
    ax.text(PRODUCTION_LIMIT_MIN + 2, 0.98, 'Production limit', transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=FONT['annot'], color=MUTED)
    ax.set_xticks(gaps)
    ax.set_xlim(gaps[0] - 6, gaps[-1] + 6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Artificial gap duration (min)')
    ax.set_ylabel(ylabel)
    style_axes(ax, grid_axis='both')


def period_bars(ax, per, scenario, variable, primary, baseline, ylabel):
    sub = per[(per.scenario == scenario) & (per.variable == variable) & (per.gap_min == 60)]
    periods = list(dict.fromkeys(per.period.tolist()))
    x = np.arange(len(periods))
    w = 0.36
    for offset, method in ((-w / 2, primary), (w / 2, baseline)):
        vals = [sub[(sub.period == p) & (sub.method == method)].mae_mean_seeds.iloc[0]
                for p in periods]
        c = METHOD_COLOR[method]
        ax.bar(x + offset, vals, width=w, color=PASTEL[c], edgecolor=STRONG[c],
               linewidth=0.8, label=METHOD_LABEL[method], zorder=2)
    labels = []
    for p in periods:
        code, season, year = p.split('_')
        labels.append(f'{code}\n{season.capitalize()} {year}')
    ax.set_xticks(x, labels)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis='y')


def draw_all(per, pooled, out: Path):
    apply_style()

    for stem, scenario, variable, _, _, ylabel in CASES:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        gap_plot(ax, pooled, scenario, variable, ylabel)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncols=3,
                  handlelength=2.2, columnspacing=1.4, borderaxespad=0.2)
        save(fig, out, stem)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10.5), layout='constrained')
    fig.get_layout_engine().set(hspace=0.06)
    for ax, letter, (_, scenario, variable, primary, baseline, ylabel) in zip(axes, 'abc', CASES):
        period_bars(ax, per, scenario, variable, primary, baseline, ylabel)
        panel_label(ax, letter, x=-0.11)
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.02), ncols=2)
        ax.margins(y=0.18)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    save(fig, out, 'period_comparison_60min')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()
    pooled = pd.read_csv(args.results / 'summary_across_five_periods.csv')
    per = pd.read_csv(args.results / 'summary_by_period.csv')
    draw_all(per, pooled, args.results / 'figures')


if __name__ == '__main__':
    main()
