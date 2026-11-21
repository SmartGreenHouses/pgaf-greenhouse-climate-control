#!/usr/bin/env python3
"""Reproducible, publication-ready visualization of ETIFA V1/V2/V3 temporal partitions.

Inputs are the audited, inclusive 15-minute partition boundaries and counts.
Each span is plotted to (last timestamp + 15 minutes) to depict adjacent intervals.
All percentages use the 88,745 *unique* timestamps assigned in each configuration\n(state this in the figure caption; the figure carries no title or footnote).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Patch

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'partitions'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL = 88745
SPLITS = {
    'V1': [
        ('Train', '2019-04-29 07:30', '2021-02-08 23:45', 62562),
        ('Validation', '2021-02-09 00:00', '2021-06-27 17:30', 13319),
        ('Test', '2021-11-01 00:00', '2022-03-14 23:45', 12864),
    ],
    'V2': [
        ('Train', '2019-04-29 07:30', '2021-02-08 23:45', 62562),
        ('Test', '2021-02-09 00:00', '2021-06-27 17:30', 13319),
        ('Validation', '2021-11-01 00:00', '2022-03-14 23:45', 12864),
    ],
    'V3': [
        ('Test', '2019-04-29 07:30', '2019-09-10 07:15', 12864),
        ('Train', '2019-09-10 07:30', '2021-06-27 17:30', 63017),
        ('Validation', '2021-11-01 00:00', '2022-03-14 23:45', 12864),
    ],
}
COLORS = {'Train': '#B3D4E8', 'Validation': '#F6CA65', 'Test': '#B2E0B0'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.0,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.edgecolor': '#666666', 'axes.linewidth': 0.6,
    'savefig.facecolor': 'white',
})
fig, ax = plt.subplots(figsize=(14.5, 4.6))
starts = [pd.Timestamp('2019-04-29'), pd.Timestamp('2022-03-16')]
ax.set_xlim(*[mdates.date2num(x) for x in starts])
ax.set_ylim(-0.44, 2.62)
for y, (name, parts) in zip([2, 1, 0], SPLITS.items()):
    # validate disjoint periods and source counts before visualizing
    assert sum(p[3] for p in parts) == TOTAL, name
    for role, begin, end, n in parts:
        start = pd.Timestamp(begin)
        end_exclusive = pd.Timestamp(end) + pd.Timedelta(minutes=15)
        expected_n = int((end_exclusive-start)/pd.Timedelta(minutes=15))
        assert n == expected_n, (name, role, n, expected_n)
        x0, x1 = mdates.date2num(start), mdates.date2num(end_exclusive)
        rect = Rectangle((x0, y-0.29), x1-x0, 0.58, facecolor=COLORS[role],
                         edgecolor='white', linewidth=0.8, zorder=2)
        ax.add_patch(rect)
        ax.text((x0+x1)/2, y, f'{role}\n{n/TOTAL*100:.1f}%', fontsize=14,
                ha='center', va='center', color='#14232B', linespacing=1.3,
                zorder=3)
    ax.text(mdates.date2num(pd.Timestamp('2019-04-11')), y, name,
            ha='right', va='center', fontsize=15, fontweight='bold')

# Mark the intentional exclusion (rather than presenting it as a missing horizontal-axis segment).
cut1 = mdates.date2num(pd.Timestamp('2021-06-27 17:45'))
cut2 = mdates.date2num(pd.Timestamp('2021-11-01'))
ax.add_patch(Rectangle((cut1, -0.33), cut2-cut1, 2.66, facecolor='#F0F1F3', lw=0, zorder=0))
ax.text((cut1 + cut2)/2, 2.47, 'Excluded period', ha='center', va='center',
        color='#58616B', fontsize=13)
for y in [2,1,0]:
    ax.hlines(y, cut1+8, cut2-8, color='#B2B6BD', lw=0.8, ls='--', zorder=1)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4, bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.grid(axis='x', color='#ECEFF2', linewidth=0.6, zorder=0)
ax.set_yticks([])
ax.tick_params(axis='x', which='major', labelsize=13, length=3, pad=5, colors='#31383E')
for spine in ('top','left','right'): ax.spines[spine].set_visible(False)
for extension in ('pdf','svg','png'):
    dest=OUTPUT_DIR / f'ETIFA_V1_V2_V3_temporal_partitions.{extension}'
    fig.savefig(dest, dpi=320, bbox_inches='tight', pad_inches=0.14)
    print(dest)
plt.close(fig)
