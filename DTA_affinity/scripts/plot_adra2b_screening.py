#!/usr/bin/env python3
"""Plot screening performance for all prespecified ADRA2B controls."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/lsy/.agents/skills/figure-layout-hygiene/scripts')
from check_overlap import assert_figure_clean

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/adra2b_screening_20261026'
MODELS = ['deep', 'neighbor_prior', 'equal_blend', 'fixed_blend', 'similarity_gate', 'pure_similarity']
COLORS = {name: '#515D69' for name in MODELS}
COLORS.update(similarity_gate='#007E78', reference='#7B8186')
NAMES = ['Deep learning model', 'Neighbor predictor', 'Equal-weight fusion',
         'Fixed-weight fusion', 'SGF-DTA', 'Structural similarity']


def main():
    verified = json.loads((REPORT / 'verification.json').read_text())
    assert verified['state'] == 'passed' and verified['network_replayed']
    for name, expected in verified['files'].items():
        assert hashlib.sha256((REPORT / name).read_bytes()).hexdigest() == expected
    data = pd.read_csv(REPORT / 'summary.csv').set_index('model').loc[MODELS]
    n, positives = int(data.iloc[0]['n']), int(data.iloc[0]['actives'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.titlesize': 9, 'axes.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.subplots_adjust(left=.21, right=.98, bottom=.28, top=.78, wspace=1.3)
    for ax, metric, ref, title, xlabel, limits in zip(
        axes, ['top10_hits', 'average_precision'], [10 * positives / n, positives / n],
        ['Top-10 retrieval', 'Overall ranking'], ['Active compounds in Top-10', 'Average precision'],
        [(0, 10), (0, .5)]
    ):
        ax.barh(np.arange(len(MODELS)), data[metric], height=.5, color=[COLORS[x] for x in MODELS])
        ax.set_yticks(np.arange(len(MODELS)), NAMES)
        ax.invert_yaxis()
        ax.set_ylim(len(MODELS) - .35, -.65)
        ax.set_xlim(*limits)
        ax.set_xticks(np.linspace(*limits, 3))
        ax.axvline(ref, color=COLORS['reference'], linestyle=':', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', pad=13)
    for tag, ax in zip('ab', axes):
        ax.text(-.49, 1.12, tag, transform=ax.transAxes, fontsize=10, fontweight='bold')
    fig.text(.03, .94, 'ADRA2B candidate prioritization', fontsize=11, fontweight='bold')
    fig.text(.03, .14, f'{n:,} external candidates; {positives} active at AC50 <= 10 micromolar. Predictions average five fitted models.', fontsize=8)
    fig.text(.03, .06, 'All methods use the same external panel. Dotted lines: random expected hits (a) and active fraction (b).', fontsize=8)
    assert_figure_clean(fig, include_ticks=True)
    output = REPORT / 'figures'
    output.mkdir(exist_ok=True)
    for extension in ['svg', 'png']:
        fig.savefig(output / f'adra2b_screening.{extension}', dpi=300)
    (output / 'qa.json').write_text(json.dumps(dict(
        artist_check='passed including ticks',
        contract='Primary: Top-10 active retrieval across all six prespecified methods. Supporting: average precision over the entire fixed external panel.',
        legend='Direct method labels; dotted reference lines explained in the footer.',
        source_sha256=verified['files'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ), indent=2) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
