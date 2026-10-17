"""Build Figure 6 from the packaged class-label tables.

Run with Python + pandas + matplotlib + Pillow. Inputs/outputs are package-relative.
No PCA, nearest-reference assignment, scientific model, or robustness analysis is run.
"""
from pathlib import Path
import hashlib
import json
import platform
import sys
from io import BytesIO

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Patch
from PIL import Image
import PIL

ROOT = Path(__file__).resolve().parents[4]
FINAL = ROOT / 'outputs' / 'figure_06_rebuild'
PROV = FINAL
SOURCES = {
    'MICRO': ROOT / 'data' / 'derived' / 'figure_06' / 'Figure06_MICRO_pastoral_human_class_labels.csv',
    'MACRO': ROOT / 'data' / 'derived' / 'figure_06' / 'Figure06_MACRO_pastoral_human_class_labels.csv',
}
CLASSES = ['GC', 'LFC', 'HFC', 'PSC']
EXPECTED = {'MICRO': {'GC': 8, 'LFC': 0, 'HFC': 30, 'PSC': 4},
            'MACRO': {'GC': 28, 'LFC': 0, 'HFC': 10, 'PSC': 4}}
COLORS = {'GC': '#2ca02c', 'LFC': '#1f77b4', 'HFC': '#f28e2b', 'PSC': '#d62728'}
INK = {'GC': 'black', 'LFC': 'white', 'HFC': 'black', 'PSC': 'white'}
DPI = 300
WIDTH_PX = round(180 / 25.4 * DPI)
HEIGHT_PX = round(150 / 25.4 * DPI)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_class(x):
    if pd.isna(x):
        return 'UNK'
    x = str(x).strip().upper()
    return x.replace('DIST_TO_', '').replace('NEAREST_', '').replace('CLASS_', '')


def shorten_pastoral_label(x):
    x = str(x).strip().lower()
    mapping = {
        'afr_fulani_instrument': 'Fulani inst.',
        'afr_fulani_mixed': 'Fulani mixed',
        'afr_fulani_voice': 'Fulani vocal',
        'alpine_kuhreihen_instrument': 'Kuhreihen inst.',
        'alpine_kuhreihen_mixed': 'Kuhreihen mixed',
        'br_aboio_mixed': 'Aboio mixed',
        'br_aboio_voice': 'Aboio vocal',
        'cove_candle_night_chants_voice': 'Candle-night chants',
        'cove_cattle_driving_songs_mixed': 'Driving songs mixed',
        'cove_cattle_driving_songs_voice': 'Driving songs vocal',
        'cove_cattle_herding_chants_voice': 'Herding chants',
        'cove_milking_mixed': 'Milking mixed',
        'cove_milking_vocal': 'Milking vocal',
        'cove_pastoral_song_mixed': 'Pastoral song mixed',
        'cove_pastoral_song_voice': 'Pastoral song vocal',
        'sc_kulning_voice': 'Kulning',
        'usa_cowboy_mixed': 'Cowboy mixed',
        'usa_cowboy_voice': 'Cowboy vocal',
    }
    return mapping.get(x, x.replace('_', ' '))


def get_composition(df):
    tab = pd.crosstab(df['subdomain'], df['class'])
    for c in CLASSES:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[CLASSES]
    return tab.div(tab.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)


def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    # Preserve the defined normalization, subdomain labels, and composition.
    frames, comp, counts = {}, {}, {}
    for scale, path in SOURCES.items():
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        df['class'] = df['nearest_bovine_class'].apply(clean_class)
        df = df[df['domain'] == 'pastoral_human'].copy()
        assert set(df['class']).issubset(CLASSES)
        counts[scale] = df['class'].value_counts().reindex(CLASSES, fill_value=0).to_dict()
        assert counts[scale] == EXPECTED[scale], (scale, counts[scale])
        assert len(df) == 42 and df['subdomain'].nunique() == 18
        frames[scale] = df
        comp[scale] = get_composition(df)
    assert set(comp['MICRO'].index) == set(comp['MACRO'].index)
    common = sorted(set(comp['MICRO'].index).intersection(comp['MACRO'].index))
    comp = {scale: table.loc[common] for scale, table in comp.items()}
    # Apply the canonical sort, including its initial alphabetical order.
    m = comp['MACRO']
    order = m.assign(gc_sort=m['GC'], lfc_sort=m['LFC'], psc_sort=m['PSC'], hfc_sort=m['HFC']).sort_values(
        by=['gc_sort', 'lfc_sort', 'psc_sort', 'hfc_sort'],
        ascending=[False, False, False, True]).index
    comp = {scale: table.loc[order] for scale, table in comp.items()}
    labels = [shorten_pastoral_label(s) for s in order]
    summary = [{'scale': scale, 'class': c, 'count': counts[scale][c], 'total': 42}
               for scale in SOURCES for c in CLASSES]
    pd.DataFrame(summary).to_csv(PROV / 'Figure06__class_summary.csv', index=False)
    rows = []
    for scale in SOURCES:
        rawcounts = pd.crosstab(frames[scale]['subdomain'], frames[scale]['class']).reindex(index=order, columns=CLASSES, fill_value=0)
        for rank, subdomain in enumerate(order, 1):
            for c in CLASSES:
                rows.append({'scale': scale, 'display_order': rank, 'subdomain': subdomain,
                             'display_label': labels[rank-1], 'class': c,
                             'count': int(rawcounts.loc[subdomain, c]),
                             'proportion': float(comp[scale].loc[subdomain, c])})
    pd.DataFrame(rows).to_csv(PROV / 'Figure06__subdomain_composition.csv', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                         'axes.linewidth': 2, 'xtick.major.width': 2,
                         'ytick.major.width': 2})
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_PX/DPI, HEIGHT_PX/DPI), dpi=DPI, sharey=True)
    fig.subplots_adjust(left=.255, right=.985, bottom=.13, top=.90, wspace=.16)
    segment_texts = []
    for ax, scale in zip(axes, SOURCES):
        left = pd.Series(0., index=order)
        for c in CLASSES:
            vals = comp[scale][c]
            ax.barh(range(18), vals.values, left=left.values, color=COLORS[c], height=.78, linewidth=0)
            for i, value in enumerate(vals):
                if value > 0:
                    segment_texts.append(ax.text(float(left.iloc[i] + value/2), i, c,
                                                ha='center', va='center', color=INK[c], fontsize=8))
            left += vals
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ['0', '0.25', '0.50', '0.75', '1'])
        ax.set_xlabel('Proportion', fontsize=9)
        ax.set_yticks(range(18), labels)
        ax.set_ylim(17.65, -.65)
        ax.tick_params(axis='y', length=0, labelsize=8)
        ax.tick_params(axis='x', length=4, width=2, labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(0, 1.08, '(A)' if scale == 'MICRO' else '(B)', transform=ax.transAxes,
                fontsize=10, fontweight='bold', ha='left')
        ax.set_title(f'{scale}-scale correspondence', fontsize=9, pad=8)
    fig.legend([Patch(facecolor=COLORS[c], edgecolor='none') for c in CLASSES], CLASSES,
               loc='lower center', bbox_to_anchor=(.59, .015), ncol=4, frameon=False, fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(t.get_fontsize() for t in visible_texts) >= 8
    # Confirm every label fits the canvas and every class label fits its segment.
    bounds = [t.get_window_extent(renderer) for t in visible_texts]
    assert all(b.x0 >= 0 and b.y0 >= 0 and b.x1 <= WIDTH_PX and b.y1 <= HEIGHT_PX for b in bounds)
    for ax, scale in zip(axes, SOURCES):
        for t in [t for t in segment_texts if t.axes is ax]:
            i = int(t.get_position()[1])
            c = t.get_text()
            segment_width = float(comp[scale].iloc[i][c]) * ax.get_window_extent(renderer).width
            assert t.get_window_extent(renderer).width + 4 <= segment_width
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor='white')  # No tight bounding-box rescaling.
    buf.seek(0)
    rgb = Image.open(buf).convert('RGB')
    assert rgb.size == (WIDTH_PX, HEIGHT_PX)
    for ext in ['png', 'jpg', 'tif']:
        options = {'quality': 100, 'subsampling': 0} if ext == 'jpg' else ({'compression': 'tiff_lzw'} if ext == 'tif' else {})
        rgb.save(FINAL / f'Figure_06.{ext}', dpi=(DPI, DPI), **options)
    exports = []
    for path in sorted(FINAL.glob('Figure_06.*')):
        with Image.open(path) as im:
            dpi = im.info.get('dpi')
            assert im.mode == 'RGB' and im.size == (WIDTH_PX, HEIGHT_PX)
            assert dpi and all(abs(float(d)-DPI) < .01 for d in dpi)
            exports.append({'file': str(path.relative_to(ROOT)), 'pixels': list(im.size),
                            'mode': im.mode, 'dpi': [float(d) for d in dpi], 'sha256': sha(path)})
    metadata = {
        'figure': 6, 'status': 'reproduced_from_packaged_class_label_tables',
        'build_script_sha256': sha(Path(__file__)), 'nominal_width_mm': 180,
        'raster_width_mm_at_300_dpi': WIDTH_PX/DPI*25.4, 'height_mm_at_300_dpi': HEIGHT_PX/DPI*25.4,
        'dpi': DPI, 'minimum_font_pt': min(t.get_fontsize() for t in visible_texts),
        'visible_line_width_pt': 2, 'bar_edge_line_width': 'none; solid fills',
        'palette': COLORS, 'accessibility': 'Every nonzero segment directly labeled GC/HFC/PSC; four-class legend retains LFC at zero.',
        'counts': counts, 'subdomain_order': list(order),
        'sorting': 'MACRO GC desc, LFC desc, PSC desc, HFC asc; canonical pandas sort from alphabetical common-subdomain index',
        'inputs': [{'scale': s, 'file': str(p.relative_to(ROOT)), 'sha256': sha(p), 'columns': ['domain', 'subdomain', 'nearest_bovine_class']} for s,p in SOURCES.items()],
        'upstream_release_limit': 'Figure generation reproducible from preserved derived inputs. Source permissions/performer provenance and upstream pipeline are not closed by this production pass.',
        'interpretation': 'Scale-specific correspondence distributions; no physical transition, temporal trajectory, attractor, affect-state assignment, or aggregation-independent emergence.',
        'guideline': 'https://www.frontiersin.org/guidelines/author-guidelines',
        'runtime': {'python': platform.python_version(), 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__, 'Pillow': PIL.__version__},
        'checks': {'frozen_counts': True, 'same_18_subdomains': True, 'text_inside_canvas': True,
                   'segment_labels_fit': True, 'text_at_least_8_pt': True, 'rgb_exports': True, '300_dpi_exports': True},
        'exports': exports,
    }
    (PROV / 'Figure_06_build_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    plt.close(fig)
    print(json.dumps({'status': metadata['status'], 'pixels': [WIDTH_PX, HEIGHT_PX], 'counts': counts}))


if __name__ == '__main__':
    main()
