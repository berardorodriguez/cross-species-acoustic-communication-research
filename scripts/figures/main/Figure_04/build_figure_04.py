from pathlib import Path
import argparse, json, hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from PIL import Image

STATES = ['GC','LFC','HFC','PSC']
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPOSITORY_ROOT / 'data' / 'derived' / 'figure_04'
OUTPUT_DIR = REPOSITORY_ROOT / 'outputs' / 'figure_04_rebuild'
GROUP_COLORS = {
    'Included in O1 and O2': '#1f77b4',
    'Included in O2 only': '#2ca02c',
    'Explored but not included': '#bdbdbd',
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(DATA_DIR / 'Figure04_plot_data.csv'))
    ap.add_argument('--outdir', default=str(OUTPUT_DIR))
    args = ap.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    required = {'material','group','GC','LFC','HFC','PSC','closest_reference'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')

    for c in STATES:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[STATES].isna().any().any():
        raise ValueError('Missing/non-numeric distance values detected')

    calc = df[STATES].idxmin(axis=1)
    if not (calc == df['closest_reference']).all():
        bad = df.loc[calc != df['closest_reference'], ['material','closest_reference']].copy()
        bad['calculated_closest'] = calc[calc != df['closest_reference']]
        raise ValueError(f'closest_reference mismatch:\n{bad}')

    # Preserve the display ordering: increasing GC distance.
    df = df.sort_values('GC', kind='mergesort').reset_index(drop=True)
    mat = df.set_index('material')[STATES]

    # Two-column figure, sized for legibility at final width.
    mm = 25.4
    fig_w = 180/mm
    fig_h = 145/mm
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    im = ax.imshow(mat.values, aspect='auto', cmap='Blues_r')
    ax.set_xticks(range(len(STATES)))
    ax.set_xticklabels(STATES, fontweight='bold', fontsize=9)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0, pad=18)

    ax.set_xticks([x - 0.5 for x in range(1, len(STATES)+1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(mat.index)+1)], minor=True)
    ax.grid(which='minor', color='white', linewidth=2.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    vmin = float(np.nanmin(mat.values))
    vmax = float(np.nanmax(mat.values))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat.iloc[i, j])
            n = (val-vmin)/(vmax-vmin+1e-12)
            txt_color = 'white' if n < 0.52 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=txt_color)

    # Side stripe encodes only compositional-use status.
    for i, row in df.iterrows():
        ax.add_patch(Rectangle(
            (-0.68, i-0.5), 0.16, 1.0,
            facecolor=GROUP_COLORS[row['group']],
            edgecolor='white', linewidth=2.0, clip_on=False
        ))

    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.04)
    cbar.set_label('Euclidean distance to bovine reference centroid (3D PCA)', fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=2.0, length=4)
    cbar.outline.set_linewidth(2.0)

    legend_handles = [Patch(facecolor=GROUP_COLORS[k], label=k) for k in GROUP_COLORS]
    fig.legend(handles=legend_handles, loc='lower center', ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 0.012), fontsize=8)

    # No title inside artwork; publication legend carries interpretation.
    fig.subplots_adjust(left=0.33, right=0.90, top=0.97, bottom=0.18)

    png = outdir/'Figure_04.png'
    jpg = outdir/'Figure_04.jpg'
    tif = outdir/'Figure_04.tif'
    fig.savefig(png, dpi=300, facecolor='white')
    plt.close(fig)

    # Frontiers main-figure TIFF/JPEG must be RGB; flatten the opaque render.
    rgb = Image.open(png).convert('RGB')
    rgb.save(png, format='PNG', dpi=(300, 300))
    rgb.save(jpg, format='JPEG', quality=95, dpi=(300, 300))
    rgb.save(tif, format='TIFF', compression='tiff_lzw', dpi=(300, 300))
    rgb.close()

    # Save exact plotting table used.
    plotdata = outdir/'Figure04_plot_data.csv'
    df.to_csv(plotdata, index=False)

    qc = {
        'n_materials': int(len(df)),
        'states': STATES,
        'all_closest_reference_values_match_row_minimum': True,
        'sort_order': 'ascending GC distance',
        'distance_definition': 'Euclidean distance to bovine OAAS reference centroids in three-dimensional PCA space',
        'group_encoding': 'compositional-use status only; not affective information',
        'plotting_input': inp.name,
        'voco_note': 'Voco+ Human Full is an Ableton synthesizer-type instrument with human-like vocal timbre; explored but not included in Obonuco.',
        'scientific_boundary': [
            'lower values indicate greater acoustic proximity to the corresponding bovine reference centroid',
            'instrument identity does not imply intrinsic affective state',
            'compositional-use status does not demonstrate that OAAS distance caused instrument inclusion',
            'figure does not establish optimization toward GC or another centroid'
        ],
        'frontiers_format': {'width_mm': 180, 'height_mm': round(fig_h * 25.4, 2), 'dpi': 300, 'color_mode': 'RGB', 'minimum_visible_text_pt': 8, 'minimum_figure_line_width_pt': 2.0, 'panel_labels': 'not applicable (single-panel figure)'},
        'build_status': 'reproduced_from_packaged_plot_data',
        'input_sha256': hashlib.sha256(inp.read_bytes()).hexdigest(),
        'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [png, jpg, tif]}
    }
    (outdir/'Figure_04_build_metadata.json').write_text(json.dumps(qc, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
