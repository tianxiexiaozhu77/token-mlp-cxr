#!/usr/bin/env python3
"""Draw the Token MLP overview using a real image and audited model responses."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import Normalize
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
FIGURES = PAPER / 'figures'
INK, MUTED = '#263746', '#657481'
BLUE, BLUE_FILL = '#326C91', '#EEF5F9'
RED, RED_FILL = '#B4463C', '#FFF0EC'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    asset_path = FIGURES / 'token_mlp_real_assets.npz'
    provenance_path = PAPER / 'supplement_data/token_mlp_figure_provenance.json'
    evidence = json.loads(provenance_path.read_text())
    assert evidence['status'] == 'verified' and evidence['split'] == 'train'
    assert digest(asset_path) == evidence['exported_assets_sha256']
    data = np.load(asset_path, allow_pickle=False)
    classes = data['class_names'].tolist()
    query_indices = [classes.index(name) for name in ('Effusion', 'Atelectasis')]
    maps = np.stack([data[name][query_indices] for name in ('original_attention', 'adapted_attention')])
    assert maps.shape == (2, 2, 37, 37) and np.isfinite(maps).all() and (maps >= 0).all()
    vmax = float(maps.max())
    norm = Normalize(vmin=0., vmax=vmax)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7,
        'mathtext.fontset': 'stixsans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'axes.unicode_minus': False})
    width, height = 180., 126.
    fig = plt.figure(figsize=(width/25.4, height/25.4), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis('off')

    def text(x, y, value, size=7., color=INK, ha='center', bold=False, **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center',
                       fontweight='bold' if bold else 'normal', zorder=10, **kwargs)

    def box(x, y, w, h, face=BLUE_FILL, edge='#B8CFDE', lw=.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.1',
                                   facecolor=face, edgecolor=edge, linewidth=lw, zorder=1))

    def arrow(points, color=BLUE, dashed=False, lw=.9):
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, linewidth=lw, linestyle='--' if dashed else '-', zorder=4)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=7, color=color, linewidth=lw, linestyle='--' if dashed else '-',
            shrinkA=0, shrinkB=0, zorder=5))

    def child(x, y, w, h):
        axis = fig.add_axes([x/width, y/height, w/width, h/height])
        axis.set_xticks([])
        axis.set_yticks([])
        return axis

    def lock(x, y, color=BLUE):
        ax.add_patch(Arc((x, y+.65), 1.3, 1.5, theta1=0, theta2=180,
                         color=color, linewidth=.65, zorder=8))
        ax.add_patch(Rectangle((x-.85, y-.65), 1.7, 1.35, fill=False,
                               edgecolor=color, linewidth=.65, zorder=8))

    def chest(x, y, side, heat=None):
        axis = child(x, y, side, side)
        axis.imshow(data['image_rgb'], interpolation='bilinear')
        if heat is not None:
            axis.imshow(heat, cmap='magma', norm=norm, alpha=.45,
                        extent=(-.5, 517.5, 517.5, -.5), interpolation='bilinear')
        for spine in axis.spines.values():
            spine.set_edgecolor('#AABAC5')
            spine.set_linewidth(.6)
        return axis

    # (a) A common method interface, illustrated with actual AlphaRAD inputs.
    text(2, 122, '(a) Shared adaptation before disease-conditioned aggregation', size=8.2, ha='left', bold=True)
    lock(145, 122)
    text(148, 122, 'Frozen', size=6.5, color=BLUE, ha='left')
    box(164, 120.7, 2.6, 2.6, RED_FILL, RED)
    text(168, 122, 'Trainable', size=6.5, color=RED, ha='left')
    chest(2, 86, 26)
    text(15, 82.5, 'Real NIH chest X-ray', size=6.3)
    text(15, 79, 'Training example', size=6., color=MUTED)
    box(34, 90, 23, 17)
    lock(37, 104.2)
    text(46, 98.5, 'Image encoder\n+ projection', size=7.1, linespacing=1.4)
    matrix = child(64, 92, 13, 13)
    limit = float(np.abs(data['token_matrix']).max())
    matrix.imshow(data['token_matrix'], cmap='RdBu_r', vmin=-limit, vmax=limit,
                  interpolation='nearest', aspect='auto')
    for spine in matrix.spines.values():
        spine.set_linewidth(.6)
        spine.set_edgecolor('#AABAC5')
    text(70.5, 109, 'Projected tokens', size=6.1)
    text(70.5, 88, r'$T\in\mathbb{R}^{N\times d}$', size=8.)
    box(85, 89, 26, 19, RED_FILL, RED, lw=1.)
    text(98, 101.2, 'Shared', size=7.3, color=RED, bold=True)
    text(98, 97.2, 'Token MLP', size=7.9, color=RED, bold=True)
    text(98, 92.2, r'$d\rightarrow64\rightarrow d$', size=8., color=RED)
    text(98, 85, r'Only $\theta$ is optimized', size=6.2, color=RED)
    box(119, 89, 28, 19)
    lock(122, 105.2)
    text(133.5, 99., 'Disease-conditioned\naggregation / fusion', size=6.6, linespacing=1.4)
    text(133, 92.2, r'$\widetilde{T},Q\longmapsto H$', size=8.5)
    box(155, 89, 23, 19)
    lock(158, 105.2)
    text(166.5, 100, 'Frozen readout', size=6.7)
    text(166.5, 94, r'$s\in\mathbb{R}^{C}$', size=10)
    for x0, x1 in [(28.8,33.2),(57.8,63.2),(77.8,84.2),(111.8,118.2),(147.8,154.2)]:
        arrow([(x0,98.5),(x1,98.5)], color=RED if x0 in (77.8,111.8) else BLUE)
    box(2, 69, 55, 8.6, face='#F8F6F0', edge='#D9D2C4')
    text(29.5, 75.1, 'Fixed disease prompts', size=6.6, bold=True)
    text(29.5, 71.6, '“There is pleural effusion.”', size=6.4, color=MUTED)
    box(64, 69, 47, 8.6)
    lock(67, 73.3)
    text(89, 73.3, 'Text encoder + projection', size=6.9)
    box(119, 69, 28, 8.6)
    text(133, 74.7, 'Fixed text features', size=6.2)
    text(133, 71.2, r'$Q$', size=9)
    arrow([(57.8,73.3),(63.2,73.3)])
    arrow([(111.8,73.3),(118.2,73.3)])
    arrow([(133,78.3),(133,88.2)])
    arrow([(147.8,73.3),(166.5,73.3),(166.5,88.2)], dashed=True, lw=.75)
    text(162, 81.5, 'similarity heads', size=5.7, color=MUTED, ha='right')
    ax.plot([1,179],[64,64], color='#DBE3E9', linewidth=.7)

    # (b) Residual bypass and zero initialization, retaining the shared module.
    text(2,60.4,'(b) One shared residual MLP',size=8.2,ha='left',bold=True)
    box(1,7,86,47,face='#FFFAF8',edge='#E4C9C3')
    text(6.5,36,r'$t_i$',size=10)
    for x, label in [(16,r'$d\rightarrow64$'), (54,r'$64\rightarrow d$')]:
        box(x,30,15,12,face=RED_FILL,edge=RED)
        text(x+7.5,38.5,'Linear',size=6.9,color=RED)
        text(x+7.5,33.7,label,size=8.,color=RED)
    box(37,30,12,12,face=RED_FILL,edge=RED)
    text(43,36,'GELU',size=6.9,color=RED)
    for x0,x1 in [(9,15.2),(31.8,36.2),(49.8,53.2),(69.8,74.5)]:
        arrow([(x0,36),(x1,36)],color=RED)
    ax.add_patch(Circle((77,36),2.2,facecolor='white',edgecolor=RED,linewidth=.9))
    text(77,36,'+',size=10,color=RED)
    arrow([(79.5,36),(83.6,36)],color=RED)
    text(85.2,36,r'$\tilde t_i$',size=9,color=RED)
    arrow([(11,36),(11,49),(77,49),(77,38.6)],color=MUTED,lw=.85)
    text(44,51.5,'Identity skip',size=6.4,color=MUTED)
    text(61.5,26.5,'Zero-initialized',size=6.,color=RED)
    text(44,20,r'$\tilde t_i=t_i+W_u\,\mathrm{GELU}(W_dt_i+b_d)+b_u$',size=8.6)
    text(44,12.3,'Same weights for every token and every disease',size=6.45,color=MUTED)
    text(44,3.4,r'Only $\theta$ is trained; frozen operators transmit gradients.',size=6.1,color=MUTED)

    # (c) Native attention, computed from real frozen/adapted tokens. The same
    # color range is shared by both methods and both disease queries.
    text(94,60.4,'(c) Actual query attention',size=8.2,ha='left',bold=True)
    text(128.5,55.7,'Original',size=6.7,color=BLUE,bold=True)
    text(158.5,55.7,'Pre MLP',size=6.7,color=RED,bold=True)
    for j, (y, label) in enumerate([(30.5,'Pleural\neffusion'),(5.5,'Atelectasis')]):
        text(94, y+11.5, label, size=6.5, ha='left', linespacing=1.3)
        chest(117,y,23,maps[0,j])
        chest(147,y,23,maps[1,j])
    bar = child(174,5.5,1.8,48)
    gradient = np.linspace(0,vmax,256)[:,None]
    bar.imshow(gradient,origin='lower',aspect='auto',cmap='magma',norm=norm)
    bar.axis('off')
    text(174.9,55.7,f'{vmax:.4f}',size=5.7,color=MUTED)
    text(174.9,3.1,'0',size=5.7,color=MUTED)
    text(178.3,29.5,'Shared attention scale',size=5.7,color=MUTED,rotation=90)
    text(142,1.8,r'AlphaRAD: $37\times37$ patches, mean of 8 heads',size=5.8,color=MUTED)

    stem = FIGURES / 'token_mlp_pipeline_real'
    outputs = []
    for suffix in ('pdf','svg','png'):
        path = stem.with_suffix('.'+suffix)
        fig.savefig(path, dpi=350, facecolor='white')
        outputs.append(path)
    plt.close(fig)
    render = dict(status='rendered', assets_sha256=digest(asset_path), provenance_sha256=digest(provenance_path),
        query_order=['Effusion','Atelectasis'], methods=['Original','Pre MLP'],
        attention_normalization=dict(vmin=0.,vmax=vmax,shared_across_all_four_maps=True,
            clipping=False,patch_renormalization=False,colormap='magma',overlay_alpha=.45),
        output_sha256={p.name:digest(p) for p in outputs}, renderer_sha256=digest(Path(__file__)))
    (PAPER / 'supplement_data/token_mlp_figure_render.json').write_text(json.dumps(render,indent=2)+'\n')
    print('Rendered vector PDF/SVG and 350-dpi PNG; all four attention maps share range 0 to',vmax)


if __name__ == '__main__':
    main()
