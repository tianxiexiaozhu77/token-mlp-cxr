"""Standalone scientific figure of patient-paired RSNA transfer differences."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parent/'runs/20261120/rsna_transfer'
    data=json.loads((root/'summary.json').read_text());models=['AlphaRAD','KAD','MaCo']
    contrasts=[next(c for c in data['comparisons'] if c['model']==m and c['method_b']=='original') for m in models]
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.5),sharey=True)
    for ax,metric,title,color in zip(axes,['auroc','ap'],['AUROC','Average precision'],['#276FBF','#B85C00']):
        y=np.arange(len(models));point=np.array([r['macro'][metric]['point'] for r in contrasts])*100
        ci=np.array([r['macro'][metric]['ci95'] for r in contrasts])*100
        ax.errorbar(point,y,xerr=np.stack([point-ci[:,0],ci[:,1]-point]),fmt='o',color=color,capsize=4,lw=1.8,ms=6)
        ax.axvline(0,color='#555555',ls='--',lw=1);ax.set_title(title);ax.set_yticks(y,models)
        ax.set_xlabel('Pre MLP − Original (percentage points)');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        xmin=ci[:,0].min();xmax=ci[:,1].max();margin=(xmax-xmin)*.17
        ax.set_xlim(xmin-margin,xmax+margin);ax.set_ylim(2.5,-.7)
        for yy,v,(lo,hi) in zip(y,point,ci):ax.text(v,yy-.19,f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=8.3)
    axes[1].tick_params(labelleft=True)
    fig.suptitle('CheXpert-trained models transferred to RSNA pneumonia/opacity',fontsize=12,y=.99)
    fig.text(.5,.01,'5,297 images / 3,713 patients · Mean of 3 fixed training seeds · 2,000 paired patient draws\nUnadjusted percentile 95% intervals; source-only checkpoint selection; no RSNA fitting',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.12,1,.95])
    fig.savefig(root/'pre_vs_original.pdf',bbox_inches='tight');fig.savefig(root/'pre_vs_original.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(root/'pre_vs_original.pdf')


if __name__=='__main__':main()
