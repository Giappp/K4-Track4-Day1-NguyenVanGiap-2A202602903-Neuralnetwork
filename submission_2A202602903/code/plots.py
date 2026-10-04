"""Export experiment curves using a noninteractive backend."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_run(result,path):
    h,c=result['history'],result['cfg']
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for key in ('train_loss','val_loss'):
        ax[0].plot(h['epoch'],h[key],label=key)
    for key in ('val_acc','val_macro_f1'):
        ax[1].plot(h['epoch'],h[key],label=key)
    for key in ('grad_norm','grad_norm_max'):
        ax[2].plot(h['epoch'],h[key],label=key)
    for a,label in zip(ax,('Loss','Validation score','Gradient L2 before clipping')):
        a.set(xlabel='Epoch',ylabel=label)
        a.axvline(result['summary']['best_epoch'],color='grey',ls=':',label='Best val loss')
        a.legend(fontsize=8)
        a.grid(alpha=.2)
    fig.suptitle(f"{c['exp_id']} | {c['optimizer']} lr={c['lr']} batch={c['batch']} | {c['loss']} {c['init']} dropout={c['dropout']} clip={c['clip_norm']} {c['precision']}")
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130,bbox_inches='tight')
    plt.close(fig)

def plot_compare(results,metric,path,title=''):
    fig,ax=plt.subplots(figsize=(9,5))
    for r in results:
        ax.plot(r['history']['epoch'],r['history'][metric],label=r['cfg']['exp_id'])
    ax.set(xlabel='Epoch',ylabel=metric,title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=130,bbox_inches='tight')
    plt.close(fig)
