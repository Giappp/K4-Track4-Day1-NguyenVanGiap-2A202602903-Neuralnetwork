"""Reproducible training; validation selects checkpoints, never eval labels."""
import copy
import random
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params, activation_stats
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(exp_id='base-s1', group='baseline', description='Baseline M-base',
    loss='ce', optimizer='sgd_momentum', lr=.05, weight_decay=0., momentum=.9,
    batch=512, epochs=20, hidden=(256,128), dropout=0., init='he', clip_norm=None,
    precision='fp32', seed=1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def macro_f1_from_confusion(cm):
    den = cm.sum(0) + cm.sum(1)
    return float(np.divide(2*np.diag(cm), den, out=np.zeros(7), where=den>0).mean())

def compute_loss(logits, y, loss_name):
    if loss_name == 'ce':
        return F.cross_entropy(logits, y)
    if loss_name == 'mse':
        # Raw logits vs one-hot; mean over samples AND all seven output coordinates.
        return F.mse_loss(logits, F.one_hot(y, 7).to(logits.dtype))
    raise ValueError(loss_name)

@torch.no_grad()
def predict(model, X, batch_size=8192):
    model.eval()
    return torch.cat([model(x).argmax(1) for x in X.split(batch_size)])

@torch.no_grad()
def evaluate(model, X, y, loss_name='ce', batch_size=8192):
    model.eval()
    total = 0.
    cm = torch.zeros(49, dtype=torch.int64, device=X.device)
    for xb, yb in iterate_batches(X,y,batch_size,shuffle=False):
        logits = model(xb)
        total += float(compute_loss(logits,yb,loss_name))*len(yb)
        cm += torch.bincount(yb*7+logits.argmax(1), minlength=49)
    cm = cm.reshape(7,7).cpu().numpy()
    return dict(loss=total/len(y), acc=float(np.trace(cm)/cm.sum()),
                macro_f1=macro_f1_from_confusion(cm))

def run_experiment(cfg, data):
    cfg = {**DEFAULT_CFG, **cfg}
    set_seed(cfg['seed'])
    device = data['X_tr'].device
    if cfg['precision'] not in ('fp32','fp16','bf16'):
        raise ValueError(cfg['precision'])
    if cfg['precision']=='fp16' and device.type!='cuda':
        raise ValueError('FP16 experiment requires CUDA; CPU supports the BF16 experiment.')
    model = MLP(hidden=tuple(cfg['hidden']), dropout=cfg['dropout'], init=cfg['init']).to(device)
    assert count_params(model)==EXPECTED_PARAMS[tuple(cfg['hidden'])]
    opt = build_optimizer(cfg['optimizer'],model.parameters(),cfg['lr'],cfg['weight_decay'],cfg['momentum'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['precision']=='fp16')
    generator = torch.Generator(device=device).manual_seed(cfg['seed'])
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    initial = evaluate(model,data['X_val'],data['y_val'],cfg['loss'])
    stats = activation_stats(model,data['X_val'][:1024])
    history = {k:[] for k in ('epoch','train_loss','val_loss','val_acc','val_macro_f1',
                              'grad_norm','grad_norm_max','clip_fraction','epoch_time_s')}
    best_loss, best_epoch, best_state, diverged = float('inf'), 0, copy.deepcopy(model.state_dict()), False
    # Same fixed first 50,000 training samples for every run; dropout disabled for measurement.
    all_norms=[]
    for epoch in range(1,cfg['epochs']+1):
        start=time.perf_counter()
        model.train()
        norms=[]
        for xb,yb in iterate_batches(data['X_tr'],data['y_tr'],cfg['batch'],generator):
            opt.zero_grad(set_to_none=True)
            context = nullcontext() if cfg['precision']=='fp32' else torch.autocast(
                device_type=device.type, dtype=torch.float16 if cfg['precision']=='fp16' else torch.bfloat16)
            with context:
                loss=compute_loss(model(xb),yb,cfg['loss'])
            if not torch.isfinite(loss):
                diverged=True
                break
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn=clip_gradients(model.parameters(),cfg['clip_norm'])
            if not np.isfinite(gn):
                diverged=True
                break
            norms.append(gn)
            scaler.step(opt)
            scaler.update()
        if diverged:
            break
        all_norms.extend(norms)
        tr=evaluate(model,data['X_tr'][:50000],data['y_tr'][:50000],cfg['loss'])
        val=evaluate(model,data['X_val'],data['y_val'],cfg['loss'])
        if not np.isfinite(tr['loss']+val['loss']):
            diverged=True
            break
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        values=(epoch,tr['loss'],val['loss'],val['acc'],val['macro_f1'],float(np.mean(norms)),
                max(norms),float(np.mean(np.array(norms)>cfg['clip_norm'])) if cfg['clip_norm'] else 0.,
                time.perf_counter()-start)
        for k,v in zip(history,values):
            history[k].append(v)
        if val['loss']<best_loss:
            best_loss,best_epoch,best_state=val['loss'],epoch,copy.deepcopy(model.state_dict())
        print(f"{cfg['exp_id']} {epoch:02d}/{cfg['epochs']} val_loss={val['loss']:.4f} F1={val['macro_f1']:.4f} ({values[-1]:.1f}s)",flush=True)
    i=best_epoch-1
    summary=dict(step0_loss=initial['loss'],best_val_loss=best_loss if best_epoch else None,
        best_epoch=best_epoch, final_train_loss=history['train_loss'][-1] if best_epoch else None,
        final_val_loss=history['val_loss'][-1] if best_epoch else None,
        val_acc=history['val_acc'][i] if best_epoch else None,
        val_macro_f1=history['val_macro_f1'][i] if best_epoch else None,
        time_per_epoch_s=float(np.mean(history['epoch_time_s'])) if best_epoch else None,
        peak_mem_MB=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
        diverged=diverged, activation_std=stats,
        grad_norm_p90=float(np.quantile(all_norms,.9)) if all_norms else None)
    return dict(cfg=cfg,history=history,summary=summary,best_state=best_state)

def write_predictions(row_id, preds, path):
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id)==len(preds) and len(np.unique(row_id))==len(row_id)
    assert np.issubdtype(preds.dtype,np.integer) and np.all((preds>=0)&(preds<7))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(row_id=row_id,pred=preds)).to_csv(path,index=False)

def final_eval(cfg,result,data,pred_path):
    model=MLP(hidden=tuple(cfg['hidden']),dropout=cfg['dropout'],init=cfg['init']).to(data['X_eval'].device)
    model.load_state_dict(result['best_state'])
    write_predictions(data['eval_row_id'],predict(model,data['X_eval']).cpu().numpy(),pred_path)
