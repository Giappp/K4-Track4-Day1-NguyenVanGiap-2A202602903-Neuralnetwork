"""CoverType tensors; validation split and scaling use training data only."""
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10

def load_split(processed_dir='data/processed'):
    with np.load(Path(processed_dir)/'train.npz') as tr, np.load(Path(processed_dir)/'eval.npz') as ev:
        arrays=(tr['X'],tr['y'],ev['X'],ev['y'],ev['row_id'])
    for X,y in ((arrays[0],arrays[1]),(arrays[2],arrays[3])):
        assert X.ndim==2 and X.shape[1]==54 and X.dtype==np.float32
        assert y.shape==(len(X),) and y.dtype==np.int64
        assert np.isfinite(X).all() and np.all((y>=0)&(y<7))
    assert len(np.unique(arrays[4]))==len(arrays[2])
    return arrays

def make_val_split(X,y,val_fraction=.2,seed=42):
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=val_fraction,stratify=y,random_state=seed)
    return X_tr,y_tr,X_val,y_val

def fit_standardizer(X_tr):
    # Float64 accumulation avoids rounding error on the large training matrix.
    mean=X_tr[:,:N_NUMERIC].mean(0,dtype=np.float64).astype(np.float32)
    std=X_tr[:,:N_NUMERIC].std(0,dtype=np.float64).astype(np.float32)
    return mean,std

def apply_standardizer(X,mean,std):
    result=X.copy()
    result[:,:N_NUMERIC]=(result[:,:N_NUMERIC]-mean)/np.where(std==0,1,std)
    return result

def prepare_data(device,val_fraction=.2,seed=42,processed_dir='data/processed'):
    X_full,y_full,X_eval,y_eval,row_id=load_split(processed_dir)
    X_tr,y_tr,X_val,y_val=make_val_split(X_full,y_full,val_fraction,seed)
    mean,std=fit_standardizer(X_tr)
    result={}
    for split,X,y in (('tr',X_tr,y_tr),('val',X_val,y_val),('eval',X_eval,y_eval)):
        result['X_'+split]=torch.as_tensor(apply_standardizer(X,mean,std),dtype=torch.float32,device=device)
        result['y_'+split]=torch.as_tensor(y,dtype=torch.int64,device=device)
    result.update(eval_row_id=row_id,mean=mean,std=std,split_seed=seed,val_fraction=val_fraction)
    majority=int(np.bincount(y_tr).argmax())
    print(f'Training set size: {len(y_tr)}\nValidation set size: {len(y_val)}\nEval set size: {len(y_eval)}')
    print(f'Majority class from train: {majority}, validation accuracy: {np.mean(y_val==majority):.6f}')
    return result

def iterate_batches(X,y,batch_size,generator=None,shuffle=True):
    if batch_size<=0:
        raise ValueError('batch_size must be positive')
    perm=torch.randperm(len(X),generator=generator,device=X.device) if shuffle else torch.arange(len(X),device=X.device)
    for idx in perm.split(batch_size):
        yield X[idx],y[idx]
