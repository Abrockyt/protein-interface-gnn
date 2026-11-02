"""Reproducible DB5-Plus interface experiment for Kaggle's two T4 GPUs.

Commands: prepare, train --kind {mlp,gnn} --seed N, evaluate.
No test labels are used for training, early stopping, scaling or threshold choice.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import tarfile
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.spatial.distance import cdist
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, matthews_corrcoef,
    balanced_accuracy_score, auc, roc_curve)

ROOT = Path(os.environ.get('GNN_RUN_DIR', '/kaggle/working/interface_run'))
SOURCE = 'https://zenodo.org/records/5134732/files/final_raw_db5.tar.gz?download=1'
MD5 = '9cbc07672e705f9ba8549168b06e1e06'
SEEDS = [224, 225, 226]
AA = 'ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL UNK'.split()
SS = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-', '?']
SCALARS = ['rsa_value', 'rd_value', 'avg_cx', 's_avg_cx', 's_ch_avg_cx',
           's_ch_s_avg_cx', 'max_cx', 'min_cx', 'cn_value']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def vector(value, width):
    try:
        a = np.asarray(value, dtype=np.float32).ravel()
    except (TypeError, ValueError):
        return np.full(width, np.nan, dtype=np.float32)
    return a if len(a) == width else np.full(width, np.nan, dtype=np.float32)


def graph(frame):
    df = frame.loc[frame.atom_name == 'CA'].copy()
    if not df.index.is_unique:
        raise ValueError('CA dataframe index must be unique')
    coords = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
    if len(df) < 2 or not np.isfinite(coords).all():
        raise ValueError('Invalid protein coordinates')
    continuous = np.column_stack([df[SCALARS].to_numpy(dtype=np.float32),
        np.stack([vector(v, 42) for v in df.hsaac]),
        np.stack([vector(v, 27) for v in df.sequence_feats])])
    continuous[~np.isfinite(continuous)] = np.nan
    res = np.eye(len(AA), dtype=np.float32)[[AA.index(v) if v in AA else len(AA)-1 for v in df.resname]]
    ss = np.eye(len(SS), dtype=np.float32)[[SS.index(v) if v in SS else len(SS)-1 for v in df.ss_value]]
    pos = np.arange(len(df), dtype=np.float32)[:, None] / max(len(df)-1, 1)
    x = np.column_stack([res, ss, pos, continuous])
    # Intramolecular geometry only. Cross-partner distances would leak the bound pose.
    distances = cdist(coords, coords).astype(np.float32)
    np.fill_diagonal(distances, np.inf)
    k = min(20, len(df)-1)
    hood = np.argsort(distances, axis=1, kind='stable')[:, :k]
    d = np.take_along_axis(distances, hood, axis=1)
    # Scalar distance features preserve rigid rotation/translation invariance.
    rbf = np.exp(-((d[..., None]-np.linspace(0, 20, 16)) / 2.0)**2)
    gap = np.minimum(np.abs(np.arange(len(df))[:, None]-hood), 32)[..., None]/32.0
    edge = np.concatenate([rbf, gap], axis=-1).astype(np.float32)
    seq = '-'.join(df.resname.astype(str))
    return x, hood.astype(np.int64), edge, df.index.to_numpy(), seq


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'final_raw_db5.tar.gz'
    if not archive.exists():
        print('Downloading the official DB5-Plus archive (42.8 MB)', flush=True)
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(SOURCE, archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3)
    actual = hashlib.md5(archive.read_bytes()).hexdigest()
    if actual != MD5:
        raise ValueError(f'Archive checksum mismatch: {actual}')
    cases = []
    manifest = {'source': SOURCE, 'md5': actual, 'label_policy':
        'Released positive CA row-index pairs; full CA Cartesian complement is negative, '
        'following DIPS-Plus build_complex_labels. Stored sampled neg_idx is not used.',
        'splits': {}, 'cases': [], 'prior_test_exposure':
        'Underlying DB5 test complexes were examined in earlier project experiments. '
        'This is a fixed benchmark rerun, not a new blind holdout.'}
    with tarfile.open(archive) as tar:
        for split in ('train', 'val', 'test'):
            names = tar.extractfile(f'raw/pairs-postprocessed-{split}.txt').read().decode().splitlines()
            manifest['splits'][split] = []
            for name in names:
                # Pickle is used only for this checksum-verified, publisher-provided archive.
                obj = pd.read_pickle(io.BytesIO(tar.extractfile('raw/'+name).read()))
                gl, gr = graph(obj.df0), graph(obj.df1)
                xl, hl, el, il, sl = gl
                xr, hr, er, ir, sr = gr
                if not all('_u_' in str(s) for s in obj.srcs.values()):
                    raise ValueError(f'{name}: expected unbound input structures')
                left, right = {v:i for i,v in enumerate(il)}, {v:i for i,v in enumerate(ir)}
                positive = np.asarray(obj.pos_idx, dtype=np.int64)
                if positive.ndim != 2 or positive.shape[1] != 2:
                    raise ValueError(f'{name}: invalid contact labels')
                try:
                    pos = np.array([left[a]*len(xr)+right[b] for a,b in positive], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError(f'{name}: contact refers to a missing CA atom') from exc
                if len(np.unique(pos)) != len(pos) or not 0 < len(pos) < len(xl)*len(xr):
                    raise ValueError(f'{name}: duplicate or degenerate contact labels')
                # Check the released sampled negatives against the exhaustive complement.
                neg = np.array([left[a]*len(xr)+right[b] for a,b in obj.neg_idx], dtype=np.int64)
                if np.intersect1d(pos, neg).size:
                    raise ValueError(f'{name}: conflicting positive/negative labels')
                code = Path(name).stem
                seq_key = hashlib.sha256('|'.join(sorted([sl, sr])).encode()).hexdigest()
                cases.append(dict(code=code, split=split, xl=xl, xr=xr, hl=hl, hr=hr,
                                  el=el, er=er, pos=pos, sequence_pair_hash=seq_key))
                manifest['splits'][split].append(code)
                manifest['cases'].append(dict(code=code, split=split, n_left=len(xl),
                    n_right=len(xr), n_pairs=len(xl)*len(xr), positives=len(pos),
                    prevalence=len(pos)/(len(xl)*len(xr)), sequence_pair_hash=seq_key))
                if len(cases) % 25 == 0:
                    print(f'Checked {len(cases)}/230 complexes', flush=True)
    if [len(manifest['splits'][s]) for s in ('train','val','test')] != [140,35,55]:
        raise ValueError('Unexpected official split sizes')
    # Exact pair-sequence duplicates across splits would be a concrete leakage error.
    seen = {}
    for c in cases:
        key = c['sequence_pair_hash']
        if key in seen and seen[key]['split'] != c['split']:
            raise ValueError(f"Duplicate sequence pair across splits: {seen[key]['code']}, {c['code']}")
        seen[key] = c
    # Hold back part of the official validation set for threshold fitting only.
    val = np.array(manifest['splits']['val'])
    perm = np.random.default_rng(224).permutation(len(val))
    manifest['selection_ids'] = val[perm[:18]].tolist()
    manifest['calibration_ids'] = val[perm[18:]].tolist()
    groups = [set(manifest['splits']['train']), set(manifest['selection_ids']),
              set(manifest['calibration_ids']), set(manifest['splits']['test'])]
    assert all(not a & b for i,a in enumerate(groups) for b in groups[i+1:])
    # Fit imputation and feature scaling on the training proteins only.
    fit = np.concatenate([c[k] for c in cases if c['split']=='train' for k in ('xl','xr')])
    median = np.nanmedian(fit, axis=0)
    median = np.nan_to_num(median).astype(np.float32)
    filled = np.where(np.isfinite(fit), fit, median)
    mean, std = filled.mean(0), filled.std(0)
    std = np.maximum(std, 0.05)
    np.savez_compressed(ROOT/'feature_scaler.npz', median=median, mean=mean, std=std)
    data_dir = ROOT/'processed'
    data_dir.mkdir(exist_ok=True)
    for c in cases:
        for key in ('xl','xr'):
            x = c[key]
            missing = (~np.isfinite(x)).astype(np.float32)
            scaled = np.clip((np.where(np.isfinite(x),x,median)-mean)/std, -10, 10)
            c[key] = np.column_stack([scaled,missing]).astype(np.float32)
            assert np.isfinite(c[key]).all()
        np.savez_compressed(data_dir/f"{c['code']}.npz", **{k:c[k] for k in
            ('xl','xr','hl','hr','el','er','pos')})
    manifest['feature_dimension'] = int(cases[0]['xl'].shape[1])
    manifest['edge_dimension'] = int(cases[0]['el'].shape[-1])
    manifest['normalization'] = 'Training-only median imputation and z scaling, clipped at +/-10; missing masks appended.'
    manifest['sequence_check_limit'] = 'Exact pair sequences checked; shared chains and homologous families can still cross splits.'
    write_json(ROOT/'manifest.json', manifest)
    print('DATA READY:', {s:len(v) for s,v in manifest['splits'].items()},
          'selection=18, calibration=17, feature dimension=', manifest['feature_dimension'], flush=True)


class ResidualBlock(nn.Module):
    def __init__(self, width, edges, neighbors):
        super().__init__()
        self.neighbors = neighbors
        self.center = nn.Linear(width, width)
        if neighbors:
            self.message = nn.Linear(width, width, bias=False)
            self.gate = nn.Sequential(nn.Linear(edges, 32), nn.SiLU(), nn.Linear(32,1), nn.Sigmoid())
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, hood, edge):
        update = self.center(x)
        if self.neighbors:
            weights = self.gate(edge)
            pooled = (x[hood]*weights).sum(1)/weights.sum(1).clamp_min(1e-6)
            update = update+self.message(pooled)
        return self.norm(x+self.dropout(nn.functional.silu(update)))


class Model(nn.Module):
    def __init__(self, dim, edges, neighbors=True, width=128):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(dim,width), nn.LayerNorm(width), nn.SiLU())
        self.layers = nn.ModuleList([ResidualBlock(width,edges,neighbors) for _ in range(3)])
        self.pair = nn.Sequential(nn.Linear(width*3,width), nn.SiLU(), nn.Dropout(0.25),
                                  nn.Linear(width,64), nn.SiLU(), nn.Linear(64,1))
        self.interface = nn.Linear(width,1)

    def encode(self,x,h,e):
        x = self.project(x)
        for layer in self.layers:
            x = layer(x,h,e)
        return x

    def embed(self,c):
        return self.encode(c['xl'],c['hl'],c['el']), self.encode(c['xr'],c['hr'],c['er'])

    def score(self,l,r,flat):
        a,b = l[flat//len(r)],r[flat%len(r)]
        return self.pair(torch.cat([a+b,torch.abs(a-b),a*b],-1)).flatten()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_case(code, device):
    with np.load(ROOT/'processed'/f'{code}.npz', allow_pickle=False) as z:
        c = {k:torch.from_numpy(z[k]).to(device) for k in z.files}
    n = len(c['xl'])*len(c['xr'])
    c['y'] = np.zeros(n, dtype=np.uint8)
    c['y'][c['pos'].cpu().numpy()] = 1
    c['positive'] = np.flatnonzero(c['y'])
    c['negative'] = np.flatnonzero(1-c['y'])
    c['code'] = code
    c['node_l'] = torch.zeros(len(c['xl']),device=device)
    c['node_r'] = torch.zeros(len(c['xr']),device=device)
    c['node_l'][c['pos']//len(c['xr'])] = 1
    c['node_r'][c['pos']%len(c['xr'])] = 1
    return c


@torch.no_grad()
def predict(model,c,chunk=8192):
    model.eval()
    l,r = model.embed(c)
    scores=[]
    for start in range(0,len(c['y']),chunk):
        idx = torch.arange(start,min(start+chunk,len(c['y'])),device=l.device)
        scores.append(model.score(l,r,idx).cpu().numpy())
    out=np.concatenate(scores)
    assert np.isfinite(out).all()
    return out


def train(kind, seed, epochs, patience):
    if not torch.cuda.is_available() and os.environ.get('GNN_ALLOW_CPU') != '1':
        raise RuntimeError('Kaggle GPU unavailable; do not silently start the long run on CPU')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_all(seed)
    manifest=json.loads((ROOT/'manifest.json').read_text())
    training=[load_case(c,device) for c in manifest['splits']['train']]
    validation=[load_case(c,device) for c in manifest['selection_ids']]
    model=Model(manifest['feature_dimension'],manifest['edge_dimension'],kind=='gnn').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=6)
    rng=np.random.default_rng(seed)
    folder=ROOT/f'{kind}_{seed}'
    folder.mkdir(exist_ok=True)
    best,wait,history=-1.,0,[]
    print('TRAINING',kind,seed,'on',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
          'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,epochs+1):
        start=time.time()
        model.train()
        total=0.
        for i in rng.permutation(len(training)):
            c=training[int(i)]
            # Resample negatives each epoch. Balanced TRAINING only, never validation/test.
            p=rng.choice(c['positive'],size=min(512,len(c['positive'])),replace=False)
            n=rng.choice(c['negative'],size=min(len(c['negative']),10*len(p)),replace=False)
            idx=np.concatenate([p,n])
            target=torch.from_numpy(c['y'][idx].astype(np.float32)).to(device)
            idx=torch.from_numpy(idx).to(device)
            opt.zero_grad(set_to_none=True)
            l,r=model.embed(c)
            score=model.score(l,r,idx)
            # Effective balanced pair objective; logits are NOT calibrated probabilities.
            loss=nn.functional.binary_cross_entropy_with_logits(score,target,
                    pos_weight=score.new_tensor(len(n)/len(p)))
            aux=(nn.functional.binary_cross_entropy_with_logits(model.interface(l).flatten(),c['node_l'])+
                 nn.functional.binary_cross_entropy_with_logits(model.interface(r).flatten(),c['node_r']))/2
            loss=loss+0.2*aux
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step()
            total+=float(loss.detach())
        pred=[predict(model,c) for c in validation]
        ap=float(np.mean([average_precision_score(c['y'],s) for c,s in zip(validation,pred)]))
        auc_median=float(np.median([roc_auc_score(c['y'],s) for c,s in zip(validation,pred)]))
        scheduler.step(ap)
        row=dict(epoch=epoch,loss=total/len(training),selection_macro_ap=ap,
                 selection_median_auc=auc_median,lr=opt.param_groups[0]['lr'],seconds=time.time()-start)
        history.append(row)
        if ap>best+1e-5:
            best,wait=ap,0
            torch.save({'state':model.state_dict(),'kind':kind,'seed':seed,'epoch':epoch,
                'selection_macro_ap':ap,'dim':manifest['feature_dimension'],
                'edges':manifest['edge_dimension']},folder/'best.pt')
        else:
            wait+=1
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'{kind} seed={seed} epoch={epoch:03d} loss={row["loss"]:.4f} '
              f'validation AP={ap:.5f} median AUC={auc_median:.4f} '
              f'best AP={best:.5f} wait={wait}/{patience} seconds={row["seconds"]:.1f}',flush=True)
        if wait>=patience:
            break
    write_json(folder/'completed.json',dict(kind=kind,seed=seed,epochs=len(history),best_selection_ap=best))


def threshold_from_calibration(y,score):
    # Compute exact F1 over distinct score thresholds, including score ties correctly.
    p,r,t=precision_recall_curve(y,score)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
    # If tied, choose the higher threshold (fewer predicted positives).
    i=int(np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1])
    return float(t[i]),float(f[i])


def measures(y,score,threshold):
    pred=score>=threshold
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    precision=tp/max(tp+fp,1)
    recall=tp/max(tp+fn,1)
    p,r,_=precision_recall_curve(y,score)
    return dict(n_pairs=len(y),positives=int(y.sum()),prevalence=float(y.mean()),
        roc_auc=float(roc_auc_score(y,score)),average_precision=float(average_precision_score(y,score)),
        trapezoid_pr_auc=float(auc(r,p)),precision=precision,recall=recall,
        f1=2*tp/max(2*tp+fp+fn,1),mcc=float(matthews_corrcoef(y,pred)),
        accuracy=(tp+tn)/len(y),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        specificity=tn/max(tn+fp,1),tp=tp,fp=fp,fn=fn,tn=tn,threshold=threshold)


def top_precision(y,score,k):
    k=min(k,len(y))
    boundary=np.partition(score,len(score)-k)[len(score)-k]
    above=score>boundary
    tied=score==boundary
    tp=float(y[above].sum())+(k-int(above.sum()))*float(y[tied].mean())
    return tp/k


def evaluate():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((ROOT/'manifest.json').read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calibration=[load_case(c,device) for c in manifest['calibration_ids']]
    test=[load_case(c,device) for c in manifest['splits']['test']]
    calibration_y=np.concatenate([c['y'] for c in calibration])
    test_y=np.concatenate([c['y'] for c in test])
    models={}
    selected={}
    for kind,seed in [('mlp',224)]+[('gnn',s) for s in SEEDS]:
        ckpt=torch.load(ROOT/f'{kind}_{seed}'/'best.pt',map_location=device,weights_only=True)
        model=Model(ckpt['dim'],ckpt['edges'],kind=='gnn').to(device)
        model.load_state_dict(ckpt['state'])
        name=f'{kind}_{seed}'
        # Architecture sanity checks on a calibration example only.
        c=calibration[0]
        model.eval()
        with torch.no_grad():
            l,r=model.embed(c)
            flat=torch.tensor([0,len(c['y'])//2,len(c['y'])-1],device=device)
            swap=(flat%len(r))*len(l)+(flat//len(r))
            torch.testing.assert_close(model.score(l,r,flat),model.score(r,l,swap),atol=1e-5,rtol=1e-5)
        models[name]={'calibration':[predict(model,c) for c in calibration],
                      'test':[predict(model,c) for c in test]}
        selected[name]={'epoch':ckpt['epoch'],'selection_macro_ap':ckpt['selection_macro_ap']}
        del model
    # Predeclared equal logit ensemble; no seed or model is selected on test scores.
    models['gnn_ensemble']={split:[np.mean([models[f'gnn_{s}'][split][i] for s in SEEDS],axis=0)
        for i in range(len(calibration) if split=='calibration' else len(test))]
        for split in ('calibration','test')}
    summaries={}
    records=[]
    outputs=ROOT/'predictions'
    outputs.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for name,scores in models.items():
        cp=np.concatenate(scores['calibration'])
        threshold,cal_f1=threshold_from_calibration(calibration_y,cp)
        ts=np.concatenate(scores['test'])
        summary=measures(test_y,ts,threshold)
        summary['calibration_f1']=cal_f1
        summary['calibration_pairs']=len(calibration_y)
        per=[]
        for c,s in zip(test,scores['test']):
            row=measures(c['y'],s,threshold)
            row.update(model=name,complex=c['code'],precision_at_10=top_precision(c['y'],s,10),
                precision_at_l=top_precision(c['y'],s,min(len(c['xl']),len(c['xr']))))
            per.append(row)
            records.append(row)
            np.savez_compressed(outputs/f"{name}_{c['code']}.npz",y=c['y'],logit=s,threshold=threshold)
        summary.update(median_complex_roc_auc=float(np.median([x['roc_auc'] for x in per])),
            macro_average_precision=float(np.mean([x['average_precision'] for x in per])),
            macro_f1=float(np.mean([x['f1'] for x in per])),
            mean_precision_at_10=float(np.mean([x['precision_at_10'] for x in per])))
        # Resample whole complexes for intervals, preserving within-complex dependence.
        rng=np.random.default_rng(224)
        hashes={c['code']:c['sequence_pair_hash'] for c in manifest['cases']}
        grouped={}
        for row in per:
            key=hashes[row['complex']]
            grouped.setdefault(key,np.zeros(3,dtype=np.int64))
            grouped[key]+=np.array([row['tp'],row['fp'],row['fn']])
        counts=np.stack(list(grouped.values()))
        boot=[]
        for _ in range(2000):
            tp,fp,fn=counts[rng.integers(0,len(counts),len(counts))].sum(0)
            boot.append(2*tp/max(2*tp+fp+fn,1))
        summary['pooled_f1_complex_bootstrap_95ci']=np.quantile(boot,[0.025,0.975]).tolist()
        summary['bootstrap_sequence_pair_groups']=len(counts)
        summaries[name]=summary
        if name in ('mlp_224','gnn_ensemble'):
            p,r,_=precision_recall_curve(test_y,ts)
            axes[0].plot(r,p,label=name)
            fpr,tpr,_=roc_curve(test_y,ts)
            axes[1].plot(fpr,tpr,label=name)
            axes[2].bar(np.arange(3)+(0 if name=='mlp_224' else .36),
                [summary['precision'],summary['recall'],summary['f1']],.36,label=name)
        print('RESULT',name,json.dumps(summary),flush=True)
    axes[0].axhline(test_y.mean(),color='gray',ls='--',label='contact prevalence')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Full-candidate test precision-recall')
    axes[1].plot([0,1],[0,1],'k--',alpha=.5)
    axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='Test ROC')
    axes[2].set_xticks(np.arange(3)+.18,['Precision','Recall','F1'])
    axes[2].set(title='Threshold chosen on calibration only',ylim=(0,1))
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(ROOT/'test_results.png',dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(ROOT/'per_complex_metrics.csv',index=False)
    pd.DataFrame(summaries).T.to_csv(ROOT/'summary_metrics.csv')
    pd.DataFrame([{'model':n,**v} for n,v in selected.items()]).to_csv(ROOT/'selected_checkpoints.csv',index=False)
    report={'models':summaries,'selected_checkpoints':selected,'runtime':{
        'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'protocol':{'training_complexes':140,'selection_complexes':18,'threshold_calibration_complexes':17,
            'test_complexes':55,'max_epochs':150,'early_stopping_patience':20,
            'checkpoint_metric':'mean per-complex validation average precision',
            'threshold':'pooled F1 maximum on 17 calibration complexes only',
            'heldout_distribution':'Every CA residue pair, with no negative downsampling',
            'main_output':'Pair contact prediction, not auxiliary single-residue interface classification',
            'prior_test_exposure':manifest['prior_test_exposure'],
            'uncertainty':'Bootstrap resamples exact sequence-pair groups, conditional on fitted models and threshold; '
            '3 GNN seeds report training variation. Homology dependencies are not removed.'}}
    write_json(ROOT/'results.json',report)
    (ROOT/'COMPLETE.txt').write_text('Completed successfully. Read results.json and manifest.json.\n')
    print('EXPERIMENT COMPLETE. Artifacts:',ROOT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train','evaluate'])
    parser.add_argument('--kind',choices=['mlp','gnn'],default='gnn')
    parser.add_argument('--seed',type=int,default=224)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--patience',type=int,default=20)
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='train': train(args.kind,args.seed,args.epochs,args.patience)
    else: evaluate()
