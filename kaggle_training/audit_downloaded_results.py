"""Independently audit the downloaded Kaggle output, without retraining.

Usage: python audit_downloaded_results.py PATH_TO_interface_run
Uses the project's previously verified rank-sum AUC and threshold-group AP.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper_benchmark'))
from evaluation import independent_auc, independent_ap, threshold_metrics, topk_expected


def audit(root):
    report=json.loads((root/'results.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    rows=pd.read_csv(root/'per_complex_metrics.csv')
    meta={c['code']:c for c in manifest['cases'] if c['split']=='test'}
    assert len(meta)==55
    assert set(manifest['selection_ids']).isdisjoint(manifest['calibration_ids'])
    differences=[]
    def compare(actual,expected,label):
        gap=abs(float(actual)-float(expected))
        differences.append(gap)
        if gap>1e-10:
            raise AssertionError(f'{label}: actual={actual}, reported={expected}, error={gap}')
    for name,summary in report['models'].items():
        yy,ss=[],[]
        threshold=float(summary['threshold'])
        threshold_counts=dict(tp=0,fp=0,fn=0,tn=0)
        for code in manifest['splits']['test']:
            with np.load(root/'predictions'/f'{name}_{code}.npz',allow_pickle=False) as z:
                y,s=z['y'],z['logit']
                compare(z['threshold'],threshold,'threshold consistency')
            assert len(y)==meta[code]['n_pairs']
            assert int(y.sum())==meta[code]['positives']
            m=threshold_metrics(y,s,threshold)
            row=rows.loc[(rows.model==name)&(rows.complex==code)]
            assert len(row)==1
            row=row.iloc[0]
            for k in ('tp','fp','fn','tn','precision','recall','f1','mcc','balanced_accuracy','specificity','accuracy'):
                compare(m[k],row[k],f'{name}/{code}/{k}')
            for k in threshold_counts: threshold_counts[k]+=m[k]
            compare(independent_auc(y,s),row['roc_auc'],f'{name}/{code}/ROC AUC')
            compare(independent_ap(y,s),row['average_precision'],f'{name}/{code}/AP')
            compare(topk_expected(y,s,10)['precision'],row['precision_at_10'],f'{name}/{code}/P@10')
            yy.append(y)
            ss.append(s)
        y,s=np.concatenate(yy),np.concatenate(ss)
        m=threshold_metrics(y,s,threshold)
        for k in threshold_counts:
            compare(m[k],threshold_counts[k],f'{name}/pooled/sum/{k}')
        for k in ('tp','fp','fn','tn','precision','recall','f1','mcc','balanced_accuracy','specificity','accuracy'):
            compare(m[k],summary[k],f'{name}/pooled/{k}')
        compare(independent_auc(y,s),summary['roc_auc'],f'{name}/pooled/AUC')
        compare(independent_ap(y,s),summary['average_precision'],f'{name}/pooled/AP')
        print('AUDITED',name,'pairs',len(y),'F1',m['f1'],flush=True)
    outcome={'status':'passed','checks':len(differences),'maximum_absolute_metric_error':max(differences),
        'method':'Independent confusion arithmetic, Mann-Whitney rank AUC and tied-threshold AP; '
                 'per-complex arrays and pooled results verified against exported summaries.',
        'scope':'Does not establish a fresh blind test, homology-disjointness, or retrain checkpoints.'}
    (root/'independent_metric_audit.json').write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_directory',type=Path)
    audit(parser.parse_args().output_directory)
