"""Build the self-contained notebook uploaded through Kaggle's UI."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
cells=[]
def markdown(text):
    cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True)))
def code(text):
    cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(keepends=True)))

markdown('''# Protein interface prediction with graph neural networks

## Plan and research question
Can shared graph encoders improve residue-pair contact prediction over an MLP using the same protein features?

1. Verify Kaggle GPUs and the publisher's DB5-Plus archive checksum.
2. Construct separate 20-neighbor graphs from **unbound** protein chains. Use amino acid, secondary structure, surface, sequence and local geometry features. Never use cross-partner distances.
3. Fit feature preprocessing on 140 official training complexes only. Split the 35 official validation complexes into 18 checkpoint-selection and 17 threshold-calibration complexes, using a fixed seed before examining performance.
4. Train an MLP baseline and three GNN seeds, up to 150 epochs with early stopping. Use the two T4 GPUs for independent training jobs.
5. Freeze checkpoints using validation average precision. Choose each classifier's F1 threshold using calibration pairs only.
6. Evaluate all 55 test complexes on their full residue-pair distribution. Export precision, recall, F1, ROC-AUC, average precision, MCC, confusion counts, per-complex results, predictions and checkpoints.

## References and scope
- [Fout et al., NeurIPS 2017](https://papers.neurips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)
- [DIPS-Plus / DB5-Plus data](https://zenodo.org/records/5134732)
- [Publisher's implementation](https://github.com/BioinfoMachineLearning/DIPS-Plus)

This is a modern educational experiment, not an exact reproduction of either paper. This dataset was obtained from its research publisher and is processed **inside Kaggle**; it is not being relabeled as a Kaggle-hosted dataset.

**Prior exposure:** These underlying DB5 test complexes were examined in earlier project experiments. These results are a fixed benchmark rerun, not an independently blind holdout. Exact sequence-pair overlap across splits is checked; family-level homology leakage is not ruled out. Generic 0.75 or 0.90 F1 targets are not guaranteed for sparse residue contacts.
''')

code('''import os, sys, subprocess, platform, json
from pathlib import Path
import torch
print('KAGGLE CLOUD:', platform.platform())
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2 in Kaggle Settings before running.'
# dill supports the publisher's serialized namedtuple; no legacy DGL dependency is required.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'dill==0.4.0'], check=True)
os.environ['GNN_RUN_DIR']='/kaggle/working/interface_run'
RUN=Path(os.environ['GNN_RUN_DIR'])
RUN.mkdir(exist_ok=True)
''')

source=(HERE/'train.py').read_text(encoding='utf-8')
code("# Preserve the entire experiment implementation with the notebook output.\nTRAIN_SOURCE="+repr(source)+"\nPath('/kaggle/working/train.py').write_text(TRAIN_SOURCE, encoding='utf-8')\nprint('Saved train.py:',len(TRAIN_SOURCE),'characters')\n")

code('''# Verify source data, build graphs, and freeze the split before training.
subprocess.run([sys.executable, '-u', '/kaggle/working/train.py', 'prepare'], check=True)
manifest=json.loads((RUN/'manifest.json').read_text())
print('Split sizes:', {k:len(v) for k,v in manifest['splits'].items()})
print('Checkpoint selection:',len(manifest['selection_ids']),
      'Threshold calibration:',len(manifest['calibration_ids']))
''')

markdown('''## Training objective and model selection

The shared GNN encoder applies three residual neighborhood layers. Distance-based edge gates weight messages from neighbors within each chain. A symmetric pair head uses the sum, absolute difference and product of the two residue embeddings. A small auxiliary interface loss trains the encoder; the reported primary task remains **residue-pair contact prediction**.

Training resamples negatives at 10 per positive and balances their loss contribution. All validation, calibration and test pairs remain exhaustive. As a result, training logits are **not calibrated probabilities**. A default 0.5 sigmoid cutoff would be unjustified. We choose the logit cutoff on the separate calibration complexes instead.

The stopping rule maximizes mean per-complex validation average precision. The test labels are never consulted by the training processes. Each process saves its best checkpoint and epoch history. The MLP shares encoder width, pair head and auxiliary objective; removing neighborhood messages changes the parameter count, so it is a feature/width-matched baseline rather than an exact parameter-matched comparison.
''')

code('''import time
# Independent jobs use separate GPUs. No test-based model or seed selection.
waves=[ [('mlp',224,0),('gnn',224,1)], [('gnn',225,0),('gnn',226,1)] ]
for wave in waves:
    active=[]
    for kind,seed,gpu in wave:
        label=f'{kind}_{seed}'
        log_path=RUN/f'{label}.log'
        handle=log_path.open('w', buffering=1)
        env=os.environ.copy()
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
        env['PYTHONUNBUFFERED']='1'
        p=subprocess.Popen([sys.executable,'-u','/kaggle/working/train.py','train',
            '--kind',kind,'--seed',str(seed),'--epochs','150','--patience','20'],
            env=env,stdout=handle,stderr=subprocess.STDOUT)
        active.append((label,p,handle,log_path))
        print('STARTED',label,'Kaggle GPU',gpu,'pid',p.pid,flush=True)
    last={}
    while any(p.poll() is None for _,p,_,_ in active):
        for label,p,handle,path in active:
            lines=path.read_text(errors='replace').splitlines()
            line=lines[-1] if lines else 'starting'
            if last.get(label)!=line:
                print(label,':',line,flush=True)
                last[label]=line
        time.sleep(20)
    failures=[]
    for label,p,handle,path in active:
        handle.close()
        print(label,'exit',p.returncode,flush=True)
        if p.returncode:
            failures.append(label)
            print(path.read_text(errors='replace')[-12000:],flush=True)
    if failures:
        raise RuntimeError('Training failed: '+', '.join(failures))
print('ALL FOUR TRAINING RUNS FINISHED',flush=True)
''')

markdown('''## Evaluation and interpretation

- **Precision** = TP / (TP + FP): how many proposed contacts are real?
- **Recall** = TP / (TP + FN): how many real contacts were recovered?
- **F1** = 2 TP / (2 TP + FP + FN): their harmonic balance at the calibration-selected threshold.
- **Average precision** summarizes contact ranking in the precision-recall curve. It is distinct from trapezoidal PR area. Compare it with the printed contact prevalence.
- **ROC-AUC** measures overall ranking; a high AUC can coexist with low precision for rare contacts.
- **MCC** and **balanced accuracy** include both classes. Raw accuracy is accompanied by an always-negative reference.
- **P@10** is computed with fractional credit for ties at the cutoff.

Scores are reported both pooled across pairs and by complex. A sequence-pair-group bootstrap provides a conditional interval for pooled F1. Three GNN runs show variation across training seeds. These do not establish performance on a homology-disjoint external benchmark.

The equal-weight GNN logit ensemble is predeclared. Individual seeds and the ensemble are all reported, regardless of which test result looks best. The test scores are measured outcomes and may remain modest. Larger training duration alone does not guarantee improvement.
''')

code('''subprocess.run([sys.executable,'-u','/kaggle/working/train.py','evaluate'],check=True)
import pandas as pd
from IPython.display import display, Image
results=json.loads((RUN/'results.json').read_text())
table=pd.DataFrame(results['models']).T
display(table[['precision','recall','f1','roc_auc','average_precision','mcc',
               'median_complex_roc_auc','mean_precision_at_10','tp','fp','fn','tn']])
prevalence=float(table.iloc[0]['prevalence'])
print('Always-negative accuracy:',1-prevalence,'; recall=0, F1=0, MCC=0')
display(Image(filename=str(RUN/'test_results.png')))
print('All checkpoints, histories, predictions and metrics:',RUN)
assert (RUN/'COMPLETE.txt').exists()
''')

code('''# Export a compact results package plus the separately available prediction files.
import zipfile
bundle=Path('/kaggle/working/Protein_Interface_GNN_Kaggle_Results.zip')
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in RUN.rglob('*'):
        if p.is_file() and 'processed' not in p.parts and p.suffix!='.gz':
            z.write(p,p.relative_to(RUN))
    z.write('/kaggle/working/train.py','train.py')
print('RESULTS READY:',bundle,'bytes:',bundle.stat().st_size)
''')

notebook=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5)
for i,c in enumerate(cells): c['id']=f'protein-interface-{i:02d}'
target=HERE/'Protein_Interface_GNN_Kaggle.ipynb'
target.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
print(target, target.stat().st_size)
