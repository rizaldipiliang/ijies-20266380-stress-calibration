from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
from scipy.special import softmax
from sklearn.metrics import f1_score

ROOT = next(Path('evidence_unpacked').glob('CalibBERT*/CalibBERT*'))
OUT = Path('revision_analysis'); OUT.mkdir(exist_ok=True)
def load(stage, name):
    return pd.read_csv(ROOT / stage / 'Results_Expanded' / name)

fold = load('06_E4_DOMAIN_SHIFT_LODO', 'e4_lodo_fold_manifest.csv')
for stem in ['target_test', 'source_reference_test', 'source_train', 'source_calibration']:
    fold[stem+'_stress_n'] = np.rint(fold[stem+'_n'] * fold[stem+'_stress_rate']).astype(int)
    fold[stem+'_not_stress_n'] = fold[stem+'_n'] - fold[stem+'_stress_n']
fold.to_csv(OUT/'domain_counts.csv', index=False)
metrics = load('06_E4_DOMAIN_SHIFT_LODO','e4_lodo_metrics_long.csv')
metrics.to_csv(OUT/'domain_metrics_full.csv',index=False)
delta = load('06_E4_DOMAIN_SHIFT_LODO','e4_lodo_shift_deltas.csv')
delta.to_csv(OUT/'domain_deltas.csv',index=False)

parts=[]
for method in ['top2vec','bertopic']:
    local=load('08_E6_SEMANTIC_TOPIC_STABILITY',f'e6_{method}_local_heterogeneity_by_seed.csv')
    runs=load('08_E6_SEMANTIC_TOPIC_STABILITY',f'e6_{method}_run_summary.csv')
    col='n_topics_development' if method=='top2vec' else 'development_non_outlier_topics'
    local=local.merge(runs[['seed',col]],on='seed',validate='many_to_one')
    wide=local.pivot(index=['seed',col], columns='condition',values='nll_topic_std').reset_index()
    wide['method']='Top2Vec' if method=='top2vec' else 'BERTopic'
    wide=wide.rename(columns={col:'development_topic_count'})
    wide['TS_minus_Vanilla']=wide['M0 + Temperature Scaling']-wide['M0 Vanilla']
    wide['LS_minus_Vanilla']=wide['M1 LS 0.10']-wide['M0 Vanilla']
    parts.append(wide)
resolution=pd.concat(parts,ignore_index=True)
resolution.to_csv(OUT/'topic_resolution_per_seed.csv',index=False)
resolution.groupby(['method','development_topic_count']).agg(
    runs=('seed','count'),vanilla_nll_sd=('M0 Vanilla','mean'),
    ts_nll_sd=('M0 + Temperature Scaling','mean'),ls_nll_sd=('M1 LS 0.10','mean'),
    ts_reduction_seeds=('TS_minus_Vanilla',lambda s:int((s<0).sum())),
    ls_reduction_seeds=('LS_minus_Vanilla',lambda s:int((s<0).sum()))
).reset_index().to_csv(OUT/'topic_resolution_summary.csv',index=False)

raw=pd.read_csv('canonical_selected/repeated_seed_raw_metrics.csv')
for i,row in raw.iterrows():
    pred=pd.read_csv(f"canonical_selected/repeated_seed_{row.seed}_predictions_{row.model}.csv")
    assert len(pred)==715 and pred.id.is_unique
    order=np.argsort(-pred.confidence.to_numpy(),kind='stable')
    err=(pred.predicted_label.to_numpy()!=pred.true_label.to_numpy()).astype(float)[order]
    raw.loc[i,'aurc']=float(np.mean(np.cumsum(err)/np.arange(1,716)))
raw.to_csv(OUT/'historical_bert_seed_metrics.csv',index=False)
raw.groupby('model')[['f1_score','brier_score','nll','ece','aurc']].agg(['mean','std']).to_csv(OUT/'historical_bert_seed_summary.csv')

benchmark=pd.read_csv(ROOT/'04_E2_CALIBRATION_BENCHMARK/e2_official_test_calibration_benchmark.csv')
pred=pd.read_csv(ROOT/'04_E2_CALIBRATION_BENCHMARK/e2_prediction_level_long.csv')
for i,row in benchmark.iterrows():
    p=pred[(pred.model==row.model)&(pred.calibrator==row.calibrator)].sort_values('row_index')
    assert len(p)==715 and p.id.is_unique
    prob=p.p_stress.to_numpy(); y=p.true_label.to_numpy()
    assert np.isclose(np.mean((prob-y)**2),row.brier,atol=1e-7)
    order=np.argsort(-np.maximum(prob,1-prob),kind='stable')
    err=(p.predicted_label.to_numpy()!=y).astype(float)[order]
    benchmark.loc[i,'aurc']=float(np.mean(np.cumsum(err)/np.arange(1,716)))
benchmark.to_csv(OUT/'bert_posthoc_comparators.csv',index=False)

e3=load('05_E3_CROSS_BACKBONE','e3_prediction_level_long.csv')
e4=load('06_E4_DOMAIN_SHIFT_LODO','e4_prediction_level_long.csv')
e4=e4[e4.evaluation_group=='target_OOD']
pair=e4.merge(e3[e3.backbone=='BERT'],on=['id','condition'],suffixes=('_ood','_id'),validate='one_to_one')
assert len(pair)==2145
assert (pair.true_label_ood==pair.true_label_id).all()
assert (pair.post_id_ood==pair.post_id_id).all()
y=pair.true_label_ood.to_numpy()
for suffix in ['ood','id']:
    p=np.clip(pair['p_stress_'+suffix].to_numpy(),1e-12,1-1e-12)
    pair['brier_'+suffix]=(p-y)**2
    pair['nll_'+suffix]=-(y*np.log(p)+(1-y)*np.log(1-p))
for metric in ['brier','nll']:
    pair[metric+'_delta']=pair[metric+'_ood']-pair[metric+'_id']
pair[['id','post_id_ood','condition','heldout_domain','true_label_ood','p_stress_ood','p_stress_id','brier_delta','nll_delta']].to_csv(OUT/'family_b_paired_differences.csv',index=False)
assert pair[pair.condition=='Vanilla'].post_id_ood.nunique()==586
cluster_domains=pair[pair.condition=='Vanilla'].groupby('post_id_ood').heldout_domain.nunique()
print('Clusters spanning multiple domains:',int((cluster_domains>1).sum()))
print('Resolution summaries')
print(pd.read_csv(OUT/'topic_resolution_summary.csv').to_string(index=False))
print('BERT comparator summaries')
print(benchmark[['model','calibrator','brier','nll','f1','aurc']].to_string(index=False))
