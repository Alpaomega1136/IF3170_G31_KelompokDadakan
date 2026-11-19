"""Research run: class weights with MI .75 SVC."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from experiment_svm import preprocessor
from preprocessing.svm_feature_weights import MutualInformationWeights

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data/train.csv')
X=d.drop(columns=['id','is_canceled']); y=d.is_canceled.to_numpy()
records=[]
best_weight = None
for seed in (21,2026,77):
    if seed != 21 and best_weight is None:
        break
    fit,val=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
    if seed == 21:
        fit,_=train_test_split(fit,train_size=14000,random_state=seed,stratify=y[fit])
    prep=Pipeline([('prep',preprocessor(X.iloc[fit])),('weights',MutualInformationWeights(exponent=.75,min_score=.002,random_state=42))])
    a=prep.fit_transform(X.iloc[fit],y[fit]); b=prep.transform(X.iloc[val])
    for weight in ((.8,1.,1.2,1.4) if seed == 21 else (1.,best_weight)):
        start=perf_counter()
        model=SVC(C=4,gamma=.16,tol=.003,cache_size=384,class_weight={0:1.,1:weight}).fit(a,y[fit])
        scores=model.decision_function(b)
        record={'seed':seed,'n_train':len(fit),'n_valid':len(val),'weight_1':weight,
                'f1_-0.3':f1_score(y[val],scores>=-.3,average='macro'),
                'f1_0':f1_score(y[val],scores>=0,average='macro'),'seconds':perf_counter()-start}
        print(record,flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(ROOT/'output/SVM/experiments/search_class_weight.csv',index=False)
    if seed == 21:
        best = max(records,key=lambda r:r['f1_-0.3'])
        base = next(r for r in records if r['weight_1']==1.)
        if best['f1_-0.3'] > base['f1_-0.3']:
            best_weight = best['weight_1']
        print('selected weight:', best_weight, flush=True)
