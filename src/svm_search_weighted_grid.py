"""Temporary research: candidate weights and RBF-SVC settings on train split."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from experiment_svm import preprocessor
from preprocessing.svm_feature_weights import MutualInformationWeights

ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data/train.csv')
X=d.drop(columns=['id','is_canceled'])
y=d['is_canceled'].to_numpy()
seed=42
pool,val=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
fit,_=train_test_split(pool,train_size=14000,random_state=seed,stratify=y[pool])
records=[]
for indicators,exp in [(False,0.25),(False,0.5),(False,0.75),(False,1.0)]:
    prep=Pipeline([('prep',preprocessor(X.iloc[fit],indicators)),('weights',MutualInformationWeights(exponent=exp,min_score=0.002,random_state=42))])
    a=prep.fit_transform(X.iloc[fit],y[fit]); b=prep.transform(X.iloc[val])
    settings=([(4,.16),(4,.24),(8,.24)] if exp == .5 else
              [(4,.16),(4,.24)] if exp in (.25,.75) else [(4,.16)])
    for c,g in settings:
        t=time.perf_counter()
        model=SVC(C=c,gamma=g,tol=.003,cache_size=384).fit(a,y[fit])
        z=model.decision_function(b)
        scores={str(th):f1_score(y[val],z>=th,average='macro') for th in [-.3,-.2,-.1,0,.1,.2]}
        rec={'indicators':indicators,'exponent':exp,'C':c,'gamma':g,'seconds':round(time.perf_counter()-t,2),**scores}
        records.append(rec)
        print(rec,flush=True)
        pd.DataFrame(records).to_csv(ROOT/'output/SVM/experiments/search_weights_agent_42.csv',index=False)
