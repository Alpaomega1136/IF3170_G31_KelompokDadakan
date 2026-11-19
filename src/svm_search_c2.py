"""Research run: MI .75, C2 gamma .16 on 80% training splits."""
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
for seed in (2026,77):
    fit,val=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
    prep=Pipeline([('prep',preprocessor(X.iloc[fit])),('weights',MutualInformationWeights(exponent=.75,min_score=.002,random_state=42))])
    a=prep.fit_transform(X.iloc[fit],y[fit]); b=prep.transform(X.iloc[val])
    start=perf_counter()
    model=SVC(C=2,gamma=.16,tol=.003,cache_size=384).fit(a,y[fit])
    scores=model.decision_function(b)
    record={'seed':seed,'n_train':len(fit),'n_valid':len(val),'exponent':.75,'C':2,'gamma':.16,
            'f1_-0.3':f1_score(y[val],scores>=-.3,average='macro'),
            'f1_0':f1_score(y[val],scores>=0,average='macro'),'seconds':perf_counter()-start}
    print(record,flush=True)
    records.append(record)
    pd.DataFrame(records).to_csv(ROOT/'output/SVM/experiments/search_c2_80percent.csv',index=False)
