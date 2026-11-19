"""Research run: MI exponent .75 with C/gamma grid, seed 21, train 14000."""
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
pool,val=train_test_split(np.arange(len(y)),test_size=.2,random_state=21,stratify=y)
fit,_=train_test_split(pool,train_size=14000,random_state=21,stratify=y[pool])
prep=Pipeline([('prep',preprocessor(X.iloc[fit])),('weights',MutualInformationWeights(exponent=.75,min_score=.002,random_state=42))])
a=prep.fit_transform(X.iloc[fit],y[fit]); b=prep.transform(X.iloc[val])
records=[]
for C in (2,4,6):
    for gamma in (.12,.16,.20):
        start=perf_counter()
        model=SVC(C=C,gamma=gamma,tol=.003,cache_size=384).fit(a,y[fit])
        scores=model.decision_function(b)
        record={'seed':21,'n_train':len(fit),'n_valid':len(val),'exponent':.75,'C':C,'gamma':gamma,
                'f1_-0.3':f1_score(y[val],scores>=-.3,average='macro'),
                'f1_0':f1_score(y[val],scores>=0,average='macro'),'seconds':perf_counter()-start}
        print(record,flush=True)
        records.append(record)
        pd.DataFrame(records).to_csv(ROOT/'output/SVM/experiments/search_mi075_seed21.csv',index=False)
