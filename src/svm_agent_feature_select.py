"""Research-only SVC comparison: feature selection fit on labeled training splits."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from svm_safe_submission import ROOT, make_preprocessor

def main():
    data = pd.read_csv(ROOT/'data/train.csv', dtype={'id': str})
    X = data.drop(columns=['id','is_canceled'])
    y = data.is_canceled.to_numpy()
    output = ROOT/'output/SVM/experiments/agent_feature_select.csv'
    rows = []
    for seed in [2026, 303]:
        train,valid = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        prep = make_preprocessor()
        a = prep.fit_transform(X.iloc[train],y[train]); b = prep.transform(X.iloc[valid])
        for minimum in [0., .002, .005, .01]:
            selected = prep.scores_ >= minimum
            start = time.perf_counter()
            model = SVC(C=4,gamma=.16,tol=.003,cache_size=384).fit(a[:,selected],y[train])
            scores = model.decision_function(b[:,selected])
            row = {'seed':seed,'minimum_mi':minimum,'n_features':int(selected.sum()),
                   'n_train':len(train),'n_valid':len(valid),'threshold':-.2,
                   'macro_f1':f1_score(y[valid],scores>=-.2,average='macro'),
                   'seconds':time.perf_counter()-start}
            rows.append(row); print(row,flush=True)
            pd.DataFrame(rows).to_csv(output,index=False)
if __name__ == '__main__': main()
