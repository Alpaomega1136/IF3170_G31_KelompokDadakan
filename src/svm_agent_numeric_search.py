"""Train-only comparison of numeric transforms; SVC is a research proxy."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from experiment_svm import ROOT, preprocessor
from preprocessing.svm_feature_weights import MutualInformationWeights

class NumericTransform(BaseEstimator, TransformerMixin):
    def __init__(self, mode):
        self.mode = mode
    def fit(self, X, y=None):
        self.cols_ = ['days_before_arrival', 'nightly_rate_idr', 'prior_cancellations', 'prior_completed_bookings']
        self.low_ = X[self.cols_].quantile(.01)
        self.high_ = X[self.cols_].quantile(.99)
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            if self.mode == 'clip': X[c] = X[c].clip(self.low_[c], self.high_[c])
            if self.mode == 'log': X[c] = np.log1p(X[c].clip(lower=0))
            if self.mode == 'sqrt': X[c] = np.sqrt(X[c].clip(lower=0))
        return X

def main():
    d = pd.read_csv(ROOT/'data/train.csv')
    X = d.drop(columns=['id', 'is_canceled']); y = d.is_canceled.to_numpy()
    records=[]
    for seed in [303, 404]:
        tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=y)
        for mode in ['baseline','clip','sqrt','log']:
            f=NumericTransform(mode).fit(X.iloc[tr]); a=f.transform(X.iloc[tr]); b=f.transform(X.iloc[va])
            p=preprocessor(a); a=p.fit_transform(a); b=p.transform(b)
            for exponent in ([.5,.75] if mode == 'baseline' else [.75]):
                w=MutualInformationWeights(exponent=exponent).fit(a,y[tr]); aa=w.transform(a); bb=w.transform(b)
                for gamma in [.16]:
                    start=time.time(); m=SVC(C=4,gamma=gamma,tol=.003,cache_size=384).fit(aa,y[tr]); scores=m.decision_function(bb)
                    row={'seed':seed,'mode':mode,'exponent':exponent,'C':4,'gamma':gamma,'seconds':time.time()-start}
                    for t in [-.5,-.3,-.15,0,.15]: row['t'+str(t)]=f1_score(y[va],scores>=t,average='macro')
                    records.append(row); print(row,flush=True)
                    pd.DataFrame(records).to_csv(ROOT/'output/SVM/experiments/agent_numeric_search.csv',index=False)
if __name__ == '__main__': main()
