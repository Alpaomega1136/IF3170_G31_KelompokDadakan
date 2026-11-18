"""Run the scratch SVM with training-only mutual-information feature weights.

Run from the project root: python src/svm_weighted.py
Results are stored separately from the unweighted SVM experiment.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.pipeline import Pipeline

import svm_optimized as svm
from experiment_svm import preprocessor as basic_preprocessor
from preprocessing.svm_feature_weights import MutualInformationWeights


def weighted_preprocessor(X):
    return Pipeline([
        ("preprocessing", basic_preprocessor(X)),
        ("feature_weights", MutualInformationWeights(exponent=0.5, min_score=0.002,
                                                     random_state=42)),
    ])


def configure():
    svm.preprocessor = weighted_preprocessor
    svm.OUTPUT = svm.ROOT / "output" / "SVM" / "weighted"
    svm.CONFIG = {**svm.DEFAULT_CONFIG, "weighting": "sqrt mutual information",
                  "mutual_info_random_state": 42, "min_mi": 0.002}


def main():
    configure()
    svm.main()


if __name__ == "__main__":
    main()
