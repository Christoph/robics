"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
import sobol_seq

"""
for testing
from sklearn.datasets import make_multilabel_classification
X, _ = make_multilabel_classification(random_state=0)

"""


class RobustTopics():
    """Run different topic models multiple times and return them by their ranked by topic stability.

    Some Explanation
    ----------
    n_components : [min_n, max_n], default=[5, 50]
        Minimum and maximum values for the n_components parameter.
    n_samples : int, default=10
        The number of samples taken from the n_components range.
    n_iterations : int, default=20
        The number of random runs each sample is computed. 
        These are used to compute the robustness.
    models : [sklearn topic model classes], default=[LatentDirichletAllocation]
        Possibilities: LatentDirichletAllocation, NMF

    See also
    --------
    sklearn.decomposition.LatentDirichletAllocation : LDA implementation.
    sklearn.decomposition.NMF : NMF implementation.
    """

    def __init__(self, n_components=[5, 50], n_samples=5, n_iterations=10, topic_models=[LatentDirichletAllocation, NMF]):
        self.n_components = n_components
        self.samples = n_samples
        self.n_iterations = n_iterations
        self.topic_models = topic_models

        self.models = []

    def fit(self, X, y=None):
        """Fit the models

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        y : Ignored

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X

        for model in self.topic_models:
            for params in self._compute_params():
                result = model(n_components=params).fit(X)
                self.models.append(result)

        return self

    def _compute_params(self):
        seq = []

        for vec in sobol_seq.i4_sobol_generate(1, self.n_iterations):
            seq.append(int(round(
                vec[0] * (self.n_components[1] - self.n_components[0]) + self.n_components[0])))

        return seq
