"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
import sobol_seq

"""
Sources who show how to use Topic models

https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730


TEST CODE

from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data[:100]

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

lda = RobustTopics()
lda.fit(tf)
#lda.stability_report
lda.rank_models("mean")

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

    def __init__(self, n_components=[5, 50], n_samples=5, n_iterations=10,
                 topic_models=[LatentDirichletAllocation]):
        self.n_components = n_components
        self.n_samples = n_samples
        self.params = self._compute_params()

        self.n_iterations = n_iterations
        self.topic_models = topic_models

        self.models = []
        self.topic_similarities = []
        self.stability_report = []

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

        for params in self.params:
            model_iterations = []
            for model in self.topic_models:
                for it in range(self.n_iterations):
                    print("Model: ", model, " - Iteration: ", it)
                    model_iterations.append(model(n_components=params).fit(X))
            self.models.append(model_iterations)

        self._compute_topic_stability()

        return self

    def _compute_params(self):
        seq = []

        for vec in sobol_seq.i4_sobol_generate(1, self.n_samples):
            seq.append(int(round(
                vec[0] * (self.n_components[1] - self.n_components[0]) + self.n_components[0])))

        return seq

    def _compute_topic_stability(self):
        for sample_id, sample in enumerate(self.models):
            n_topics = sample[0].n_components
            terms = []
            similarities = []
            report = {}

            # Get all top terms
            for model in sample:
                terms.append(self._get_top_terms(model, 10))

            # Evaluate each topic
            for topic in range(n_topics):
                sim = pdist(np.array(terms)[
                    :, topic, :], self._jaccard_similarity)
                similarities.append(sim)

            similarities = np.array(similarities)
            self.topic_similarities.append(similarities)

            report["sample_id"] = sample_id
            report["n_topics"] = n_topics
            report["min"] = similarities.min(axis=1)
            report["max"] = similarities.max(axis=1)
            report["mean"] = similarities.mean(axis=1)
            report["std"] = similarities.std(axis=1)

            self.stability_report.append(report)

    def rank_models(self, value="mean"):
        return sorted(self.stability_report, key=lambda s: s[value].mean(), reverse=True)

    @staticmethod
    def _jaccard_similarity(a, b):
        sa = set(a)
        sb = set(b)
        return len(sa.intersection(sb))/len(sa.union(sb))

    @staticmethod
    def _get_top_terms(model, n_terms):
        topic_terms = []
        for topic_idx, topic in enumerate(model.components_):
            topic_terms.append([i for i in topic.argsort()[:-n_terms - 1:-1]])

        return topic_terms

    def display_topics(self, model_number, feature_names, no_top_words):
        for topic_idx, topic in enumerate(self.models[model_number].components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))
