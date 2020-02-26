"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
import plotly.express as px
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

# Initialize and fit the data
topics = RobustTopics()
topics.fit(X_lda=tf, X_nmf=tfidf)

# Compare different samples
# topics.stability_report
# pd.DataFrame.from_records(topics.stability_report)
topics.rank_models("mean")
topics.show_stability_histograms()

# Look at topics for a specific model
lda.display_topics(sample_id,model_id,tf_feature_names,10)

"""


class RobustTopics():
    """Run different topic models multiple times and return them by their ranked by topic stability.

    Some Explanation
    ----------
    n_components : [min_n, max_n], default=[1, 20]
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

    def __init__(self, n_components=[1, 20], n_samples=5, n_iterations=8, n_relevant_top_words=20):
        self.n_components = n_components
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_relevant_top_words = n_relevant_top_words

        self.params = self._compute_params()
        self.samples = []
        self.topic_similarities = []
        self.topic_terms = []
        self.stability_report = []

    def fit(self, X_lda, X_nmf=None, y=None):
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
        self.X_lda = X_lda
        self.X_nmf = X_nmf

        for params in self.params:
            model_iterations = []
            for it in range(self.n_iterations):
                print("Model: LDA - Iteration: ", it)
                model_iterations.append(
                    LatentDirichletAllocation(n_components=params).fit(X_lda))

            self.samples.append(model_iterations)

            if(X_nmf is not None):
                nmf_iterations = []

                for it in range(self.n_iterations):
                    print("Model: NMF - Iteration: ", it)
                    nmf_iterations.append(
                        NMF(n_components=params).fit(X_nmf))
                self.samples.append(nmf_iterations)

        self._compute_topic_stability()

        return self

    def _compute_params(self):
        seq = []

        for vec in sobol_seq.i4_sobol_generate(1, self.n_samples):
            seq.append(int(round(
                vec[0] * (self.n_components[1] - self.n_components[0]) + self.n_components[0])))

        return seq

    def _compute_topic_stability(self):
        for sample_id, sample in enumerate(self.samples):
            n_topics = sample[0].n_components
            terms = []
            similarities = []
            report = {}

            # Get all top terms
            for model in sample:
                terms.append(self._get_top_terms(
                    model, self.n_relevant_top_words))

            self.topic_terms.append(np.array(terms))

            # Evaluate each topic
            for topic in range(n_topics):
                sim = pdist(np.array(terms)[
                    :, topic, :], self._jaccard_similarity)
                similarities.append(sim)

            similarities = np.array(similarities)
            self.topic_similarities.append(similarities)

            if isinstance(sample[0], LatentDirichletAllocation):
                report["model"] = "LDA"

            if isinstance(sample[0], NMF):
                report["model"] = "NMF"

            report["sample_id"] = sample_id
            report["n_topics"] = n_topics

            report["mean/overall"] = similarities.mean()
            report["min"] = similarities.min(axis=1)
            report["max"] = similarities.max(axis=1)
            report["mean"] = similarities.mean(axis=1)
            report["std"] = similarities.std(axis=1)

            self.stability_report.append(report)

    def show_stability_histograms(self):
        for sample in self.rank_models():
            fig = px.histogram(data_frame=sample["mean"], x=0, nbins=10, range_x=[
                               0, 1], title="Model: "+sample["model"]+" Topics: "+str(sample["n_topics"]) + " Mean: "+str(sample["mean/overall"]))
            fig.show()

    def rank_models(self, value="mean"):
        return sorted(self.stability_report, key=lambda s: s[value].mean(), reverse=True)

    def analyse_sample(self, sample_id, feature_names):
        print("Intersecting words for each topic")

        # Intersect each topic
        for topic in range(len(self.samples[sample_id][0].components_)):
            inter = set(self.topic_terms[sample_id][topic][0])
            for terms in self.topic_terms[sample_id]:
                inter.intersection_update(set(list(terms[topic])))

            print("Topic: " + str(topic))
            print(" ".join([feature_names[i] for i in inter]))

    def display_topics(self, sample_number, model_number, feature_names, no_top_words):
        for topic_idx, topic in enumerate(self.samples[sample_number][model_number].components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

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
