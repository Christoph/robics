"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import math
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy.spatial.distance import pdist
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr, wasserstein_distance
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
topics.rank_models("mean")
topics.show_stability_histograms()

# Look at topics for a specific model
lda.display_topics(sample_id,model_id,tf_feature_names,10)

# Convert the stability report to a pandas dataframe
pd.DataFrame.from_records(topics.stability_report)

# Print histograms
import plotly.express as px
def show_stability_histograms(self):
    for sample in self.rank_models():
        fig = px.histogram(data_frame=sample["mean"], x=0, nbins=10, range_x=[
                        0, 1], title="Model: "+sample["model"]+" Topics: "+str(sample["n_topics"]) + " Mean: "+str(sample["mean/overall"]))
        fig.show()

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

    def __init__(self, n_components=[4, 50], n_samples=10, n_iterations=10, n_relevant_top_words=30, rank_metric="kendalls", distribution_metric="jenson_shannon"):
        self.n_components = n_components
        self.n_samples = n_samples
        self.n_iterations = n_iterations
        self.n_relevant_top_words = n_relevant_top_words
        self.rank_metric = rank_metric
        self.distribution_metric = distribution_metric

        self.params = self._compute_params()
        self.samples = []
        self.topic_similarities = []
        self.topic_terms = []
        self.stability_report = []
        self.full_stability_report = []

    def fit_models(self):
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

        for params in self.params:
            model_iterations = []
            for it in range(self.n_iterations):
                print("Model: LDA - Iteration: ", it)
                model_iterations.append(
                    LatentDirichletAllocation(n_components=params).fit(self.X_lda))

            self.samples.append(model_iterations)

            if(self.X_nmf is not None):
                nmf_iterations = []

                for it in range(self.n_iterations):
                    print("Model: NMF - Iteration: ", it)
                    nmf_iterations.append(
                        NMF(n_components=params).fit(self.X_nmf))
                self.samples.append(nmf_iterations)

        self._compute_topic_stability()

        return self

    def load_sklearn_lda_data(self, X, setup="simple", custom_params=None):
        self.X_lda = X

        if setup == "simple":
            self.params_lda = {
                "n_components": [5, 50]
            }

    def load_sklearn_nmf_data(self, X, setup="simple", custom_params=None):
        self.X_nmf = X

        if setup == "simple":
            self.params_nmf = {
                "n_components": {"type": int, "mode": "range", "values": [5, 50]},
                "init": ["random", "nndsvd", "nndsvda", None],
                "beta_loss": ["frobenius", "kullback-leibler"]
            }

    def _compute_params(self):
        seq = []

        for vec in sobol_seq.i4_sobol_generate(1, self.n_samples):
            seq.append(int(round(
                vec[0] * (self.n_components[1] - self.n_components[0]) + self.n_components[0])))

        return seq

    @staticmethod
    def _compute_param_combinations(params, n_samples):
        seq = []

        for vec in sobol_seq.i4_sobol_generate(len(params), n_samples):
            seq.append(int(round(
                vec[0] * (self.n_components[1] - self.n_components[0]) + self.n_components[0])))

        return seq

    @staticmethod
    def _range_to_value(p_range, sampling, p_type):
        value = p_range[0] + (p_range[1] - p_range[0]) * sampling
        return int(value) if p_type is int else value

    @staticmethod
    def _categories_to_value(p_values, sampling, p_type):
        return p_values[min(math.floor(sampling*len(p_values)), len(p_values)-1)]

    def _compute_topic_stability(self):
        ranking_vecs = self._create_ranking_vectors()

        for sample_id, sample in enumerate(self.samples):
            n_topics = sample[0].n_components
            terms = []
            term_distributions = []
            ranking = []
            distribution = []
            similarities = []
            report = {}
            report_full = {}

            # Get all top terms and distributions
            for model in sample:
                terms.append(self._get_top_terms(
                    model, self.n_relevant_top_words))

                term_distributions.append(
                    model.components_ / model.components_.sum(axis=1)[:, np.newaxis])

            self.topic_terms.append(np.array(terms))

            # Evaluate each topic
            for topic in range(n_topics):
                sim = pdist(np.array(terms)[
                    :, topic, :], self._jaccard_similarity)
                similarities.append(sim)

                if self.distribution_metric == "jenson_shannon":
                    jen = pdist(np.array(term_distributions)[
                        :, topic, :], self._jenson_similarity)
                    distribution.append(jen)
                if self.distribution_metric == "wasserstein":
                    jen = pdist(np.array(term_distributions)[
                        :, topic, :], self._wasserstein_similarity)
                    distribution.append(jen)

                if self.rank_metric == "kendalls":
                    rank = pdist(ranking_vecs[sample_id][
                        :, topic, :], self._kendalls)
                    ranking.append(rank)

                if self.rank_metric == "spearman":
                    spear = pdist(ranking_vecs[sample_id][
                        :, topic, :], self._spear)
                    ranking.append(spear)

            kendalls_ranking = np.array(ranking)
            similarities = np.array(similarities)
            jenson = np.array(distribution)
            self.topic_similarities.append(similarities)

            if isinstance(sample[0], LatentDirichletAllocation):
                report["model"] = "LDA"
                report_full["model"] = "LDA"

            if isinstance(sample[0], NMF):
                report["model"] = "NMF"
                report_full["model"] = "NMF"

            report["sample_id"] = sample_id
            report["n_topics"] = n_topics

            report["jaccard"] = similarities.mean()
            report["jaccard_std"] = similarities.std()
            report[self.rank_metric] = kendalls_ranking.mean()
            report[self.rank_metric+"_std"] = kendalls_ranking.std()
            report[self.distribution_metric] = jenson.mean()
            report[self.distribution_metric+"_std"] = jenson.std()

            report_full["sample_id"] = sample_id
            report_full["n_topics"] = n_topics

            report_full["jaccard"] = similarities.mean()
            report_full["kendalls"] = kendalls_ranking.mean()
            report_full["jenson"] = jenson.mean()

            report_full["jaccard_min"] = similarities.min(axis=1)
            report_full["jaccard_max"] = similarities.max(axis=1)
            report_full["jaccard_mean"] = similarities.mean(axis=1)
            report_full["jaccard_std"] = similarities.std(axis=1)
            report_full[self.rank_metric+"_min"] = kendalls_ranking.min(axis=1)
            report_full[self.rank_metric+"_max"] = kendalls_ranking.max(axis=1)
            report_full[self.rank_metric +
                        "_mean"] = kendalls_ranking.mean(axis=1)
            report_full[self.rank_metric+"_std"] = kendalls_ranking.std(axis=1)
            report_full[self.distribution_metric+"_min"] = jenson.min(axis=1)
            report_full[self.distribution_metric +
                        "_max"] = jenson.max(axis=1)
            report_full[self.distribution_metric+"_mean"] = jenson.mean(axis=1)
            report_full[self.distribution_metric +
                        "_std"] = jenson.std(axis=1)

            self.stability_report.append(report)
            self.full_stability_report.append(report_full)

    def rank_models(self, weights=[1, 1, 1]):
        return sorted(self.stability_report, key=lambda s: (s["jaccard"]*weights[0] + s[self.rank_metric]*weights[1] + s[self.distribution_metric]*weights[2])/np.sum(weights), reverse=True)

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

    def _create_ranking_vectors(self):
        vocab = set()
        sample_terms = []
        ranking_vecs = []

        for sample in self.samples:
            terms = []
            for model in sample:
                top_terms = self._get_top_terms(
                    model, self.n_relevant_top_words)
                terms.append(top_terms)
                vocab.update([e for l in top_terms for e in l])
            sample_terms.append(terms)

        vocab_vec = list(vocab)

        for sample in sample_terms:
            rankings = []
            for model_terms in sample:
                rankings.append([self._terms_to_ranking(t, vocab_vec)
                                 for t in model_terms])
            ranking_vecs.append(np.array(rankings))

        return ranking_vecs

    @staticmethod
    def _jaccard_similarity(a, b):
        sa = set(a)
        sb = set(b)
        return len(sa.intersection(sb))/len(sa.union(sb))

    @staticmethod
    def _kendalls(a, b):
        k, _ = kendalltau(a, b)
        return k

    @staticmethod
    def _spear(a, b):
        k, _ = spearmanr(a, b)
        return k

    @staticmethod
    def _jenson_similarity(a, b):
        distance = jensenshannon(a, b)
        return 1 - distance

    @staticmethod
    def _wasserstein_similarity(a, b):
        distance = wasserstein_distance(a, b)
        return 1 - distance

    @staticmethod
    def _terms_to_ranking(terms, vocab):
        vec = []
        for e in vocab:
            if e in terms:
                vec.append(terms.index(e))
            else:
                vec.append(len(vocab))
        return vec

    @staticmethod
    def _get_top_terms(model, n_terms):
        topic_terms = []
        for topic in model.components_:
            topic_terms.append([i for i in topic.argsort()[:-n_terms - 1:-1]])

        return topic_terms
