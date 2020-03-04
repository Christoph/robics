"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import math
from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import jensenshannon
from scipy.stats import kendalltau, spearmanr, wasserstein_distance, energy_distance
import sobol_seq


"""
Sources who show how to use Topic models

https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730


EXAMPLE CODE

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import LdaModel, LsiModel
from gensim.utils import simple_preprocess
from gensim import corpora

dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data[:100]

# SKLEARN
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

# GENSIM
def docs_to_words(docs):
    for doc in docs:
        yield(simple_preprocess(str(doc), deacc=True))

tokenized_data = list(docs_to_words(documents))
dictionary = corpora.Dictionary(tokenized_data)
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# TOPIC MODELLING
robustTopics = RobustTopics()

robustTopics.load_gensim_LdaModel(LdaModel, corpus, dictionary, 5, n_initializations=4)
robustTopics.load_sklearn_LatentDirichletAllocation(LatentDirichletAllocation, tf, tf_vectorizer, 5, n_initializations=4)

robustTopics.fit_models()

# Compare different samples
# topics.stability_report
topics.rank_models()

# Look at topics for a specific model
topics.analyse_sample("sklearn_nmf", 0, tfidf_feature_names)

# Convert the stability report to a pandas dataframe
pd.DataFrame.from_records(topics.stability_report)

# Print histograms
import plotly.express as px
def show_stability_histograms(self):
    for sample in self.rank_models():
        fig = px.histogram(data_frame=ARRAY, x=0, nbins=10, range_x=[
                        0, 1])
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

    def __init__(self, n_relevant_top_words=20):
        self.n_relevant_top_words = n_relevant_top_words

        self.models = []

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

        for model in self.models:
            print("")
            print("Model: ", model.topic_model_class)

            for sample in model.sampling_parameters:
                sample_initializations = []
                print(sample, end=" - ")
                print(str(model.n_initializations)+" Iterations:", end=" ")

                for it in range(1, model.n_initializations+1):
                    print(str(it), end=" ")

                    if model.source_lib == "sklearn":
                        sample_initializations.append(
                            model.topic_model_class(**sample).fit(model.data))

                    if model.source_lib == "gensim":
                        sample_initializations.append(model.topic_model_class(
                            corpus=model.data, id2word=model.word_mapping, **sample))

                model.samples.append(sample_initializations)
                print("")

        self._compute_topic_stability()

        return self

    def load_gensim_LdaModel(self, LdaModel_class, corpus, dictionary, n_samples, n_initializations=10, setup="simple", custom_params=None):
        parameters = {}
        if setup == "simple":
            parameters = {
                "num_topics":
                {"type": int, "mode": "range", "values": [5, 20]}
            }

        if setup == "complex":
            parameters = {
                "num_topics":
                {"type": int, "mode": "range", "values": [5, 50]},
                "decay":
                {"type": float, "mode": "range", "values": [0.51, 1]}
            }

        if setup == "custom":
            parameters = custom_params

        sampling_parameters = self._compute_param_combinations(
            parameters, n_samples)

        topic = TopicModel(
            "gensim", LdaModel_class, corpus, dictionary, parameters, sampling_parameters, n_samples, n_initializations, [], [], [], [])

        self.models.append(topic)

        return self

    def load_sklearn_LatentDirichletAllocation(self, LatentDirichletAllocation_class, document_vectors, vectorizer, n_samples, n_initializations=10, setup="simple", custom_params=None):
        parameters = {}
        if setup == "simple":
            parameters = {
                "n_components":
                {"type": int, "mode": "range", "values": [5, 20]}
            }

        if setup == "complex":
            parameters = {
                "n_components":
                {"type": int, "mode": "range", "values": [5, 50]},
                "learning_decayfloat":
                {"type": float, "mode": "range", "values": [0.51, 1]}
            }

        if setup == "custom":
            parameters = custom_params

        sampling_parameters = self._compute_param_combinations(
            parameters, n_samples)

        topic = TopicModel(
            "sklearn", LatentDirichletAllocation_class, document_vectors, vectorizer, parameters, sampling_parameters, n_samples, n_initializations, [], [], [], [])

        self.models.append(topic)

        return self

    def _compute_param_combinations(self, params, n_samples):
        seq = []
        changing_params = list(
            filter(lambda x: params[x]["mode"] is not "fixed", params))
        fixed_params = list(
            filter(lambda x: params[x]["mode"] is "fixed", params))

        for vec in sobol_seq.i4_sobol_generate(len(params), n_samples):
            sample = {}
            for i, name in enumerate(changing_params):
                sample[name] = self._param_to_value(
                    params[name], vec[i])
            for name in fixed_params:
                sample[name] = params[name]["values"]
            seq.append(sample)
        return seq

    def _param_to_value(self, param, sampling):
        if param["mode"] == "range":
            return self._range_to_value(param["values"], sampling, param["type"])
        if param["mode"] == "list":
            return self._list_to_value(param["values"], sampling, param["type"])

    @staticmethod
    def _range_to_value(p_range, sampling, p_type):
        value = p_range[0] + (p_range[1] - p_range[0]) * sampling
        return int(value) if p_type is int else value

    @staticmethod
    def _list_to_value(p_values, sampling, p_type):
        return p_values[min(math.floor(sampling*len(p_values)), len(p_values)-1)]

    def _compute_topic_stability(self):
        for model in self.models:
            self._fetch_top_terms(model, 20)
            model_distributions = self._fetch_term_distributions(model)
            ranking_vecs = self._create_ranking_vectors(model)

            for sample_id, sample in enumerate(model.samples):
                n_topics = 0
                if model.source_lib == "sklearn":
                    n_topics = sample[0].n_components
                if model.source_lib == "gensim":
                    n_topics = sample[0].num_topics

                terms = model.topic_terms[sample_id]
                term_distributions = model_distributions[sample_id]

                kendalls = []
                spearman = []
                jensen = []
                wasserstein = []
                energy = []
                jaccard = []

                report = {}
                report_full = {}

                # Evaluate each topic
                for topic in range(n_topics):
                    sim = pdist(terms[
                        :, topic, :], self._jaccard_similarity)
                    jaccard.append(sim)

                    jen = pdist(term_distributions[
                        :, topic, :], self._jenson_similarity)
                    jensen.append(jen)

                    wasser = pdist(term_distributions[
                        :, topic, :], self._wasserstein_similarity)
                    wasserstein.append(wasser)

                    en = pdist(term_distributions[
                        :, topic, :], self._energy_similarity)
                    energy.append(en)

                    ken = pdist(ranking_vecs[sample_id][
                        :, topic, :], self._kendalls)
                    kendalls.append(ken)

                    spear = pdist(ranking_vecs[sample_id][
                        :, topic, :], self._spear)
                    spearman.append(spear)

                kendalls_ranking = np.array(kendalls)
                spearman_ranking = np.array(spearman)
                jaccard_similarity = np.array(jaccard)
                jensen_similarity = np.array(jensen)
                wasserstein_similarity = np.array(wasserstein)
                energy_similarity = np.array(energy)

                report["model"] = model.topic_model_class
                report["sample_id"] = sample_id
                report["n_topics"] = n_topics
                report["params"] = model.sampling_parameters[sample_id]

                report["jaccard"] = jaccard_similarity.mean()
                report["kendalltau"] = kendalls_ranking.mean()
                report["spearman"] = spearman_ranking.mean()
                report["jensenshannon"] = jensen_similarity.mean()
                report["wasserstein"] = wasserstein_similarity.mean()
                report["energy"] = energy_similarity.mean()

                report_full["model"] = model.topic_model_class
                report_full["sample_id"] = sample_id
                report_full["n_topics"] = n_topics
                report_full["params"] = model.sampling_parameters[sample_id]

                report_full["jaccard"] = {
                    "mean": jaccard_similarity.mean(axis=1),
                    "std": jaccard_similarity.std(axis=1),
                    "min": jaccard_similarity.min(axis=1),
                    "max": jaccard_similarity.max(axis=1),
                }
                report_full["kendalltau"] = {
                    "mean": kendalls_ranking.mean(axis=1),
                    "std": kendalls_ranking.std(axis=1),
                    "min": kendalls_ranking.min(axis=1),
                    "max": kendalls_ranking.max(axis=1),
                }
                report_full["spearman"] = {
                    "mean": spearman_ranking.mean(axis=1),
                    "std": spearman_ranking.std(axis=1),
                    "min": spearman_ranking.min(axis=1),
                    "max": spearman_ranking.max(axis=1),
                }
                report_full["jensenshannon"] = {
                    "mean": jensen_similarity.mean(axis=1),
                    "std": jensen_similarity.std(axis=1),
                    "min": jensen_similarity.min(axis=1),
                    "max": jensen_similarity.max(axis=1),
                }
                report_full["wasserstein"] = {
                    "mean": wasserstein_similarity.mean(axis=1),
                    "std": wasserstein_similarity.std(axis=1),
                    "min": wasserstein_similarity.min(axis=1),
                    "max": wasserstein_similarity.max(axis=1),
                }
                report_full["energy"] = {
                    "mean": energy_similarity.mean(axis=1),
                    "std": energy_similarity.std(axis=1),
                    "min": energy_similarity.min(axis=1),
                    "max": energy_similarity.max(axis=1),
                }

                model.report.append(report)
                model.report_full.append(report_full)

    def rank_models(self, weights={
        "jensenshannon": 1,
        "jaccard": 1,
            "kendalltau": 1}):
        all_reports = []

        for model in self.models:
            all_reports.extend(model.report)

        nf = self._check_for_finite(all_reports)

        if "jaccard" in weights and "jaccard" in nf:
            del weights["jaccard"]
            print("Dropped jaccard from ranking as it has non-finite values.")
        if "jensenshannon" in weights and "jensenshannon" in nf:
            del weights["jensenshannon"]
            print("Dropped jensenshannon from ranking as it has non-finite values.")
        if "wasserstein" in weights and "wasserstein" in nf:
            del weights["wasserstein"]
            print("Dropped wasserstein from ranking as it has non-finite values.")
        if "energy" in weights and "energy" in nf:
            del weights["energy"]
            print("Dropped energy from ranking as it has non-finite values.")
        if "kendalltau" in weights and "kendalltau" in nf:
            del weights["kendalltau"]
            print("Dropped kendalltau from ranking as it has non-finite values.")
        if "spearman" in weights and "spearman" in nf:
            del weights["spearman"]
            print("Dropped spearman from ranking as it has non-finite values.")

        return sorted(all_reports, key=lambda s: self._linear_combination_of_reports(weights, s), reverse=True)

    @staticmethod
    def _linear_combination_of_reports(weights, report):
        total_weight = 0
        combination = 0
        for property, weight in weights.items():
            total_weight += weight
            combination += report[property]

        return combination / total_weight

    @staticmethod
    def _check_for_finite(reports):
        not_finite = set()
        for report in reports:
            if not np.isfinite(report["jaccard"]):
                print("Jaccard similarity has non-finite numbers. Cannot be used.")
                not_finite.add("jaccard")
            if not np.isfinite(report["kendalltau"]):
                print("Kendalltau ranking has non-finite numbers. Cannot be used.")
                not_finite.add("kendalltau")
            if not np.isfinite(report["spearman"]):
                print("Spearman ranking has non-finite numbers. Cannot be used.")
                not_finite.add("spearman")
            if not np.isfinite(report["jensenshannon"]):
                print("Jensenshannon similarity has non-finite numbers. Cannot be used.")
                not_finite.add("jensenshannon")
            if not np.isfinite(report["wasserstein"]):
                print("Wasserstein similarity has non-finite numbers. Cannot be used.")
                not_finite.add("wasserstein")
            if not np.isfinite(report["energy"]):
                print("Energy similarity has non-finite numbers. Cannot be used.")
                not_finite.add("energy")
        return list(not_finite)

    def analyse_sample(self, model, sample_id, feature_names, occurence_percent=1):
        print("Words per topic appearing at least in ",
              occurence_percent, " of all runs.")

        n_topics = len(self.models[model]["samples"][sample_id][0].components_)
        n_runs = len(self.models[model]["samples"][sample_id])

        # Intersect each topic
        for topic in range(n_topics):
            word_list = []
            for terms in self.models[model]["topic_terms"][sample_id]:
                word_list.extend(terms[topic])

            counter = Counter(word_list)
            selected_words = filter(lambda x: counter[x] >= n_runs *
                                    occurence_percent, counter)

            print("Topic - " + str(topic))
            print(" ".join([feature_names[i] + "("+str(counter[i])+")"
                            for i in selected_words]))

    def display_topics(self, model, sample_id, model_number, feature_names, no_top_words):
        for topic_idx, topic in enumerate(self.models[model]["samples"][sample_id][model_number].components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i]
                            for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def _fetch_top_terms(self, model, n_top_terms):
        model_terms = []
        for sample in model.samples:
            terms = []
            for instance in sample:
                if model.source_lib == "sklearn":
                    top_terms = self._get_top_terms(
                        instance, n_top_terms)
                    terms.append(top_terms)
                if model.source_lib == "gensim":
                    top_terms = []
                    for topic_id in range(instance.num_topics):
                        top_terms.append([x[0] for x in instance.get_topic_terms(
                            topic_id, n_top_terms)])
                    terms.append(top_terms)
            model_terms.append(np.array(terms))
        model.topic_terms = model_terms

    def _fetch_term_distributions(self, model):
        model_distributions = []
        for sample in model.samples:
            term_distributions = []
            for instance in sample:
                if model.source_lib == "sklearn":
                    term_distributions.append(
                        instance.components_ / instance.components_.sum(axis=1)[:, np.newaxis])
                if model.source_lib == "gensim":
                    term_distributions.append(instance.get_topics())
            model_distributions.append(np.array(term_distributions))
        return model_distributions

    def _create_ranking_vectors(self, model):
        vocab = set()
        sample_terms = []
        ranking_vecs = []

        for sample in model.samples:
            terms = []
            for instance in sample:
                if model.source_lib == "sklearn":
                    top_terms = self._get_top_terms(
                        instance, self.n_relevant_top_words)
                    terms.append(top_terms)
                    vocab.update([e for l in top_terms for e in l])
                if model.source_lib == "gensim":
                    top_terms = []
                    for topic_id in range(instance.num_topics):
                        top_terms.append([x[0] for x in instance.get_topic_terms(
                            topic_id, self.n_relevant_top_words)])
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
        # Added rounding because without often inf was the result
        # Usage of base 2 algorithm so that the range is [0, 1]
        distance = jensenshannon(a.round(12), b.round(12), base=2)
        return 1 - distance

    @staticmethod
    def _wasserstein_similarity(a, b):
        distance = wasserstein_distance(a, b)
        return 1 - distance

    @staticmethod
    def _energy_similarity(a, b):
        distance = energy_distance(a, b)
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


class TopicModel():
    def __init__(self, source_lib, topic_model_class, data, word_mapping, parameters, sampling_parameters, n_samples, n_initializations, samples, topic_terms, report, report_full) -> None:
        self.source_lib = source_lib
        self.topic_model_class = topic_model_class
        self.data = data
        self.word_mapping = word_mapping
        self.parameters = parameters
        self.sampling_parameters = sampling_parameters
        self.n_samples = n_samples
        self.n_initializations = n_initializations
        self.samples = samples
        self.topic_terms = topic_terms
        self.report = report
        self.report_full = report_full
