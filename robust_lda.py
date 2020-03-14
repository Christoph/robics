"""The :mod:`sklearn.robust_lda` module implements a parameter selection based on stability of multiple runs.
"""
# Author: Christoph Kralj <christoph.kralj@gmail.com>
#
# License: MIT

import math
from itertools import combinations
from collections import Counter

import numpy as np
from scipy.spatial.distance import pdist, squareform, jensenshannon
from scipy.stats import kendalltau
import sobol_seq


"""
Sources who show how to use Topic models

https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730


# EXAMPLE CODE

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import LdaModel, nmf, ldamulticore
from gensim.utils import simple_preprocess
from gensim import corpora
import spacy

nlp = spacy.load("en")

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
robustTopics = RobustTopics(nlp, 5)

robustTopics.load_gensim_model(ldamulticore.LdaModel, corpus, dictionary, 5, n_initializations=6)
robustTopics.load_gensim_model(nmf.Nmf, corpus, dictionary, 5, n_initializations=6)
robustTopics.load_sklearn_model(LatentDirichletAllocation, tf, tf_vectorizer, 5, n_initializations=6)
robustTopics.load_sklearn_model(NMF, tf, tf_vectorizer, 5, n_initializations=3)

robustTopics.fit_models()

# Compare different samples
robustTopics.rank_models()

# Look at the topics
robustTopics.display_sample_topics(1, 0, 0.5)
robustTopics.display_run_topics(0, 0, 0, 10)

# Convert the report to a pandas dataframe
pd.DataFrame.from_records(robustTopics.report)

# Print histograms
import plotly.express as px
def show_histograms(self):
    for sample in self.rank_models():
        fig = px.histogram(data_frame=ARRAY, x=0, nbins=10, range_x=[
                        0, 1])
        fig.show()

"""


class RobustTopics():
    """Train and evaluate topic models based on topic robustness measures like jaccard distance of topic terms or topic distributions
    computed on multiple initializations of the same model.

    Examples
    -------
    Initialize a sklearn model.
        >>> from robustTopics import robustTopics
        >>>
        >>> robustTopics = RobustTopics(spacy_nlp)
    Add a sklearn topic model.
        >>> robustTopics.load_sklearn_model(NMF, vecs, vectorizer, n_samples, n_initializations)
    Add a gensim topic model.
        >>> robustTopics.load_gensim_model(ldamulticore.LdaModel, corpus, dictionary, n_samples, n_initializations)
    """

    def __init__(self, spacy_nlp, n_relevant_top_words=20):
        """
        Parameters
        ----------
        spacy_nlp : spacy instance
            A instance of spacy with a already loaded model.
        n_relevant_topics : int, optional
            The bnumber of relevant top words for each topic.
        """

        self.n_relevant_top_words = n_relevant_top_words
        self.nlp = spacy_nlp

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

        print("Fit Models")
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

    def load_gensim_model(self, gensim_model, corpus, dictionary, n_samples, n_initializations=10, setup="simple", custom_params=None):
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
            "gensim", gensim_model, corpus, dictionary, parameters, sampling_parameters, n_samples, n_initializations, [], [], [], [])

        self.models.append(topic)

        print(gensim_model, " successfully loaded.")

        return self

    def load_sklearn_model(self, sklearn_model, document_vectors, vectorizer, n_samples, n_initializations=10, setup="simple", custom_params=None):
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
            "sklearn", sklearn_model, document_vectors, vectorizer, parameters, sampling_parameters, n_samples, n_initializations, [], [], [], [])

        self.models.append(topic)

        print(sklearn_model, " successfully loaded.")

        return self

    def rank_models(self, weights={
        "jensenshannon": 1,
        "jaccard": 1,
            "kendalltau": 1}):
        all_reports = []

        for model in self.models:
            all_reports.extend(model.report)

        return sorted(all_reports, key=lambda s: self._linear_combination_of_reports(weights, s), reverse=True)

    # def display_model_topics(self):
    #     pass

    def display_sample_topics(self, model_id, sample_id, occurence_percent=1):
        model = self.models[model_id]
        n_runs = len(model.samples[sample_id])

        print("Words per topic appearing at least in ",
              round(occurence_percent*100), "% of all runs(", n_runs, ").")

        n_topics = 0
        if model.source_lib == "sklearn":
            n_topics = len(model.samples[sample_id][0].components_)
        if model.source_lib == "gensim":
            n_topics = model.samples[sample_id][0].num_topics

        # Intersect each topic
        for topic in range(n_topics):
            word_list = []
            for terms in model.topic_terms[sample_id]:
                word_list.extend(terms[topic])

            counter = Counter(word_list)
            selected_words = filter(lambda x: counter[x] >= n_runs *
                                    occurence_percent, counter)

            print("Topic " + str(topic))
            print_data = {}
            for word in selected_words:
                count = counter[word]

                if count in print_data:
                    print_data[count].append(word)
                else:
                    print_data[count] = [word]

            for count, words in print_data.items():
                print("In", count, "runs:", " ".join(words))

    def display_run_topics(self, model_id, sample_id, run_number, no_top_words):
        model = self.models[model_id]

        for topic_idx, topic in enumerate(model.topic_terms[sample_id][run_number]):
            print("Topic %d:" % (topic_idx))
            print(" ".join(topic))

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

    def _topic_matching(self, n_topics, model, sample_id, terms, term_distributions, ranking_vecs):
        print("Topic Matching for each Initialization")
        run_coherences = []

        print("Compute Topic Coherence:", end="")
        for run_number in range(model.n_initializations):
            print(model.n_initializations-run_number, end="")
            topic_terms = model.topic_terms[sample_id][run_number]

            run_coherences.append(self.compute_tcw2c(n_topics, topic_terms))

        best_run = run_coherences.index(max(run_coherences))
        reference_topic = terms[best_run]

        print("")
        print("Match Topics:", end="")
        # Create mapping for all topics across all runs
        for run in range(model.n_initializations):
            print(model.n_initializations-run, end="")
            topics = np.concatenate((reference_topic, terms[
                run, :, :]), axis=0)
            sim = squareform(pdist(topics, self._jaccard_similarity))[
                :n_topics, n_topics:]

            run_mapping = []

            # Map reference topics to run topics based on highest similarity
            for topic in range(n_topics):
                # [1] is the reference topic index and [0] is the other index
                first_highest_index = np.argwhere(sim == sim.max())[0]
                run_mapping.append(
                    [first_highest_index[1], first_highest_index[0]])

                # Delete topic with highest value
                sim[:, first_highest_index[1]] = -1
                sim[first_highest_index[0], :] = -1

            run_mapping.sort()
            sort_indices = np.array(run_mapping)[:, 1]

            # Sort all runs
            terms[run] = terms[run, sort_indices]
            term_distributions[run] = term_distributions[run, sort_indices]
            ranking_vecs[run] = ranking_vecs[run, sort_indices]

        print("")
        return np.array(run_coherences)

    # Ideas from here: https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
    def compute_tcw2c(self, n_topics, topic_terms, max_terms=5):
        # max_terms=20 results in roughly 15 times the time
        # max_terms=10 results in roughly 4 times the time
        # max_terms=5 results in roughly 0.75s per topic

        total_coherence = []
        for topic in range(n_topics):
            pairs = []
            processed_topic_terms = (self.nlp(str(t))
                                     for t in topic_terms[topic][:max_terms])

            for pair in combinations(processed_topic_terms, 2):
                tokens = pair
                if tokens[0].has_vector and tokens[1].has_vector:
                    pairs.append(tokens[0].similarity(tokens[1]))
                else:
                    print("One of", tokens, "has no vector.")

            total_coherence.append(sum(pairs) / len(pairs))
        return sum(total_coherence) / n_topics

    def _compute_topic_stability(self):
        print("Evaluate Models")
        for model_id, model in enumerate(self.models):
            print("Model: ", model.topic_model_class)
            self._fetch_top_terms(model, 20)
            model_distributions = self._fetch_term_distributions(model)
            all_ranking_vecs = self._create_ranking_vectors(model)

            for sample_id, sample in enumerate(model.samples):
                print("Sample", sample_id+1, "of",
                      len(model.samples), " Samples")
                n_topics = 0
                if model.source_lib == "sklearn":
                    n_topics = sample[0].n_components
                if model.source_lib == "gensim":
                    n_topics = sample[0].num_topics

                terms = model.topic_terms[sample_id]
                term_distributions = model_distributions[sample_id]
                ranking_vecs = all_ranking_vecs[sample_id]

                kendalls = []
                jensen = []
                jaccard = []

                report = {}
                report_full = {}

                run_coherence = self._topic_matching(
                    n_topics, model, sample_id, terms, term_distributions, ranking_vecs)

                # Evaluate each topic
                for topic in range(n_topics):
                    sim = pdist(terms[
                        :, topic, :], self._jaccard_similarity)
                    jaccard.append(sim)

                    jen = pdist(term_distributions[
                        :, topic, :], self._jenson_similarity)
                    jensen.append(jen)

                    ken = pdist(ranking_vecs[
                        :, topic, :], self._kendalls)
                    kendalls.append(ken)

                kendalls_ranking = np.array(kendalls)
                jaccard_similarity = np.array(jaccard)
                jensen_similarity = np.array(jensen)

                report["model"] = model.topic_model_class
                report["model_id"] = model_id
                report["sample_id"] = sample_id
                report["n_topics"] = n_topics
                report["params"] = model.sampling_parameters[sample_id]

                report["topic_coherence"] = run_coherence.mean()

                report["jaccard"] = jaccard_similarity.mean()
                report["kendalltau"] = kendalls_ranking.mean()
                report["jensenshannon"] = jensen_similarity.mean()

                report_full["model"] = model.topic_model_class
                report_full["model_id"] = model_id
                report_full["sample_id"] = sample_id
                report_full["n_topics"] = n_topics
                report_full["params"] = model.sampling_parameters[sample_id]

                report_full["topic_coherence"] = {
                    "topic_coherences": run_coherence,
                }
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
                report_full["jensenshannon"] = {
                    "mean": jensen_similarity.mean(axis=1),
                    "std": jensen_similarity.std(axis=1),
                    "min": jensen_similarity.min(axis=1),
                    "max": jensen_similarity.max(axis=1),
                }

                model.report.append(report)
                model.report_full.append(report_full)
            print("")

    @staticmethod
    def _linear_combination_of_reports(weights, report):
        total_weight = 0
        combination = 0
        for property, weight in weights.items():
            total_weight += weight
            combination += report[property]

        return combination / total_weight

    def _fetch_top_terms(self, model, n_top_terms):
        model_terms = []
        for sample in model.samples:
            terms = []
            for instance in sample:
                if model.source_lib == "sklearn":
                    top_terms = self._get_top_terms(
                        model, instance, n_top_terms)
                    terms.append(top_terms)
                if model.source_lib == "gensim":
                    top_terms = []
                    for topic_id in range(instance.num_topics):
                        top_terms.append([model.word_mapping[x[0]] for x in instance.get_topic_terms(
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
                    top_terms = self._get_top_terms(model,
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
    def _jenson_similarity(a, b):
        # Added rounding because without often inf was the result
        # Usage of base 2 algorithm so that the range is [0, 1]
        distance = jensenshannon(a.round(12), b.round(12), base=2)
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
    def _get_top_terms(model, instance, n_terms):
        feature_names = model.word_mapping.get_feature_names()
        topic_terms = []
        for topic in instance.components_:
            topic_terms.append([feature_names[i]
                                for i in topic.argsort()[:-n_terms - 1:-1]])

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
