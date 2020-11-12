from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# PREPROCESSING
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

# Use only 1000 dokuments for performance reasons
documents = dataset.data[:1000]

# Load word vectors
nlp = spacy.load("en_core_web_md")

# Document vectorization using TFIDF
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()


# TOPIC MODELLING
robustTopics = RobustTopics(nlp)

# Load  NMF and LDA models
robustTopics.load_sklearn_model(
    LatentDirichletAllocation, tf, tf_vectorizer, dimension_range=[5, 50], n_samples=4, n_initializations=3)
robustTopics.load_sklearn_model(NMF, tf, tf_vectorizer, dimension_range=[
                                5, 50], n_samples=4, n_initializations=3)

robustTopics.fit_models()

# ANALYSIS
# Rank all computed models based on the topic coherence between the runs
robustTopics.rank_models()

# Look at the topics for a specific sample
robustTopics.display_sample_topics(1, 0, 0.5)
robustTopics.display_run_topics(0, 0, 0, 10)

# Look at the full reports inclusing separate values for each initialization
robustTopics.models[0].report_full

# Use the results in your own workflow
# Export specific model based on the ranked models and the analysis (model_id, sample_id, run_id)
sklearn_Model = robustTopics.export_model(1, 2, 0)

# Convert the reports to a pandas dataframe for further use or export
pd.DataFrame.from_records(robustTopics.models[0].report)
