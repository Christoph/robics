# robics
**rob**ustTop**ics** is a library targeted at **non-machine learning experts** interested in building robust
topic models. The main goal is to provide a simple to use framework to check if
a topic model reaches each run the same or at least a similar result.

## Features
- Supports sklearn (LatentDirichletAllocation, NMF) and gensim (LdaModel, ldamulticore, nmf) topic models
- Creates samples based on the [sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence) which requires less samples than grid-search and makes sure the whole parameter space is used which is not sure in random-sampling.
- Simple topic matching between the different re-initializations for each sample using word vector based coherence scores.
- Ranking of all models based on three metrics:
  - [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) of the top n words for each topic
  - Similarity of topic distributions based on the [Jensen Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
  - Ranking correlation of the top n words based on [Kendall's Tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
- Word based analysis of samples and topic model instances.

## Install
- **Python Version:** 3.5+
- **Package Managers:** pip

### pip
Using pip, robics releases are available as source packages and binary wheels:
```
pip install robics
```

## Example
This is a full example including the preprocessing steps. Feel free to adapt it to your own needs.
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models import LdaModel, nmf, ldamulticore
from gensim.utils import simple_preprocess
from gensim import corpora
import spacy
from robics import robustTopics

nlp = spacy.load("en")

# PREPROCESSING
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data[:1000] # Only 1000 dokuments for performance reasons

# sklearn
no_features = 1000

# counts for the NMF model
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# tfidf for the LDA model
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

# gensim
def docs_to_words(docs):
    for doc in docs:
        yield(simple_preprocess(str(doc), deacc=True))

tokenized_data = list(docs_to_words(documents))
dictionary = corpora.Dictionary(tokenized_data)
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

# TOPIC MODELLING
robustTopics = RobustTopics(nlp)

# Load 4 different models
robustTopics.load_gensim_model(
    ldamulticore.LdaModel, corpus, dictionary, n_samples=5, n_initializations=6)
robustTopics.load_gensim_model(
    nmf.Nmf, corpus, dictionary, n_samples=5, n_initializations=6)
robustTopics.load_sklearn_model(
    LatentDirichletAllocation, tf, tf_vectorizer, n_samples=5, n_initializations=6)
robustTopics.load_sklearn_model(NMF, tf, tf_vectorizer, n_samples=5, n_initializations=3)

robustTopics.fit_models()

# ANALYSIS
# Compare different samples
robustTopics.rank_models()

# Look at the topics
robustTopics.display_sample_topics(1, 0, 0.5)
robustTopics.display_run_topics(0, 0, 0, 10)

# Look at the full reports inclusing separate values for each initialization
robustTopics.models[model_id].report_full

# Convert the reports to a pandas dataframe
pd.DataFrame.from_records(robustTopics.models[model_id].report)
```

## Next Steps
- Adding support for more modells if required.
- Writing unit tests.
- Improving the overall performance.
- Implementing the Cv coherence measure from this [paper](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)

## Contribution
I am happy to receive help in any of the things mentioned above or other interesting feature request.