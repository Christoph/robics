# robics
**rob**ustTop**ics** is a library targeted at **non-machine learning experts** interested in building robust
topic models. The main goal is to provide a simple to use framework to check if
a topic model reaches each run the same or at least a similar result.

## Features
- Supports sklearn (LatentDirichletAllocation, NMF) and gensim (LdaModel, ldamulticore, nmf) topic models
- Creates samples based on the [sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence) which requires less samples than grid-search and makes sure the whole parameter space is used which is not sure in random-sampling.
- Simple topic matching between the different re-initializations for each sample using word vector based coherence scores.
- Ranking of all models based on four metrics:
  - [Jaccard distance](https://en.wikipedia.org/wiki/Jaccard_index) of the top n words for each topic
  - Similarity of topic distributions based on the [Jensen Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
  - Ranking correlation of the top n words based on [Kendall's Tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)
  - Word vector based coherence score (simple version of the TC-W2V)
- Word based analysis of samples and topic model instances.

## Install
- **Python Version:** 3.5+
- **Package Managers:** pip

### pip
Using pip, robics releases are available as source packages and binary wheels:
```
pip install robics
```

## Examples
Test dataset from sklearn
```python
from sklearn.datasets import fetch_20newsgroups

# PREPROCESSING
dataset = fetch_20newsgroups(
    shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

# Only 1000 dokuments for performance reasons
documents = dataset.data[:1000]
```

Load word vectors used for coherence computation
```python
import spacy

nlp = spacy.load("en_core_web_md")
```


Detect robust sklearn topic models
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from robics import RobustTopics

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
robustTopics.load_sklearn_model(NMF, tf, tf_vectorizer, dimension_range=[5, 50], n_samples=4, n_initializations=3)

# Fit the models - Warning, this might take a lot of time based on the number of samples (n_models*n_sample*n_initializations)
robustTopics.fit_models()

```

Analysis
We start by looking at the ranking of all models

```python
robustTopics.rank_models()

    # Output:
    [{'model': sklearn.decomposition._nmf.NMF,
    'model_id': 1,
    'sample_id': 0,
    'n_topics': 27,
    'params': {'n_components': 27},
    'topic_coherence': 0.3184709231908624,
    'jaccard': 0.9976484420928865,
    'kendalltau': 0.9987839560655095,
    'jensenshannon': 0.9999348301145501},
    {'model': sklearn.decomposition._nmf.NMF,
    'model_id': 1,
    'sample_id': 3,
    'n_topics': 21,
    'params': {'n_components': 21},
    'topic_coherence': 0.31157823571739196,
    'jaccard': 1.0,
    'kendalltau': 1.0,
    'jensenshannon': 0.9999784246484099},
    ...
     {'model': sklearn.decomposition._lda.LatentDirichletAllocation,
    'model_id': 0,
    'sample_id': 1,
    'n_topics': 38,
    'params': {'n_components': 38},
    'topic_coherence': 0.30808284548733383,
    'jaccard': 0.07074176815158277,
    'kendalltau': 0.1023597955928783,
    'jensenshannon': 0.22596536037699871}]

```

The top model is the NMF model with 27 topics (model_id 1 and sample_id 0). The next step is to look at the model on a word level.

```python
robustTopics.display_sample_topic_consistency(model_id=1, sample_id=0)

    # Output:
    Words in 3 out of 3 runs:
    ['edu', 'graphics', 'pub', 'mail', 'ray', '128', 'send', '3d', 'ftp', 'com', 'objects', 'server', 'amiga', 'image', 'archie', 'files', 'file', 'images', 'archive', 'package', 'section', 'firearm', 'weapon', 'military', 'license', 'shall', 'dangerous', 'person', 'division', 'application', 'means', 'device', 'use', 'following', 'issued', 'state', 'act',...
    Words in 2 out of 3 runs:
    ['know']

    Words in 1 out of 3 runs:
    ['goals']

```

Most of the words are in all of the three re-initializations. Only 'know' and 'goals' are inconsistent.

In comparison lets look at the model performing worst:
```python
robustTopics.display_sample_topic_consistency(model_id=0, sample_id=1)

    # Output:
    Words in 3 out of 3 runs:
    ['know', 'know', 'people', 'use', 'drive', 'think', 'just', 'like', 'just', 'people', 'just', 'know', 'think', 'good', 'think', 'like', 'people', 'know', 'don', 'think', 'like', 'just', 'don', 'know', 'just', 'don', 'think', 'like', 'windows', 'like', 'don']

    Words in 2 out of 3 runs:
    ['just', 'think', 'people', 'dc', 'just', 'like', 'want', 'said', 'use', 'said', 'local', 'say', 'god', 'shall', 'rights', 'know', 'like', 'window', 'like', 'just', 'time', 'new', 'like', 'just', 'good', 'don', 'used', 'does', 'think', 'like', 'new', 'state', 'like', 'contact', 'know', 'bike', 'just', 'like', 'year', 'data', 'use', 'way', 've', 'people', 'don', 'know', 'didn', 'years', 'little', 'rocket', 'like', 'generation', 'build', 'll', 'max', 'ssrt', 'just', 'time', 'using', 'edu', 'ftp', 'file', 'available', 'server', '10', 'good', 'new', 'know', 'people', 'way', 'know', 'good', 'don', 'just', 'like', 'time', 'like', 'don', 'insurance', 'year', 'time', 'car', 'years', 'people', 'say', 'new', 'new', 'need', 'just', 'think', 'like', 'good', 'just', 'don', 'work', 'time', 'government', 'rights', 'make', 'people', 'edu', 'com', 'graphics', 'time', 'good', 'people', 'need', 'like', 'don', 'know', 'just', 'people', 'server', 'edu', 'file', 'video', 'support', 'mit', 'ftp', 'linux', 'time', 'binaries', 'information', 'available', 'new', 'greek', '10', 'just', 'just', 'does', 'know', 'use', 'need', 'like', 'don', 'use', 'know', 'think', 'like', 'new', 'like', 'just', 'edu', 've', 'does', 'problem', 'think', 'just', 'way', 'power', 'wrong', 'edu', 'points', 'point', 'just', 'hp', 'don', 'good', 'people', 'phone', 'food', 'just', 'bit', 'good', 'know', 'just', 'like', 'use', 'people', 'used', 'going', 'people', 'know', 'time', 'say', 'use', 'drive', '10', 'like', 'observations', 'think', 'don', 'want', 'thing', 'know', 'll', 'good', 'like', 'mm', 'used', 'say']

    Words in 1 out of 3 runs:
    ['jews', 'israel', 'true', 'state', 'year',
    ...
```
The worst model has most words in the 1 out of 3 runs section and only filler words are consistent between the runs.
"know" appears five times which means, it is in five different topics in all three runs a top word.

Let us look now at the topic in the top and bottom models.
```python
# Top model
robustTopics.display_sample_topics(1, 0)

    # Output
    Topic 0
    In 3 runs: edu graphics pub mail ray 128 send 3d ftp com objects server amiga image archie files file images archive package
    Topic 1
    In 3 runs: section firearm weapon military license shall dangerous person division application means device use following issued state act designed code automatic
    Topic 2
    In 3 runs: aids health care said children medical infected new patients disease 1993 10 information research april national study trials service number
    Topic 3
    In 3 runs: god good people just suppose brothers makes fisher like does jews joseph did worship judaism right instead jesus end read
    Topic 4
    In 3 runs: server support edu file 386bsd ftp mit binaries vga information supported linux svga readme available os new faq files video
    Topic 5
    In 3 runs: edu com mil navy cs vote misc votes health ca hp nrl gov email cc creation au john thomas uk
    Topic 6
    In 3 runs: probe space titan earth orbiter launch mission jupiter orbit atmosphere 93 saturn gravity 10 surface satellite ray 12 possible 97

# Bottom model
robustTopics.display_sample_topics(0, 1)

    # Output
    Topic 0
    In 3 runs: know
    Topic 1
    In 3 runs: know
    Topic 2
    Topic 3
    Topic 4
    In 3 runs: people
    Topic 5
    Topic 6
    In 3 runs: use
    Topic 7
    Topic 8
    In 3 runs: drive think just
```
The top model produces over all runs consistent topic with words that are connected.
In strong contrast to the bottom model, which has very little consistent words and these are provide little information.

We want to take a look at the topics of one of the instances from the top model (model 1, sample 0, instance 0).

```python
robustTopics.display_run_topics(1, 0, 0)

    # Output
    Topic 0:
    edu graphics pub mail ray 128 send 3d ftp com objects server amiga image archie files file images archive package
    Topic 1:
    section firearm weapon military license shall dangerous person division application means device use following issued state act designed code automatic
    Topic 2:
    aids health care said children medical infected new patients disease 1993 10 information research april national study trials service number
    Topic 3:
    god good people just suppose brothers makes fisher like does jews joseph did worship judaism right instead jesus end read
    Topic 4:
    server support edu file 386bsd ftp mit binaries vga information supported linux svga readme available os new faq files video
    Topic 5:
    edu com mil navy cs vote misc votes health ca hp nrl gov email cc creation au john thomas uk
    Topic 6:
    probe space titan earth orbiter launch mission jupiter orbit atmosphere 93 saturn gravity 10 surface satellite ray 12 possible 97
```

If the output is meaningful we can now use this model for further use
or export more indept information for additional analysis.

```python
# Use the results in your own workflow
# Export specific model based on the ranked models and the analysis (model_id, sample_id, run_id)
sklearn_model = robustTopics.export_model(1,0,0)

# Look at the full reports inclusing separate values for each initialization
robustTopics.models[1].report_full

# Convert the full report to a pandas dataframe for further use or export
import pandas as pd
report = pd.DataFrame.from_records(robustTopics.models[model_id].report) # or report_full
```

Gensim topic models are also supported. Following is a example how to setup a
simple pipeline. The analysis steps are exactly the same as above.
```python
from gensim.models import LdaModel, nmf, ldamulticore
from gensim.utils import simple_preprocess
from gensim import corpora
from robics import RobustTopics

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
    ldamulticore.LdaModel, corpus, dictionary, dimension_range=[5, 50], n_samples=4, n_initializations=3)
robustTopics.load_gensim_model(
    nmf.Nmf, corpus, dictionary, dimension_range=[5, 50], n_samples=4, n_initializations=3)

robustTopics.fit_models()

# Same analysis steps as in the sklearn example
```

## Next Steps
- Visual interface
- Adding support for more modells if required.
- Add logging
- Writing unit tests.
- Improving the overall performance.
- Implementing the Cv coherence measure from this [paper](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf)

## Contribution
I am happy to receive help in any of the things mentioned above or other interesting feature request.