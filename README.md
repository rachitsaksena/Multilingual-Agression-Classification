# Low-Resourced Code-Mixed Aggression Classification
Hate Speech Recognition in Code-Mixed Social Media Texts belonging to Low-Resourced Indian Vernacular

## Description
In recent years, there has been a gradual shift from largely static, read-only web to quickly expanding user-generated content on the web due to which the number of user interactions across the globe has sky-rocketed, with a noticeable rise in the incidents of aggression and related activities like trolling, cyberbullying, flaming, hate speech, etc. The increased reach and accessibility of the Internet (especially during the Global Pandemic where the usual workflow has been shifted towards online platforms) such incidents have gained unprecedented power and influence the lives of billions of people posing a formidable challenge given the low resource environment the multilingual setting imposes. Therefore, we aim to tackle the task of domain-specific hate speech recognition in mixed multilingual social media texts (youtube comments).

## Table of Contents
  * [The Dataset](#Data-Acquisition)
  * [Methodology](#Methodology)
  * [License](#license)

## Data Acquisition
The following [dataset](#Data) was acquired from the Shared Task of TRAC-2020, the second workshop on Trolling, Aggression, and Cyberbullying co-located with and organized under the 12th edition of Language Resources and Evaluation Conference (LREC - 2020) at Marseille, France. The [TRAC-2](https://sites.google.com/view/trac2/shared-task) Dataset is publicly available under Creative Commons Noncommercial Share-Alike 4.0 license CC-BY-NC-SA 4.0.

This dataset consists of 5,000 aggression-annotated (‘Overtly Aggressive’, ‘Covertly Aggressive’, or ‘Non-aggressive’) texts each in Hindi (code-switched “Hinglish” and Devanagari script), Bangla (in both Roman and Bangla script), and English from various social media platforms for training and validation.

## Methodology
### Data Preparation
The dataset was observed to be considerably small with a heavy amount of code-mixing in each lexicon and a substantial bias on the 'NAG' class label. The data was thoroughly cleaned: Emojis, punctuations, and special characters were removed, Stop Words were extended and cleaned, and Null enteries were dropped. In order to tackle spelling errors, we used a [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) based spell checker and the Code-Mixed Data was lexically normalized using the [Google Trans API](https://py-googletrans.readthedocs.io/en/latest/) and a Transliteration Dictionary that we created. You can find a more detailed explanation in our [Exploratory Data Analysis Notebook](#EDA,-Data-Visualization,-and-Feature-Engineering.ipynb).

### Feature Generation
Besides using Bag-of-Words (BOW) and Term Frequency–Inverse Document Frequency (TF-IDF) based methods, we tried to capture the semantic footprint of the data in hand using Statistical Topic Modelling and Skip-Gram models. We used Gensim and Mallet's implementation of the Latent Dirichlet Allocation Algorithm to generate Topics and created custom document vectors, which included the sentence length and word count alongside the generated topics. For Skip-gram models, we opted for Google Research's Word2Vec ([Mikolov et al. 2013](https://arxiv.org/pdf/1301.3781.pdf)), Stanford NLP's Global Vectors for Word Representation (GloVe: [Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf)), and Facebook AI Research's FastText ([Bojanowski et al. 2017](https://arxiv.org/pdf/1607.04606.pdf)) in order to capture sub-word level information. The word vector generation is explained in detail in our [Feature Generation and Engineering Notebook](#EDA,-Data-Visualization,-and-Feature-Engineering.ipynb).

## Data Modelling
The results of different Machine Learning models on the test set:

|           Model           | English | Hindi | Bangla |
|:-------------------------:|:-------:|:-----:|:------:|
|    Logistic Regression    |         |       |        |
|  Multinomial Naive Bayes  |         |       |        |
| Support Vector Classifier |         |       |        |

Similarly, for the Deep Learning models:

|    Model   | English | Hindi | Bangla |
|:----------:|:-------:|:-----:|:------:|
|    LSTM    |         |       |        |
| DistilBERT |         |       |        |
|   RoBERTa  |         |       |        |
|   ALBERT   |         |       |        |

The following table elaborates the implementation details of each model:

|          Model          |                                  Dependencies                                 |
|:-----------------------:|:-----------------------------------------------------------------------------:|
| LR, MultinomialNB, SVC  | Self (SciKit Learn for comparison), NLTK, iNLTK, IndicNLP, CoreNLP and Gensim |
|           LSTM          |            PyTorch, TorchText, FastText Pre-Trained Word Embeddings           |
| Tranformer Based Models |    PyTorch, HuggingFace Transformers (Pre-Trained Weights for fine-tuning)    |


