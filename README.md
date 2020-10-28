# Low-Resourced Code-Mixed Aggression Classification
Hate Speech Recognition in Code-Mixed Social Media Texts belonging to Low-Resourced Indian Vernacular

## Table of Contents
  1. [Motivation](#Motivation)
  2. [The Dataset](#Data-Acquisition)
  3. [Prerequisites and Dependencies](#Prerequisites-and-Dependencies)
  4. [Methodology](#Methodology)
  5. [License](https://github.com/rachitsaksena/Multilingual-Agression-Classification/blob/master/LICENSE)

## Motivation
In recent years, there has been a gradual shift from largely static, read-only web to quickly expanding user-generated content on the web due to which the number of user interactions across the globe has sky-rocketed, with a noticeable rise in the incidents of aggression and related activities like trolling, cyberbullying, flaming, hate speech, etc. The increased reach and accessibility of the Internet (especially during the Global Pandemic where the usual workflow has been shifted towards online platforms) such incidents have gained unprecedented power and influence the lives of billions of people posing a formidable challenge given the low resource environment the multilingual setting imposes. Therefore, we aim to tackle the task of domain-specific hate speech recognition in mixed multilingual social media texts (youtube comments).

## Data Acquisition
The following [dataset](https://github.com/rachitsaksena/Multilingual-Agression-Classification/tree/master/Data) was acquired from the Shared Task of TRAC-2020, the second workshop on Trolling, Aggression, and Cyberbullying co-located with and organized under the 12th edition of Language Resources and Evaluation Conference (LREC - 2020) at Marseille, France. The [TRAC-2](https://sites.google.com/view/trac2/shared-task) Dataset is publicly available under Creative Commons Noncommercial Share-Alike 4.0 license CC-BY-NC-SA 4.0.

This dataset consists of 5,000 aggression-annotated (‘Overtly Aggressive’, ‘Covertly Aggressive’, or ‘Non-aggressive’) texts each in Hindi (code-switched “Hinglish” and Devanagari script), Bangla (in both Roman and Bangla script), and English from various social media platforms for training and validation.

## Prerequisites and Dependencies
Ensure that you have [Python 3.5+](https://www.python.org/downloads/) and [pip3](https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py) installed on your specific OS distribution.

Clone the repository and create a virtual environment.
```shell
git clone https://github.com/rachitsaksena/Multilingual-Agression-Classification/
python -m venv .env
source .env/bin/activate
```

Install [dependencies](#requirements.txt) directly by
```shell
cd Multilingual-Agression-Classification/
pip3 install -r requirements.txt
``` 
**Disclaimer:** _This may take a while._

**Downloadables:**

Enter the following command within your shell
```
python -m spacy download en_core_web_sm
```

Enter the Python console using `python3` , input the following and exit the shell
```python
import nltk
nltk.download(‘stopwords’)

import stanfordnlp
stanfordnlp.download('en')
stanfordnlp.download('hi')
```

Download and Extract the pre-trained [Wiki News FastText Word Vectors](https://fasttext.cc/docs/en/english-vectors.html) into the [Cache/Embeddings/](https://github.com/rachitsaksena/Multilingual-Agression-Classification/tree/master/Cache/Embeddings) directory of the repo.

## Methodology
### Data Preparation
The dataset was observed to be considerably small with a heavy amount of code-mixing in each lexicon and a substantial bias on the 'NAG' class label. The data was thoroughly cleaned: Emojis, punctuations, and special characters were removed, Stop Words were extended and cleaned, and Null enteries were dropped. In order to tackle spelling errors, we used a [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) based spell checker and the Code-Mixed Data was lexically normalized using the [Google Trans API](https://py-googletrans.readthedocs.io/en/latest/) and a Transliteration Dictionary that we created. You can find a more detailed explanation in our [Exploratory Data Analysis Notebook](https://github.com/rachitsaksena/Multilingual-Agression-Classification/blob/master/EDA%2C%20Data%20Visualization%2C%20and%20Feature%20Engineering.ipynb).

### Feature Generation
Besides using Bag-of-Words (BOW) and Term Frequency–Inverse Document Frequency (TF-IDF) based methods, we tried to capture the semantic footprint of the data in hand using Statistical Topic Modelling and Skip-Gram models. We used Gensim and Mallet's implementation of the Latent Dirichlet Allocation Algorithm to generate Topics and created custom document vectors, which included the sentence length and word count alongside the generated topics. For Skip-gram models, we opted for Google Research's Word2Vec ([Mikolov et al. 2013](https://arxiv.org/pdf/1301.3781.pdf)), Stanford NLP's Global Vectors for Word Representation (GloVe: [Pennington et al. 2014](https://nlp.stanford.edu/pubs/glove.pdf)), and Facebook AI Research's FastText ([Bojanowski et al. 2017](https://arxiv.org/pdf/1607.04606.pdf)) in order to capture sub-word level information. The word vector generation is explained in detail in our [Feature Generation and Engineering Notebook](https://github.com/rachitsaksena/Multilingual-Agression-Classification/blob/master/EDA%2C%20Data%20Visualization%2C%20and%20Feature%20Engineering.ipynb).

### Adversarial Validation
Given the 'NAG' bias, models, howsoever simple they may be in generalizing, are bound to overfit. In order to prevent that, we use a Kaggle Favourite "Adversarial Validation" in order to make the test and training sets undifferentiable.

### Data Modelling
We chose simple models like Logistic Regression, MultinomialNB, and SVC that wouldn't get overwhelmed by the imbalance and generalize easily. We also fine-tuned a BERT instance due to it's recorded State-of-the-Art performance with multilingual text classification tasks. The results of different Machine Learning models on the test set:
|           Model           | English | Hindi | Bangla |
|:-------------------------:|:-------:|:-----:|:------:|
|    Logistic Regression    |         |       |        |
|  Multinomial Naive Bayes  |         |       |        |
| Support Vector Classifier |         |       |        |
|             BERT          |         |       |        |

The following table elaborates the implementation details of each model:
|          Model          |                                               Dependencies                                              |
|:-----------------------:|:-------------------------------------------------------------------------------------------------------:|
| LR, MultinomialNB, SVC  | Self (SciKit Learn for comparison and post-processing utils), NLTK, iNLTK, IndicNLP, CoreNLP and Gensim |
| Tranformer Based Models |                                     Tensorflow, Tensorflow's BERT Base                                  |

**... was chosen finally because of ...**

### Future Work
 * Reliable lemmatization for Hindi and Bangla
 * Using huggingface's BERT variants (DistilBERT, RoBERTa, ALBERT, etc.)
 * [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
 * Implement bagging for BERT
 * Using other sentence embeddings for BERT
    1. Doc2Vec
    2. SentenceBERT
    3. InferSent
    4. Universal Sentence Encoder
    5. ELMo's Sentence Embeddings
 * Increase Test and Training size with TRAC-1 Data

## TO-DO List

#### Tasks
- [ ] Generating Word Embeddings for all sets
- [X] Lang generalized algos
- [ ] Clean Repo, Add Images

#### Bias Regularization
- [ ] Baseline
- [ ] Effect of upsampling
- [ ] Effect of downsampling
- [ ] Weighted classification
- [X] Adversarial Validation

#### Corpus Cleaning and Pre-Embedding EDA
- [ ] Remove corpus specific stop words
- [ ] Class based Interdependence
- [ ] Lemmatization
- [ ] Handling NaN values

#### Fine Tuning
- [ ] t-SNE perplexity vals
- [ ] BOW, TFIDF (min, max df)
- [ ] LDA vecs (after pre embedding cleaning) - test baseline as well
- [ ] Word2Vec(with/without shuffle, avg vs tfidf) (pre trained and custom)
- [ ] GloVe (window, epochs, etc) (pre trained and custom)
- [ ] FastText (pre trained and custom)
- [ ] Meta Feature Engineering

#### Bugs
- [ ] Bad hyperlink cleaning (httpsyoutbe)
- [ ] Punctuations (!, etc)
- [ ] Numbers
- [X] Bad transliteration dict (Use Sets)
- [X] Bad language tagging and translation (some Bangla got through)
- [ ] Bad spelling corrections {couture (is already a word) instead of culture}
- [ ] Handling NaN values created due to Bad Lexical Normalization
- [ ] Fix DeEmojify (Hug emoji, Peace sign, etc)
