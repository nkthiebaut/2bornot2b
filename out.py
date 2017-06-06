# coding: utf-8
# In[4]:
import re
from nltk.tokenize import wordpunct_tokenize
import numpy as np

from time import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
raw_corpus = open('corpus.txt', 'r', encoding='utf-8-sig').read()
# ## Pre-processing
# In[5]:
title_match_regex = '\n{3,}\s+THE SECRET CACHE\n{3,}.*' # used to remove headers, toc, etc.
corpus = re.search(title_match_regex, raw_corpus, flags=re.M+re.S).group()
corpus = corpus.replace('\n', ' ') 
corpus = re.sub(r' {2,}', ' ', corpus) # replace multiple blanks by one
corpus = corpus.replace('----', '') # remove consecutive hyphens that we'll as a tag for the be verb


# In[6]:

# ## Training set creation
# In[155]:
be_forms = ['am','are','were','was','is','been','being','be']
substitute = '----'
tokens = wordpunct_tokenize(corpus)
def find_targets(tokens):
    return [t for t in tokens if t in be_forms]
    
def remove_targets(tokens):
    """ Replace targets with a substitute in a tokenized text"""
    return [substitute if t in be_forms else t for t in tokens]
targets = find_targets(tokens)
tokens = remove_targets(tokens)
def create_windows(tokens, window_size=5):
    """ Create windows surrouding be forms. """
    for i in range(window_size):
        tokens.insert(0, 'BEGINNING')
    tokens.extend(['END']*window_size)
    contexts = []
    for i, word in enumerate(tokens):
        if word == substitute:
            window = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
            window = ' '.join(window)
            contexts.append(window)    
    return np.array(contexts).reshape(-1, 1)
contexts = create_windows(tokens, window_size=3)


counts = np.unique(targets, return_counts=True)

    
# Replace target names with integer label
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(targets)
# ## Modelling
# In[156]:
contexts_train, contexts_test, y_train, y_test = train_test_split(contexts, y, test_size=0.3)
def avg_accuracy(y_test, y_pred):
    """ Average classes' prediction accuracy. It is a custom score that is not 
    remniscient of the training set class imbalance. """
    conf_mat = confusion_matrix(y_test, y_pred)
    return np.mean(conf_mat.diagonal()/conf_mat.sum(axis=1))
# ### Fit one model
# In[157]:
from sklearn.preprocessing import FunctionTransformer
def texts_left(texts):
    cut_size = len(texts[0][0].split())//2
    return [' '.join(t[0].split()[:cut_size]) for t in texts]
def texts_right(texts):
    cut_size = len(texts[0][0].split())//2
    return [' '.join(t[0].split()[cut_size:]) for t in texts]
# In[163]:
left_vectorizer = TfidfVectorizer(max_df=0.5, max_features=3000, min_df=2, stop_words=None)
right_vectorizer = TfidfVectorizer(max_df=0.5, max_features=3000, min_df=2, stop_words=None)
classifier = LogisticRegression(C=0.5, class_weight='balanced')
pipeline = Pipeline([
    ('union', FeatureUnion(transformer_list=[
                 ('left_bow', Pipeline([('cut_left', FunctionTransformer(texts_left)),
                                        ('vectorizer_left', left_vectorizer)
                                       ]),),
                 ('right_bow', Pipeline([('cut_right', FunctionTransformer(texts_right)),
                                         ('vectorizer_right', right_vectorizer)]))])),
    ('clf', classifier)])
pipeline.fit(contexts_train, y_train)
y_pred_train = pipeline.predict(contexts_train)
y_pred = pipeline.predict(contexts_test)

# ## Prediction on sample
# Read from STDIN (for submission on HackerRank)
import fileinput
f = fileinput.input()
N = int(f.readline())
sample = f.readline()
contexts_sample = create_windows(wordpunct_tokenize(sample))
sample_pred = target_encoder.inverse_transform(pipeline.predict(contexts_sample))
print('\n'.join(sample_pred))
