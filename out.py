# coding: utf-8
# In[20]:
import re
from nltk.tokenize import wordpunct_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
raw_corpus = open('corpus.txt', 'r', encoding='utf-8-sig').read()
# ## Pre-processing
# In[4]:
title_match_regex = '\n{3,}\s+THE SECRET CACHE\n{3,}.*' # used to remove headers, toc, etc.
corpus = re.search(title_match_regex, raw_corpus, flags=re.M+re.S).group()
corpus = corpus.replace('\n', ' ') 
corpus = re.sub(r' {2,}', ' ', corpus) # replace multiple blanks by one
corpus = corpus.replace('----', '') # remove consecutive hyphens that we'll as a tag for the be verb


# In[5]:

# ## Training set creation
# In[18]:
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
    contexts = []
    for i, word in enumerate(tokens):
        if word == substitute:
            window = tokens[i-window_size:i] + tokens[i+1:i+window_size+1]
            window = ' '.join(window)
            contexts.append(window)    
    return contexts
contexts = create_windows(tokens)


counts = np.unique(targets, return_counts=True)
#for form, count in zip(counts[0], counts[1]):

# In[15]:
# Replace target names with integer label
le = LabelEncoder()
y = le.fit_transform(targets)
# ## Modelling
# In[19]:
contexts_train, contexts_test, y_train, y_test = train_test_split(contexts, y, test_size=0.3)
# In[24]:
# Vectorize context features
#vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english')
vectorizer = CountVectorizer()
classifier = LogisticRegression()
pipe = make_pipeline(vectorizer, classifier)
pipe.fit(contexts_train, y_train)
y_pred = pipe.predict(contexts_test)


# ## Prediction on sample
# In[27]:
sample = open('sample_input.txt', 'r', encoding='utf-8-sig').read().splitlines()
N = int(sample[0])
sample = sample[1]
sample = wordpunct_tokenize(sample)
contexts_sample = create_windows(sample)
sample_pred = le.inverse_transform(pipe.predict(contexts_sample))
print('\n'.join(sample_pred))
