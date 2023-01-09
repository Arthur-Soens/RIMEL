#!/usr/bin/env python
# coding: utf-8

# # Calculating the Bilingual Evaluation Understudy (BLEU) score: Ungraded Lab

# In this ungraded lab, we will implement a popular metric for evaluating the quality of machine-translated text: the BLEU score proposed by Kishore Papineni, et al. In their 2002 paper ["BLEU: a Method for Automatic Evaluation of Machine Translation"](https://www.aclweb.org/anthology/P02-1040.pdf), the BLEU score works by comparing "candidate" text to one or more "reference" translations. The result is better the closer the score is to 1. Let's see how to get this value in the following sections.

# # Part 1:  BLEU Score

# ## 1.1  Importing the Libraries
# 
# We will first start by importing the Python libraries we will use in the first part of this lab. For learning, we will implement our own version of the BLEU Score using Numpy. To verify that our implementation is correct, we will compare our results with those generated by the [SacreBLEU library](https://github.com/mjpost/sacrebleu). This package provides hassle-free computation of shareable, comparable, and reproducible BLEU scores. It also knows all the standard test sets and handles downloading, processing, and tokenization.

# In[1]:


import numpy as np                  # import numpy to make numerical computations.
import nltk                         # import NLTK to handle simple NL tasks like tokenization.
nltk.download("punkt")
from nltk.util import ngrams
from collections import Counter     # import the Counter module.
!pip3 install 'sacrebleu'           # install the sacrebleu package.
import sacrebleu                    # import sacrebleu in order compute the BLEU score.
import matplotlib.pyplot as plt     # import pyplot in order to make some illustrations.

# ## 1.2  Defining the BLEU Score
# 
# You have seen the formula for calculating the BLEU score in this week's lectures. More formally, we can express the BLEU score as:
# 
# $$BLEU = BP\Bigl(\prod_{i=1}^{4}precision_i\Bigr)^{(1/4)}$$
# 
# with the Brevity Penalty and precision defined as:
# 
# $$BP = min\Bigl(1, e^{(1-({ref}/{cand}))}\Bigr)$$
# 
# $$precision_i = \frac {\sum_{snt \in{cand}}\sum_{i\in{snt}}min\Bigl(m^{i}_{cand}, m^{i}_{ref}\Bigr)}{w^{i}_{t}}$$
# 
# where:
# 
# * $m^{i}_{cand}$, is the count of i-gram in candidate matching the reference translation.
# * $m^{i}_{ref}$, is the count of i-gram in the reference translation.
# * $w^{i}_{t}$, is the total number of i-grams in candidate translation.

# ## 1.3 Explaining the BLEU score

# ### Brevity Penalty (example):

# In[2]:


ref_length = np.ones(100)
can_length = np.linspace(1.5, 0.5, 100)
x = ref_length / can_length
y = 1 - x
y = np.exp(y)
y = np.minimum(np.ones(y.shape), y)

# Code for in order to make the plot
fig, ax = plt.subplots(1)
lines = ax.plot(x, y)
ax.set(
    xlabel="Ratio of the length of the reference to the candidate text",
    ylabel="Brevity Penalty",
)
plt.show()

# The brevity penalty penalizes generated translations that are too short compared to the closest reference length with an exponential decay. The brevity penalty compensates for the fact that the BLEU score has no recall term.

# ### N-Gram Precision (example):

# In[3]:


data = {"1-gram": 0.8, "2-gram": 0.7, "3-gram": 0.6, "4-gram": 0.5}
names = list(data.keys())
values = list(data.values())

fig, ax = plt.subplots(1)
bars = ax.bar(names, values)
ax.set(ylabel="N-gram precision")

plt.show()

# The n-gram precision counts how many unigrams, bigrams, trigrams, and four-grams (i=1,...,4) match their n-gram counterpart in the reference translations. This term acts as a precision metric. Unigrams account for adequacy while longer n-grams account for fluency of the translation. To avoid overcounting, the n-gram counts are clipped to the maximal n-gram count occurring in the reference ($m_{n}^{ref}$). Typically precision shows exponential decay with the with the degree of the n-gram.

# ### N-gram BLEU score (example):

# In[4]:


data = {"1-gram": 0.8, "2-gram": 0.77, "3-gram": 0.74, "4-gram": 0.71}
names = list(data.keys())
values = list(data.values())

fig, ax = plt.subplots(1)
bars = ax.bar(names, values)
ax.set(ylabel="Modified N-gram precision")

plt.show()

# When the n-gram precision is multiplied by the BP, then the exponential decay of n-grams is almost fully compensated. The BLEU score corresponds to a geometric average of this modified n-gram precision.

# ## 1.4 Example Calculations of the BLEU score

# In this example we will have a reference translation and 2 candidates translations. We will tokenize all sentences using the NLTK package introduced in Course 2 of this NLP specialization.

# In[5]:


reference = "The NASA Opportunity rover is battling a massive dust storm on planet Mars."
candidate_1 = "The Opportunity rover is combating a big sandstorm on planet Mars."
candidate_2 = "A NASA rover is fighting a massive storm on planet Mars."

tokenized_ref = nltk.word_tokenize(reference.lower())
tokenized_cand_1 = nltk.word_tokenize(candidate_1.lower())
tokenized_cand_2 = nltk.word_tokenize(candidate_2.lower())

print(f"{reference} -> {tokenized_ref}")
print("\n")
print(f"{candidate_1} -> {tokenized_cand_1}")
print("\n")
print(f"{candidate_2} -> {tokenized_cand_2}")

# ### STEP 1: Computing the Brevity Penalty

# In[6]:


def brevity_penalty(reference, candidate):
    ref_length = len(reference)
    can_length = len(candidate)

    # Brevity Penalty
    if ref_length > can_length:
        BP = 1
    else:
        penalty = 1 - (ref_length / can_length)
        BP = np.exp(penalty)

    return BP

# ### STEP 2: Computing the Precision

# In[7]:


def clipped_precision(reference, candidate):
    """
    Bleu score function given a original and a machine translated sentences
    """

    clipped_precision_score = []

    for i in range(1, 5):
        candidate_n_gram = Counter(
            ngrams(candidate, i)
        )  # counts of n-gram n=1...4 tokens for the candidate
        reference_n_gram = Counter(
            ngrams(reference, i)
        )  # counts of n-gram n=1...4 tokens for the reference

        c = sum(
            reference_n_gram.values()
        )  # sum of the values of the reference the denominator in the precision formula

        for j in reference_n_gram:  # for every n_gram token in the reference
            if j in candidate_n_gram:  # check if it is in the candidate n-gram

                if (
                    reference_n_gram[j] > candidate_n_gram[j]
                ):  # if the count of the reference n-gram is bigger
                    # than the corresponding count in the candidate n-gram
                    reference_n_gram[j] = candidate_n_gram[
                        j
                    ]  # then set the count of the reference n-gram to be equal
                    # to the count of the candidate n-gram
            else:

                reference_n_gram[j] = 0  # else reference n-gram = 0

        clipped_precision_score.append(sum(reference_n_gram.values()) / c)

    weights = [0.25] * 4

    s = (w_i * np.log(p_i) for w_i, p_i in zip(weights, clipped_precision_score))
    s = np.exp(np.sum(s))
    return s

# ### STEP 3: Computing the BLEU score

# In[8]:


def bleu_score(reference, candidate):
    BP = brevity_penalty(reference, candidate)
    precision = clipped_precision(reference, candidate)
    return BP * precision

# ### STEP 4: Testing with our Example Reference and Candidates Sentences

# In[9]:


print(
    "Results reference versus candidate 1 our own code BLEU: ",
    round(bleu_score(tokenized_ref, tokenized_cand_1) * 100, 1),
)
print(
    "Results reference versus candidate 2 our own code BLEU: ",
    round(bleu_score(tokenized_ref, tokenized_cand_2) * 100, 1),
)

# ### STEP 5: Comparing the Results from our Code with the SacreBLEU Library

# In[ ]:


print(
    "Results reference versus candidate 1 sacrebleu library sentence BLEU: ",
    round(sacrebleu.corpus_bleu(reference, candidate_1).score, 1),
)
print(
    "Results reference versus candidate 2 sacrebleu library sentence BLEU: ",
    round(sacrebleu.corpus_bleu(reference, candidate_2).score, 1),
)

# # Part 2:  BLEU computation on a corpus

# ## Loading Data Sets for Evaluation Using the BLEU Score

# In this section, we will show a simple pipeline for evaluating machine translated text. Due to storage and speed constraints, we will not be using our own model in this lab (you'll get to do that in the assignment!). Instead, we will be using [Google Translate](https://translate.google.com) to generate English to German translations and we will evaluate it against a known evaluation set. There are three files we will need:
# 
# 1. A source text in English. In this lab, we will use the first 1671 words of the [wmt19](http://statmt.org/wmt19/translation-task.html) evaluation dataset downloaded via SacreBLEU. We just grabbed a subset because of limitations in the number of words that can be translated using Google Translate. 
# 2. A reference translation to German of the corresponding first 1671 words from the original English text. This is also provided by SacreBLEU.
# 3. A candidate machine translation to German from the same 1671 words. This is generated by feeding the source text to a machine translation model. As mentioned above, we will use Google Translate to generate the translations in this file.
# 
# With that, we can now compare the reference an candidate translation to get the BLEU Score.

# In[ ]:


# Loading the raw data
wmt19news_src = open("wmt19_src.txt", "rU")
wmt19news_src_1 = wmt19news_src.read()
wmt19news_src.close()
wmt19news_ref = open("wmt19_ref.txt", "rU")
wmt19news_ref_1 = wmt19news_ref.read()
wmt19news_ref.close()
wmt19news_can = open("wmt19_can.txt", "rU")
wmt19news_can_1 = wmt19news_can.read()
wmt19news_can.close()

# Tokenizing the raw data
tokenized_corpus_src = nltk.word_tokenize(wmt19news_src_1.lower())
tokenized_corpus_ref = nltk.word_tokenize(wmt19news_ref_1.lower())
tokenized_corpus_cand = nltk.word_tokenize(wmt19news_can_1.lower())

# Inspecting the first sentence of the data.

# In[ ]:


print("English source text:")
print("\n")
print(f"{wmt19news_src_1[0:170]} -> {tokenized_corpus_src[0:30]}")
print("\n")
print("German reference translation:")
print("\n")
print(f"{wmt19news_ref_1[0:219]} -> {tokenized_corpus_ref[0:35]}")
print("\n")
print("German machine translation:")
print("\n")
print(f"{wmt19news_can_1[0:199]} -> {tokenized_corpus_cand[0:29]}")

# In[ ]:


print(
    "Results reference versus candidate 1 our own BLEU implementation: ",
    round(bleu_score(tokenized_corpus_ref, tokenized_corpus_cand) * 100, 1),
)

# In[ ]:


print(
    "Results reference versus candidate 1 sacrebleu library BLEU: ",
    round(sacrebleu.corpus_bleu(wmt19news_ref_1, wmt19news_can_1).score, 1),
)

# **BLEU Score Interpretation on a Corpus**
# 
# |Score      | Interpretation                                                |
# |:---------:|:-------------------------------------------------------------:|
# | < 10      | Almost useless                                                |
# | 10 - 19   | Hard to get the gist                                          |
# | 20 - 29   | The gist is clear, but has significant grammatical errors     |
# | 30 - 40   | Understandable to good translations                           |
# | 40 - 50   | High quality translations                                     |
# | 50 - 60   | Very high quality, adequate, and fluent translations          |
# | > 60      | Quality often better than human                               |

# From the table above (taken [here](https://cloud.google.com/translate/automl/docs/evaluate)), we can see the gist of the translation is clear but has significant grammatical errors. Nonetheless, the results of our coded BLEU score are almost identical to those of the SacreBLEU package.