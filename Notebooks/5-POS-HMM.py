# %% Import libraries | Start of the program

import pandas as pd
from collections import defaultdict
from collections import Counter
import math
import numpy as np
import string

# %% Create Vocab from the training data and save it to hmm_vocab.txt

with open("../data/WSJ_02-21.pos", 'r') as f:
    lines = f.readlines()

print('First 5 lines read from the \'WSJ_02-21.pos\' file:\n', lines[:5], end='')

# %% Creating list of words, frequencyDictionary and vocabulary

# Splitting the lines read by '\t'
words = [line.split('\t')[0] for line in lines]

# Using a counter to derive a frequency dictionary
frequencyDictionary = Counter(words)

# Creating a vocabulary list using the dictionary we just derived
# Only those values of count that are greater than 2 and aren't a new line character, are considered
vocab = [key for key, value in frequencyDictionary.items() if (value > 2 and key != '\n')]
vocab.sort()

# %% Printing out the first 5 elements in our frequency dictionary

# Enumerating over the dictionary to get an index
print('Printing the first 5 elements in our sorted dictionary of frequencies:')
for i, item in enumerate(frequencyDictionary.items()):
    print(item)
    
    if i > 5:
        break

# %% Using the set of punctuation marks from the string library

print(f'Here are the set of punctuation marks in the string library:\n{set(string.punctuation)}\nLength: {len(set(string.punctuation))}')

# %%
def assign_unk(word):

    punct = set(string.punctuation)
    
    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    if any(char.isdigit() for char in word):
        return "--unk_digit--"
    
    elif any(char in punct for char in word):
        return "--unk_punct--"
    
    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    
    return "--unk--"

# %%
print('\n'.split())

# %%
def get_word_tag(line, vocab):
    # If line is empty return placeholders for word and tag
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        word, tag = line.split()
#         Handling unknown words 
        if word not in vocab: 
            word = assign_unk(word)
    return word, tag

# %%
get_word_tag('\n', vocab)

# %%
get_word_tag('In\tIN\n', vocab)

# %%
get_word_tag('tardigrade\tNN\n', vocab)

# %%
def preprocess(vocab, data_fp):
    orig = []
    prep = []

    with open(data_fp, "r") as data_file:

        for cnt, word in enumerate(data_file):
#             cnt=0, word='The'
#             word= '\n'
            if not word.split():
                orig.append(word.strip())
                word = "--n--"
                prep.append(word)
                continue

            # Handle unknown words
# 
            elif word.strip() not in vocab:
                orig.append(word.strip())
                word = assign_unk(word)
                prep.append(word)
                continue

            else:
 #             word='The'
                orig.append(word.strip())
                prep.append(word.strip())

    assert(len(orig) == len(open(data_fp, "r").readlines()))
    assert(len(prep) == len(open(data_fp, "r").readlines()))

    return orig, prep

# %% [markdown]
# <a name='0'></a>
# ## Part 0: Data Sources
# use two tagged data sets collected from the **Wall Street Journal (WSJ)**. 
# 
# [Here](http://relearn.be/2015/training-common-sense/sources/software/pattern-2.6-critical-fork/docs/html/mbsp-tags.html) is an example 'tag-set' or Part of Speech designation describing the two or three letter tag and their meaning. 
# - One data set (**WSJ-2_21.pos**) will be used for **training**.
# - The other (**WSJ-24.pos**) for **testing**. 
# - The tagged training data will be  preprocessed to form a vocabulary (**hmm_vocab.txt**). 
# - The words in the vocabulary are words from the training set that were used two or more times. 
# - The vocabulary is augmented with a set of 'unknown word tokens'
# 
# The training set will be used to create the emission, transmission and tag counts. 
# 
# The test set (WSJ-24.pos) is read in to create `y`. 
# - This contains both the test text and the true tag. 
# - The test set has also been preprocessed to remove the tags to form **test_words.txt**. 
# - This is read in and further processed to identify the end of sentences and handle words not in the vocabulary using functions provided in **utils_pos.py**. 
# - This forms the list `prep`, the preprocessed text used to test our  POS taggers.
# 
# A POS tagger will necessarily encounter words that are not in its datasets. 
# - To improve accuracy, these words are further analyzed during preprocessing to extract available hints as to their appropriate tag. 
# - For example, the suffix 'ize' is a hint that the word is a verb, as in 'final-ize' or 'character-ize'. 
# - A set of unknown-tokens, such as '--unk-verb--' or '--unk-noun--' will replace the unknown words in both the training and test corpus and will appear in the emission, transmission and tag data structures.
# 

# %%
# load training data
with open("../data/WSJ_02-21.pos", 'r') as f:
    training_corpus = f.readlines()
# load vocab
with open("../data/vocab.txt", 'r') as f:
    voc_l = f.read().split('\n')

# %%
voc_l[:10]

# %%
# create vocab: dictionary that has the index of the corresponding words
vocab = {} 
for i, word in enumerate(sorted(voc_l)): 
    vocab[word] = i       

# %%
len(vocab)

# %%
print("Vocabulary dictionary, key is the word, value is a unique integer")
cnt = 0
for k,v in vocab.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break

# %%
# load in the test corpus
with open("../data/WSJ_24.pos", 'r') as f:
    y = f.readlines()
print(y[0:10])

# %%
#corpus without tags, preprocessed
ori, prep = preprocess(vocab, "../data/test.words")
print(prep[600:800])

# %%
print(ori[600:800])

# %% [markdown]
# ## Part 1- Train a model without HMM
# Training
# You will start with the simplest possible parts-of-speech tagger and we will build up to the state of the art. 
# 
# In this section, you will find the words that are not ambiguous. 
# - For example, the word `is` is a verb and it is not ambiguous. 
# - In the `WSJ` corpus, $86$% of the token are unambiguous (meaning they have only one tag) 
# - About $14\%$ are ambiguous (meaning that they have more than one tag)
# 
# Before you start predicting the tags of each word, we will need to compute a few dictionaries that will help you to generate the tables. 

# %% [markdown]
# #### Transition counts
# - The first dictionary is the `transition_counts` dictionary which computes the number of times each tag happened next to another tag. 
# 
# This dictionary will be used to compute: 
# $$P(t_i |t_{i-1}) \tag{1}$$
# 
# This is the probability of a tag at position $i$ given the tag at position $i-1$.
# 
# In order for you to compute equation 1, you will create a `transition_counts` dictionary where 
# - The keys are `(prev_tag, tag)`
# - The values are the number of times those two tags appeared in that order. 

# %% [markdown]
# #### Emission counts
# 
# The second dictionary you will compute is the `emission_counts` dictionary. This dictionary will be used to compute:
# 
# $$P(w_i|t_i)\tag{2}$$
# 
# In other words, you will use it to compute the probability of a word given its tag. 
# 
# In order for you to compute equation 2, you will create an `emission_counts` dictionary where 
# - The keys are `(tag, word)` 
# - The values are the number of times that pair showed up in your training set. 

# %% [markdown]
# #### Tag counts
# 
# The last dictionary you will compute is the `tag_counts` dictionary. 
# - The key is the tag 
# - The value is the number of times each tag appeared.

# %%
def create_dictionaries(training_corpus, vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    
    # Initialize "prev_tag" (previous tag) with the start state, denoted by '--s--'
    prev_tag = '--s--' 
    i = 0 
    
    # Each item in the training corpus contains a word and its POS tag
    # Go through each word and its tag in the training corpus
    for word_tag in training_corpus:
        
        # Increment the word_tag count
        i += 1
    
        # get the word and tag using the get_word_tag helper function 
        word, tag = get_word_tag(word_tag,vocab) 
        
        # Increment the transition count for the previous word and tag
        transition_counts[(prev_tag, tag)] += 1
        
        # Increment the emission count for the tag and word
        emission_counts[(tag, word)] += 1

        # Increment the tag count
        tag_counts[tag] += 1

        # Set the previous tag to this tag (for the next iteration of the loop)
        prev_tag = tag
    
        
    return emission_counts, transition_counts, tag_counts

# %%
emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)

# %%
print("transition_counts\n")
cnt = 0
for k,v in transition_counts.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break

# %%
print("emission_counts\n")
cnt = 0
for k,v in emission_counts.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break

# %%
print("tag_counts\n")
cnt = 0
for k,v in tag_counts.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break

# %%
# get all the POS states
states = sorted(tag_counts.keys())
print(len(states))
print(states)

# %%
print("transition examples: ")
for ex in list(transition_counts.items())[:3]:
    print(ex)
print()

print("emission examples: ")
for ex in list(emission_counts.items())[200:203]:
    print (ex)
print()

print("ambiguous word example: ")
for tup,cnt in emission_counts.items():
    if tup[1] == 'back': print (tup, cnt) 

# %% [markdown]
# ### Testing
# 
# Test the accuracy of your parts-of-speech tagger using your `emission_counts` dictionary. 
# - Given your preprocessed test corpus `prep`, you will assign a parts-of-speech tag to every word in that corpus. 
# - Using the original tagged test corpus `y`, you will then compute what percent of the tags you got correct. 

# %%
# print the test set actual data
print(y[:10])

# %%
# The test set is preprocessed to get only words not tag
print(prep[:10])

# %%
print(len(y), len(prep))

# %%
def predict_pos(prep, y, emission_counts, vocab, states):
    num_correct = 0
    
    # Get the (tag, word) tuples, stored as a set
    all_words = set(emission_counts.keys())
    
    # Get the number of (word, POS) tuples in the corpus 'y'
    total = len(y)
    
# pre--> 'The'
# Get the true label of the word from the test set (y)--> 'The\tDT\n'
# zip--> ('The', 'The\tDT\n')
# word= 'The' and y_tup= 'The\tDT\n'
# y_tup_l= ['The', 'DT']

# Example2- word= '--unk--'
# y='vantage\tNN\n'
#  zip(prep, y)= ('--unk--', 'vantage\tNN\n')
# y_tup_l= ['vantage', 'NN']
# true_label= 'NN'
# word= '--unk--'
    for word, y_tup in zip(prep, y): 
        y_tup_l = y_tup.split()
        if len(y_tup_l) == 2:
            true_label = y_tup_l[1]
        else:
            # If the y_tup didn't contain word and POS, go to next word
            continue
    
       
        
# Example--> test set word = 'The'
# check for all possible pos from states, if that (pos, word) exists in emmision count (from training 
# set)and save the max count in the count_final and get its tag, now compare this tag with the 
# true_label to calculate the accuracy

#Example2--> if '--unk--' in vocab (yes)
# states= ['#', '$', "''", '(', ')', ',', '--s--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
# key1= ('#', --unk--')
# key2= ('$', '--unk--')
# ..
# ..
        count_final = 0
        pos_final = ''
        if word in vocab:
            for pos in states:
                key = (pos,word)
                if key in emission_counts: 
                    count = emission_counts[key]

                    if count>count_final:
                        count_final = count
                        pos_final = pos

            if pos_final == true_label:
                num_correct += 1
    
    accuracy = num_correct / total
    return accuracy

# %%
accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)
print(f"Accuracy of prediction using predict_pos is {accuracy_predict_pos:.4f}")

# %% [markdown]
# ##### Expected Output
# ```py
# Accuracy of prediction using predict_pos is 0.8889
# ```
# 
# 88.9% is really good for this warm up exercise. With hidden markov models, you should be able to get **95% accuracy.**

# %% [markdown]
# <a name='2'></a>
# # Part 2: Hidden Markov Models for POS
# 
# Implementing a Hidden Markov Model (HMM) with a Viterbi decoder
# - The HMM is one of the most commonly used algorithms in Natural Language Processing, and is a foundation to many deep learning techniques you will see in this specialization. 
# - In addition to parts-of-speech tagging, HMM is used in speech recognition, speech synthesis, etc. 
# - By completing this part of the assignment you will get a 95% accuracy on the same dataset you used in Part 1.
# 
# The Markov Model contains a number of states and the probability of transition between those states. 
# - In this case, the states are the parts-of-speech. 
# - A Markov Model utilizes a transition matrix, `A`. 
# - A Hidden Markov Model adds an observation or emission matrix `B` which describes the probability of a visible observation when we are in a particular state. 
# - In this case, the emissions are the words in the corpus
# - The state, which is hidden, is the POS tag of that word.

# %% [markdown]
# <a name='2.1'></a>
# ## Part 2.1 Generating Matrices
# 
# ### Creating the 'A' transition probabilities matrix
# Now that you have your `emission_counts`, `transition_counts`, and `tag_counts`, you will start implementing the Hidden Markov Model. 
# 
# This will allow you to quickly construct the 
# - `A` transition probabilities matrix.
# - and the `B` emission probabilities matrix. 
# 
# You will also use some smoothing when computing these matrices. 
# 
# Here is an example of what the `A` transition matrix would look like (it is simplified to 5 tags for viewing. It is 46x46 in this assignment.):
# 
# 
# |**A**  |...|         RBS  |          RP  |         SYM  |      TO  |          UH|...
# | --- ||---:-------------| ------------ | ------------ | -------- | ---------- |----
# |**RBS**  |...|2.217069e-06  |2.217069e-06  |2.217069e-06  |0.008870  |2.217069e-06|...
# |**RP**   |...|3.756509e-07  |7.516775e-04  |3.756509e-07  |0.051089  |3.756509e-07|...
# |**SYM**  |...|1.722772e-05  |1.722772e-05  |1.722772e-05  |0.000017  |1.722772e-05|...
# |**TO**   |...|4.477336e-05  |4.472863e-08  |4.472863e-08  |0.000090  |4.477336e-05|...
# |**UH**  |...|1.030439e-05  |1.030439e-05  |1.030439e-05  |0.061837  |3.092348e-02|...
# | ... |...| ...          | ...          | ...          | ...      | ...        | ...
# 
# Note that the matrix above was computed with smoothing. 
# 
# Each cell gives you the probability to go from one part of speech to another. 
# - In other words, there is a 4.47e-8 chance of going from parts-of-speech `TO` to `RP`. 
# - The sum of each row has to equal 1, because we assume that the next POS tag must be one of the available columns in the table.
# 
# The smoothing was done as follows: 
# 
# $$ P(t_i | t_{i-1}) = \frac{C(t_{i-1}, t_{i}) + \alpha }{C(t_{i-1}) +\alpha * N}\tag{3}$$
# 
# - $N$ is the total number of tags
# - $C(t_{i-1}, t_{i})$ is the count of the tuple (previous POS, current POS) in `transition_counts` dictionary.
# - $C(t_{i-1})$ is the count of the previous POS in the `tag_counts` dictionary.
# - $\alpha$ is a smoothing parameter.

# %% [markdown]
# <a name='ex-03'></a>
# ### Exercise 03
# 
# **Instructions:** Implement the `create_transition_matrix` below for all tags. Your task is to output a matrix that computes equation 3 for each cell in matrix `A`. 

# %%
print("tag_counts\n")
cnt = 0
for k,v in tag_counts.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break

# %%
all_tags = sorted(tag_counts.keys()) 
print(all_tags)
# print(len(all_tags))
# count_prev_tag = tag_counts[all_tags[i]]
count_prev_tag = tag_counts['#']
print(count_prev_tag)
print(transition_counts[('#', '#')])
print(transition_counts[('#', '$')])

# %%
print("transition count dictionary values ")
for ex in list(transition_counts.items())[:3]:
    print(ex)
print()
trans_keys = set(transition_counts.keys())
cnt = 0

# %%
def create_transition_matrix(alpha, tag_counts, transition_counts):
 
    all_tags = sorted(tag_counts.keys()) 
    num_tags = len(all_tags)
    A = np.zeros((num_tags,num_tags))
    
    trans_keys = set(transition_counts.keys())
    
    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i],all_tags[j])
            if transition_counts: 
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]
            A[i,j] = (count + alpha) / (count_prev_tag + alpha*num_tags)

    return A

# %%
alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
# Testing your function
print(f"A at row 0, col 0: {A[0,0]:.9f}")
print(f"A at row 3, col 1: {A[3,1]:.4f}")

print("View a subset of transition matrix A")
A_sub = pd.DataFrame(A[30:35,30:35], index=states[30:35], columns = states[30:35] )
print(A_sub)

# %% [markdown]
# ##### Expected Output
# ```CPP
# A at row 0, col 0: 0.000007040
# A at row 3, col 1: 0.1691
# View a subset of transition matrix A
#               RBS            RP           SYM        TO            UH
# RBS  2.217069e-06  2.217069e-06  2.217069e-06  0.008870  2.217069e-06
# RP   3.756509e-07  7.516775e-04  3.756509e-07  0.051089  3.756509e-07
# SYM  1.722772e-05  1.722772e-05  1.722772e-05  0.000017  1.722772e-05
# TO   4.477336e-05  4.472863e-08  4.472863e-08  0.000090  4.477336e-05
# UH   1.030439e-05  1.030439e-05  1.030439e-05  0.061837  3.092348e-02
# ```

# %% [markdown]
# ### Create the 'B' emission probabilities matrix
# 
# Now you will create the `B` transition matrix which computes the emission probability. 
# 
# You will use smoothing as defined below: 
# 
# $$P(w_i | t_i) = \frac{C(t_i, word_i)+ \alpha}{C(t_{i}) +\alpha * N}\tag{4}$$
# 
# - $C(t_i, word_i)$ is the number of times $word_i$ was associated with $tag_i$ in the training data (stored in `emission_counts` dictionary).
# - $C(t_i)$ is the number of times $tag_i$ was in the training data (stored in `tag_counts` dictionary).
# - $N$ is the number of words in the vocabulary
# - $\alpha$ is a smoothing parameter. 
# 
# The matrix `B` is of dimension (num_tags, N), where num_tags is the number of possible parts-of-speech tags. 
# 
# Here is an example of the matrix, only a subset of tags and words are shown: 
# <p style='text-align: center;'> <b>B Emissions Probability Matrix (subset)</b>  </p>
# 
# |**B**| ...|          725 |     adroitly |    engineers |     promoted |      synergy| ...|
# |----|----|--------------|--------------|--------------|--------------|-------------|----|
# |**CD**  | ...| **8.201296e-05** | 2.732854e-08 | 2.732854e-08 | 2.732854e-08 | 2.732854e-08| ...|
# |**NN**  | ...| 7.521128e-09 | 7.521128e-09 | 7.521128e-09 | 7.521128e-09 | **2.257091e-05**| ...|
# |**NNS** | ...| 1.670013e-08 | 1.670013e-08 |**4.676203e-04** | 1.670013e-08 | 1.670013e-08| ...|
# |**VB**  | ...| 3.779036e-08 | 3.779036e-08 | 3.779036e-08 | 3.779036e-08 | 3.779036e-08| ...|
# |**RB**  | ...| 3.226454e-08 | **6.456135e-05** | 3.226454e-08 | 3.226454e-08 | 3.226454e-08| ...|
# |**RP**  | ...| 3.723317e-07 | 3.723317e-07 | 3.723317e-07 | **3.723317e-07** | 3.723317e-07| ...|
# | ...    | ...|     ...      |     ...      |     ...      |     ...      |     ...      | ...|
# 
# 

# %% [markdown]
# <a name='ex-04'></a>
# ### Exercise 04
# **Instructions:** Implement the `create_emission_matrix` below that computes the `B` emission probabilities matrix. Your function takes in $\alpha$, the smoothing parameter, `tag_counts`, which is a dictionary mapping each tag to its respective count, the `emission_counts` dictionary where the keys are (tag, word) and the values are the counts. Your task is to output a matrix that computes equation 4 for each cell in matrix `B`. 

# %%
print("emission examples: ")
for ex in list(emission_counts.items())[200:203]:
    print (ex)
print()

# %%
num_words = len(vocab)
print(num_words)
# print(type(vocab))
# print(vocab)
print("vocab is a dictionary where key is the word, value is a unique integer")
cnt = 0
for k,v in vocab.items():
    print(f"{k}:{v}")
    cnt += 1
    if cnt > 5:
        break
print(list(vocab)[:5])

# %%
def create_transition_matrix(alpha, tag_counts, transition_counts):
 
    num_tags = len(all_tags)
    all_tags = sorted(tag_counts.keys()) 
    A = np.zeros((num_tags,num_tags))
    
#     trans_keys = set(transition_counts.keys())
    
    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i],all_tags[j])
            if key in transition_counts.keys(): 
                count = transition_counts[key]
            count_prev_tag = tag_counts[all_tags[i]]
            
            A[i,j] = (count + alpha) / (count_prev_tag + alpha*num_tags)
    return A

# %%
def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    
    num_tags = len(tag_counts)
    all_tags = sorted(tag_counts.keys())
    num_words = len(vocab)
    B = np.zeros((num_tags, num_words))

#     emis_keys = set(list(emission_counts.keys()))
    
    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key =  (all_tags[i],vocab[j])
            if key in emission_counts.keys():
                count = emission_counts[key]
            count_tag = tag_counts[all_tags[i]]
            
            B[i,j] = (count + alpha) / (count_tag+ alpha*num_words)
    return B

# %%
# creating your emission probability matrix. this takes a few minutes to run. 
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

print(f"View Matrix position at row 0, column 0: {B[0,0]:.9f}")
print(f"View Matrix position at row 3, column 1: {B[3,1]:.9f}")

# Try viewing emissions for a few words in a sample dataframe
cidx  = ['725','adroitly','engineers', 'promoted', 'synergy']

# Get the integer ID for each word
cols = [vocab[a] for a in cidx]

# Choose POS tags to show in a sample dataframe
rvals =['CD','NN','NNS', 'VB','RB','RP']

# For each POS tag, get the row number from the 'states' list
rows = [states.index(a) for a in rvals]

# Get the emissions for the sample of words, and the sample of POS tags
B_sub = pd.DataFrame(B[np.ix_(rows,cols)], index=rvals, columns = cidx )
print(B_sub)

# %% [markdown]
# ##### Expected Output
# 
# ```CPP
# View Matrix position at row 0, column 0: 0.000006032
# View Matrix position at row 3, column 1: 0.000000720
#               725      adroitly     engineers      promoted       synergy
# CD   8.201296e-05  2.732854e-08  2.732854e-08  2.732854e-08  2.732854e-08
# NN   7.521128e-09  7.521128e-09  7.521128e-09  7.521128e-09  2.257091e-05
# NNS  1.670013e-08  1.670013e-08  4.676203e-04  1.670013e-08  1.670013e-08
# VB   3.779036e-08  3.779036e-08  3.779036e-08  3.779036e-08  3.779036e-08
# RB   3.226454e-08  6.456135e-05  3.226454e-08  3.226454e-08  3.226454e-08
# RP   3.723317e-07  3.723317e-07  3.723317e-07  3.723317e-07  3.723317e-07
# ```

# %% [markdown]
# <a name='3'></a>
# # Part 3: Viterbi Algorithm and Dynamic Programming
# 
# In this part of the assignment you will implement the Viterbi algorithm which makes use of dynamic programming. Specifically, you will use your two matrices, `A` and `B` to compute the Viterbi algorithm. We have decomposed this process into three main steps for you. 
# 
# * **Initialization** - In this part you initialize the `best_paths` and `best_probabilities` matrices that you will be populating in `feed_forward`.
# * **Feed forward** - At each step, you calculate the probability of each path happening and the best paths up to that point. 
# * **Feed backward**: This allows you to find the best path with the highest probabilities. 
# 
# <a name='3.1'></a>
# ## Part 3.1:  Initialization 
# 
# You will start by initializing two matrices of the same dimension. 
# 
# - best_probs: Each cell contains the probability of going from one POS tag to a word in the corpus.
# 
# - best_paths: A matrix that helps you trace through the best possible path in the corpus. 

# %%
prep[0]

# %% [markdown]
# <a name='ex-05'></a>
# ### Exercise 05
# **Instructions**: 
# Write a program below that initializes the `best_probs` and the `best_paths` matrix. 
# 
# Both matrices will be initialized to zero except for column zero of `best_probs`.  
# - Column zero of `best_probs` is initialized with the assumption that the first word of the corpus was preceded by a start token ("--s--"). 
# - This allows you to reference the **A** matrix for the transition probability
# 
# Here is how to initialize column 0 of `best_probs`:
# - The probability of the best path going from the start index to a given POS tag indexed by integer $i$ is denoted by $\textrm{best_probs}[s_{idx}, i]$.
# - This is estimated as the probability that the start tag transitions to the POS denoted by index $i$: $\mathbf{A}[s_{idx}, i]$ AND that the POS tag denoted by $i$ emits the first word of the given corpus, which is $\mathbf{B}[i, vocab[corpus[0]]]$.
# - Note that vocab[corpus[0]] refers to the first word of the corpus (the word at position 0 of the corpus). 
# - **vocab** is a dictionary that returns the unique integer that refers to that particular word.
# 
# Conceptually, it looks like this:
# $\textrm{best_probs}[s_{idx}, i] = \mathbf{A}[s_{idx}, i] \times \mathbf{B}[i, corpus[0] ]$
# 
# 
# In order to avoid multiplying and storing small values on the computer, we'll take the log of the product, which becomes the sum of two logs:
# 
# $best\_probs[i,0] = log(A[s_{idx}, i]) + log(B[i, vocab[corpus[0]]$
# 
# Also, to avoid taking the log of 0 (which is defined as negative infinity), the code itself will just set $best\_probs[i,0] = float('-inf')$ when $A[s_{idx}, i] == 0$
# 
# 
# So the implementation to initialize $best\_probs$ looks like this:
# 
# $ if A[s_{idx}, i] <> 0 : best\_probs[i,0] = log(A[s_{idx}, i]) + log(B[i, vocab[corpus[0]]])$
# 
# $ if A[s_{idx}, i] == 0 : best\_probs[i,0] = float('-inf')$
# 
# Please use [math.log](https://docs.python.org/3/library/math.html) to compute the natural logarithm.

# %%
len(prep)

# %%
print(prep[:10])
print(prep[0])
vocab['The']
s_idx = states.index("--s--")
print(states)
print(s_idx)

# %%
print(math.log(A[6,0])+math.log(B[0,8614]))
print(math.log(A[6,11])+math.log(B[11,8614]))

# %%
# A['--s--', '#'], A['--s--', '$'], A['--s--', ''], A['--s--', '(']..., A['--s--', 'NNP']]

# %%
print(vocab['The'])
print(B[0,8614])
print(B[11,8614])

# %%
states, len(states)

# %%
def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Input: 
        states: a list of all possible parts-of-speech
        tag_counts: a dictionary mapping each tag to its respective count
        A: Transition Matrix of dimension (num_tags, num_tags)
        B: Emission Matrix of dimension (num_tags, len(vocab))
        corpus: a sequence of words whose POS is to be identified in a list 
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''
   
    num_tags = len(tag_counts)
    
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    
    # Index for the <start> tag
    s_idx = states.index("--s--")
    
    # Checking the probability of a tag given the <start> tag occurring first 
    for i in range(num_tags):
        
        if A[s_idx,i] == 0: 
            
            # Initialize best_probs at POS tag 'i', column 0, to negative infinity
            best_probs[i,0] = float('-inf')
        
        # For all other cases when transition from start token to POS tag i is non-zero:
        else:
            # Initialize best_probs at POS tag 'i', column 0
            # Check the formula in the instructions above
            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])
                        

    return best_probs, best_paths

# %%
best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)

# %%
best_probs.shape

# %%
# print first column of the best_prob matrix 
for i in range(len(states)):
    print(f"best_probs[{i},0]: {best_probs[i,0]:.4f}") 

# %% [markdown]
# ##### Expected Output
# 
# ```CPP
# best_probs[0,0]: -22.6098
# best_paths[2,3]: 0.0000
# ```
# 

# %% [markdown]
# <a name='3.2'></a>
# ## Part 3.2 Viterbi Forward
# 
# Implement the `viterbi_forward` segment. 
# - Walk forward through the corpus.
# - For each word, compute a probability for each possible tag. 
# - Unlike the previous algorithm `predict_pos` (the 'warm-up' exercise), this will include the path up to that (word,tag) combination. 
# 
# Here is an example with a three-word corpus "Loss tracks upward":
# - Note, in this example, only a subset of states (POS tags) are shown in the diagram below, for easier reading. 
# - In the diagram below, the first word "Loss" is already initialized. 
# - The algorithm will compute a probability for each of the potential tags in the second and future words. 
# 
# Compute the probability that the tag of the second work ('tracks') is a verb, 3rd person singular present (VBZ).  
# - In the `best_probs` matrix, go to the column of the second word ('tracks'), and row 40 (VBZ), this cell is highlighted in light orange in the diagram below.
# - Examine each of the paths from the tags of the first word ('Loss') and choose the most likely path.  
# - An example of the calculation for **one** of those paths is the path from ('Loss', NN) to ('tracks', VBZ).
# - The log of the probability of the path up to and including the first word 'Loss' having POS tag NN is $-14.32$.  The `best_probs` matrix contains this value -14.32 in the column for 'Loss' and row for 'NN'.
# - Find the probability that NN transitions to VBZ.  To find this probability, go to the `A` transition matrix, and go to the row for 'NN' and the column for 'VBZ'.  The value is $4.37e-02$, which is circled in the diagram, so add $-14.32 + log(4.37e-02)$. 
# - Find the log of the probability that the tag VBS would 'emit' the word 'tracks'.  To find this, look at the 'B' emission matrix in row 'VBZ' and the column for the word 'tracks'.  The value $4.61e-04$ is circled in the diagram below.  So add $-14.32 + log(4.37e-02) + log(4.61e-04)$.
# - The sum of $-14.32 + log(4.37e-02) + log(4.61e-04)$ is $-25.13$. Store $-25.13$ in the `best_probs` matrix at row 'VBZ' and column 'tracks' (as seen in the cell that is highlighted in light orange in the diagram).
# - All other paths in best_probs are calculated.  Notice that $-25.13$ is greater than all of the other values in column 'tracks' of matrix `best_probs`, and so the most likely path to 'VBZ' is from 'NN'.  'NN' is in row 20 of the `best_probs` matrix, so $20$ is the most likely path.
# - Store the most likely path $20$ in the `best_paths` table.  This is highlighted in light orange in the diagram below.

# %% [markdown]
# The formula to compute the probability and path for the $i^{th}$ word in the $corpus$, the prior word $i-1$ in the corpus, current POS tag $j$, and previous POS tag $k$ is:
# 
# $\mathrm{prob} = \mathbf{best\_prob}_{k, i-1} + \mathrm{log}(\mathbf{A}_{k, j}) + \mathrm{log}(\mathbf{B}_{j, vocab(corpus_{i})})$
# 
# where $corpus_{i}$ is the word in the corpus at index $i$, and $vocab$ is the dictionary that gets the unique integer that represents a given word.
# 
# $\mathrm{path} = k$
# 
# where $k$ is the integer representing the previous POS tag.
# 

# %% [markdown]
# <a name='ex-06'></a>
# 
# ### Exercise 06
# 
# Instructions: Implement the `viterbi_forward` algorithm and store the best_path and best_prob for every possible tag for each word in the matrices `best_probs` and `best_tags` using the pseudo code below.
# 
# `for each word in the corpus
# 
#     for each POS tag type that this word may be
#     
#         for POS tag type that the previous word could be
#         
#             compute the probability that the previous word had a given POS tag, that the current word has a given POS tag, and that the POS tag would emit this current word.
#             
#             retain the highest probability computed for the current word
#             
#             set best_probs to this highest probability
#             
#             set best_paths to the index 'k', representing the POS tag of the previous word which produced the highest probability `
# 
# Please use [math.log](https://docs.python.org/3/library/math.html) to compute the natural logarithm.

# %% [markdown]
# <img src = "Forward4.PNG"/>

# %% [markdown]
# <details>    
# <summary>
#     <font size="3" color="darkgreen"><b>Hints</b></font>
# </summary>
# <p>
# <ul>
#     <li>Remember that when accessing emission matrix B, the column index is the unique integer ID associated with the word.  It can be accessed by using the 'vocab' dictionary, where the key is the word, and the value is the unique integer ID for that word.</li>
# </ul>
# </p>
# 

# %%
len(prep)
print(prep[:10])
print(states)
print(states.index('NN'))

# %%
# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: viterbi_forward
def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    '''
    Input: 
        A, B: The transiton and emission matrices respectively
        test_corpus: a list containing a preprocessed corpus
        best_probs: an initilized matrix of dimension (num_tags, len(corpus))
        best_paths: an initilized matrix of dimension (num_tags, len(corpus))
        vocab: a dictionary where keys are words in vocabulary and value is an index 
    Output: 
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''
    # Get the number of unique POS tags (which is the num of rows in best_probs)
    num_tags = best_probs.shape[0]
 
    for i in range(1, len(test_corpus)): 
        for j in range(num_tags):
            
            best_prob_i = float("-inf")
            best_path_i = None

            for k in range(num_tags):
                prob = best_probs[k,i-1]+math.log(A[k,j]) +math.log(B[j,vocab[test_corpus[i]]])
                
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
                    
            best_probs[j,i] = best_prob_i
            best_paths[j,i] = best_path_i

    return best_probs, best_paths

# %% [markdown]
# Run the `viterbi_forward` function to fill in the `best_probs` and `best_paths` matrices.
# 
# **Note** that this will take a few minutes to run.  There are about 30,000 words to process.

# %%
# this will take a few minutes to run => processes ~ 30,000 words
best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)

# %%
prep[1]

# %%
print(states)

# %%
# Test this function --> check for the word 'economy'
for i in range(len(states)):
    print(f"best_probs[{i},1]: {best_probs[i,1]:.4f}") 

# %% [markdown]
# ##### Expected Output
# 
# ```CPP
# best_probs[0,1]: -24.7822
# best_probs[0,4]: -49.5601
# ```

# %% [markdown]
# <a name='3.3'></a>
# ## Part 3.3 Viterbi backward
# 
# Now you will implement the Viterbi backward algorithm.
# - The Viterbi backward algorithm gets the predictions of the POS tags for each word in the corpus using the `best_paths` and the `best_probs` matrices.
# 
# The example below shows how to walk backwards through the best_paths matrix to get the POS tags of each word in the corpus. Recall that this example corpus has three words: "Loss tracks upward".
# 
# POS tag for 'upward' is `RB`
# - Select the the most likely POS tag for the last word in the corpus, 'upward' in the `best_prob` table.
# - Look for the row in the column for 'upward' that has the largest probability.
# - Notice that in row 28 of `best_probs`, the estimated probability is -34.99, which is larger than the other values in the column.  So the most likely POS tag for 'upward' is `RB` an adverb, at row 28 of `best_prob`. 
# - The variable `z` is an array that stores the unique integer ID of the predicted POS tags for each word in the corpus.  In array z, at position 2, store the value 28 to indicate that the word 'upward' (at index 2 in the corpus), most likely has the POS tag associated with unique ID 28 (which is `RB`).
# - The variable `pred` contains the POS tags in string form.  So `pred` at index 2 stores the string `RB`.
# 
# 
# POS tag for 'tracks' is `VBZ`
# - The next step is to go backward one word in the corpus ('tracks').  Since the most likely POS tag for 'upward' is `RB`, which is uniquely identified by integer ID 28, go to the `best_paths` matrix in column 2, row 28.  The value stored in `best_paths`, column 2, row 28 indicates the unique ID of the POS tag of the previous word.  In this case, the value stored here is 40, which is the unique ID for POS tag `VBZ` (verb, 3rd person singular present).
# - So the previous word at index 1 of the corpus ('tracks'), most likely has the POS tag with unique ID 40, which is `VBZ`.
# - In array `z`, store the value 40 at position 1, and for array `pred`, store the string `VBZ` to indicate that the word 'tracks' most likely has POS tag `VBZ`.
# 
# POS tag for 'Loss' is `NN`
# - In `best_paths` at column 1, the unique ID stored at row 40 is 20.  20 is the unique ID for POS tag `NN`.
# - In array `z` at position 0, store 20.  In array `pred` at position 0, store `NN`.

# %% [markdown]
# <img src = "Backwards5.PNG"/>

# %% [markdown]
# <a name='ex-07'></a>
# ### Exercise 07
# Implement the `viterbi_backward` algorithm, which returns a list of predicted POS tags for each word in the corpus.
# 
# - Note that the numbering of the index positions starts at 0 and not 1. 
# - `m` is the number of words in the corpus.  
#     - So the indexing into the corpus goes from `0` to `m - 1`.
#     - Also, the columns in `best_probs` and `best_paths` are indexed from `0` to `m - 1`
# 
# 
# **In Step 1:**       
# Loop through all the rows (POS tags) in the last entry of `best_probs` and find the row (POS tag) with the maximum value.
# Convert the unique integer ID to a tag (a string representation) using the dictionary `states`.  
# 
# Referring to the three-word corpus described above:
# - `z[2] = 28`: For the word 'upward' at position 2 in the corpus, the POS tag ID is 28.  Store 28 in `z` at position 2.
# - states(28) is 'RB': The POS tag ID 28 refers to the POS tag 'RB'.
# - `pred[2] = 'RB'`: In array `pred`, store the POS tag for the word 'upward'.
# 
# **In Step 2:**  
# - Starting at the last column of best_paths, use `best_probs` to find the most likely POS tag for the last word in the corpus.
# - Then use `best_paths` to find the most likely POS tag for the previous word. 
# - Update the POS tag for each word in `z` and in `preds`.
# 
# Referring to the three-word example from above, read best_paths at column 2 and fill in z at position 1.  
# `z[1] = best_paths[z[2],2]`  
# 
# The small test following the routine prints the last few words of the corpus and their states to aid in debug.

# %%
# UNQ_C7 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: viterbi_backward
def viterbi_backward(best_probs, best_paths, corpus, states):
    '''
    This function returns the best path.
    
    '''
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1] 
    
    # Initialize array z, same length as the corpus
    z = [None] * m
    
    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]
    
    # Initialize the best probability for the last word
    best_prob_for_last_word = float('-inf')
    
    # Initialize pred array, same length as corpus
    pred = [None] * m
    
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    ## Step 1 ##
    
    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID) 
    # with highest probability for the last word
    for k in range(num_tags): # complete this line

        # If the probability of POS tag at row k 
        # is better than the previosly best probability for the last word:
        if best_probs[k,-1]>best_prob_for_last_word: # complete this line
            
            # Store the new best probability for the lsat word
            best_prob_for_last_word = best_probs[k,-1]
    
            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k
            
    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' dictionary
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[k]
    
    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(len(corpus)-1, -1, -1): # complete this line
        
        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = best_paths[np.argmax(best_probs[:,i]),i]
        
        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = best_paths[pos_tag_for_word_i,i]
        
        # Get the previous word's POS tag in string form
        # Use the 'states' dictionary, 
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i - 1] = states[pos_tag_for_word_i]
        
     ### END CODE HERE ###
    return pred

# %%
# Run and test your function
pred = viterbi_backward(best_probs, best_paths, prep, states)
m=len(pred)
print('The prediction for pred[-7:m-1] is: \n', prep[-7:m-1], "\n", pred[-7:m-1], "\n")
print('The prediction for pred[0:8] is: \n', pred[0:7], "\n", prep[0:7])

# %% [markdown]
# **Expected Output:**   
# 
# ```CPP
# The prediction for prep[-7:m-1] is:  
#  ['see', 'them', 'here', 'with', 'us', '.']  
#  ['VB', 'PRP', 'RB', 'IN', 'PRP', '.']   
# The prediction for pred[0:8] is:    
#  ['DT', 'NN', 'POS', 'NN', 'MD', 'VB', 'VBN']   
#  ['The', 'economy', "'s", 'temperature', 'will', 'be', 'taken'] 
# ```
# 
# Now you just have to compare the predicted labels to the true labels to evaluate your model on the accuracy metric!

# %% [markdown]
# <a name='4'></a>
# # Part 4: Predicting on a data set
# 
# Compute the accuracy of your prediction by comparing it with the true `y` labels. 
# - `pred` is a list of predicted POS tags corresponding to the words of the `test_corpus`. 

# %%
print('The third word is:', prep[3])
print('Your prediction is:', pred[3])
print('Your corresponding label y is: ', y[3])

# %% [markdown]
# <a name='ex-08'></a>
# ### Exercise 08
# 
# Implement a function to compute the accuracy of the viterbi algorithm's POS tag predictions.
# - To split y into the word and its tag you can use `y.split()`. 

# %%
# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: compute_accuracy
def compute_accuracy(pred, y):
    '''
    Input: 
        pred: a list of the predicted parts-of-speech 
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output: 
        
    '''
    num_correct = 0
    total = 0
    
    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # Split the label into the word and the POS tag
        word_tag_tuple = y.split()
        
        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple)!=2: # complete this line
            continue 

        # store the word and tag separately
        word, tag = word_tag_tuple
        
        # Check if the POS tag label matches the prediction
        if prediction == tag: # complete this line
            
            # count the number of times that the prediction
            # and label match
            num_correct += 1
            
        # keep track of the total number of examples (that have valid labels)
        total += 1
        
        ### END CODE HERE ###
    return num_correct/total

# %%
print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")

# %% [markdown]
# ##### Expected Output
# 
# ```CPP
# Accuracy of the Viterbi algorithm is 0.9531
# ```
# 
# Congratulations you were able to classify the parts-of-speech with 95% accuracy. 

# %% [markdown]
# ### Key Points and overview
# 
# In this assignment you learned about parts-of-speech tagging. 
# - In this assignment, you predicted POS tags by walking forward through a corpus and knowing the previous word.
# - There are other implementations that use bidirectional POS tagging.
# - Bidirectional POS tagging requires knowing the previous word and the next word in the corpus when predicting the current word's POS tag.
# - Bidirectional POS tagging would tell you more about the POS instead of just knowing the previous word. 
# - Since you have learned to implement the unidirectional approach, you have the foundation to implement other POS taggers used in industry.

# %% [markdown]
# ### References
# 
# - ["Speech and Language Processing", Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/)
# - We would like to thank Melanie Tosik for her help and inspiration


