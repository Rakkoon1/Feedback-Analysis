
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


amazing = pd.read_csv('C:/Users/marwi/OneDrive/Desktop/t_direct/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt', delimiter='\t', header=None)
bark = pd.read_csv('C:/Users/marwi/OneDrive/Desktop/t_direct/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt', delimiter='\t', header=None)

amazing.columns = ['review', 'feels']
bark.columns = ['review', 'feels']


# In[3]:


amazing.head()


# In[4]:


feato_feels = ['love', 'loved', 'loves', 'quality', 'fantastic', 'amazing',
               'incredible', 'recommend', 'impressive', 'winner', 'wonderful',
               'impressed', 'pleased', 'good', 'well', 'favorite', 'perfect', 'great',
               'worst', 'horrible', 'disappointing', 'disappointed', 'unfortunately']

for a in feato_feels:
    amazing[str(a)] = amazing.review.str.contains(str(a), case=False) 
    
for b in feato_feels:
    bark[str(b)] = bark.review.str.contains(str(b), case=False)


# In[5]:


sns.set_style('darkgrid')
plt.figure(figsize=(8,8))
sns.heatmap(amazing.corr(), square=True);


# In[6]:


amazing.head()


# In[7]:


vibe = amazing['feels']
pos = amazing[feato_feels]


# In[8]:


bern = BernoulliNB()
bern.fit(pos, vibe)
y_feels = bern.predict(pos)

print("Mislabeled points out of {}: {}".format(
    pos.shape[0], (vibe != y_feels).sum()))


# At first the mislableld points were nearly half of a thousand and after exploring the data I understood that half the reviews were positive and negative. So I experimented with more negative features and then more positive features and I found that with more positive features I slowly accumulated a higher score. Upon reaching above 70% I decided I was satisfied, but it is interesting that for some reason I have only 5 negative features and 18 positive to reach this score.

# In[9]:


plt.figure(figsize=(8,8))
sns.heatmap(bark.corr(), square=True);


# In[10]:


bark.head()


# In[11]:


bib = bark['feels']
yelp = bark[feato_feels]


# In[12]:


bern.fit(yelp, bib)
y_rev = bern.predict(yelp)

print("Mislabeled points out of {}: {}".format(
    yelp.shape[0], (bib != y_rev).sum()))


# It is interesting to find that with the difference between amazon product reviews and restaurant reviews that the scores are fairly similar with a slightly lower score for the yelp reviews. It does bother me that there is a serious imbalance of negative and positive features. 
