Title: implementing a transformer to evaluate tweets
Date: 2023-05-13
Category: machine learning
Tags: transformers
Slug: implementing a transformer to evaluate tweets
Authors: Kayla Lewis
Summary: we implement a transformer architecture to predict whether tweets are worthy or subpar

<img align=right src="images/judge_hammer.jpg" width="150"/>

In this post implement a transformer architecture and use it to predict whether a tweet is good (i.e. with number of likes $>3$) or bad (number of likes $\le 3$). At the heart of the transformer is the self-attention operation, which allows the network to attend to all of the words in sentence at once. We implement it as follows.


```python
import torch.nn as nn
import torch
import torch.nn.functional as F

class self_attention(nn.Module):
    def __init__(self,vec_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.vec_dim   = vec_dim

        self.to_keys     = nn.Linear(vec_dim,vec_dim,bias=False)
        self.to_queries  = nn.Linear(vec_dim,vec_dim,bias=False)
        self.to_values   = nn.Linear(vec_dim,vec_dim,bias=False)
        self.unify_heads = nn.Linear(vec_dim,vec_dim,bias=False)
    def forward(self,x):
        num_batches, num_vecs, vec_dim = x.size()
        num_heads = self.num_heads

        trunc_dim = vec_dim // num_heads

        queries = self.to_queries(x)
        keys    = self.to_keys(x)
        values  = self.to_values(x)

        queries = \
            queries.view(num_batches,num_vecs,num_heads,trunc_dim)
        keys =    \
            keys.view(num_batches,num_vecs,num_heads,trunc_dim)
        values =  \
            values.view(num_batches,num_vecs,num_heads,trunc_dim)

        keys = keys.transpose(1, 2).contiguous(). \
            view(num_batches*num_heads,num_vecs,trunc_dim)
        queries = queries.transpose(1, 2).contiguous(). \
            view(num_batches*num_heads,num_vecs,trunc_dim)
        values = values.transpose(1, 2).contiguous(). \
            view(num_batches*num_heads,num_vecs,trunc_dim)

        dot_prod = torch.bmm(queries, keys.transpose(1, 2))
        dot_prod = dot_prod / (vec_dim**(1.0/2.0))
        dot_prod = F.softmax(dot_prod, dim=2)

        out = torch.bmm(dot_prod, values). \
            view(num_batches,num_heads,num_vecs,trunc_dim)
        out = out.transpose(1, 2).contiguous(). \
            view(num_batches,num_vecs,trunc_dim*num_heads)
        self.unify_heads(out)

        return out
```

Once we have self-attention defined, we insert it into a transformer.


```python
class transformer(nn.Module):
    def __init__(self,vec_dim,num_heads):
        super().__init__()

        self.attention = self_attention(vec_dim,num_heads)

        self.norm1 = nn.LayerNorm(vec_dim)
        self.norm2 = nn.LayerNorm(vec_dim)

        dim_multiplier = 4
        self.feed_forward = nn.Sequential( \
            nn.Linear(vec_dim,dim_multiplier*vec_dim),
            nn.ReLU(),
            nn.Linear(dim_multiplier*vec_dim,vec_dim))

    def forward(self,x):
        attended = self.attention(x)
        x = self.norm1(attended+x)

        feed_forward = self.feed_forward(x)
        return self.norm2(feed_forward+x)
```

For the full network we'll stack a number of transformers on top of one another; we'll also add a layer that predicts class log probabilities.


```python
class tweet_classifier(nn.Module):
    # vec_dim = dimension of the vector that a word maps to in
    #           the word embedding and also that a position
    #           maps to in the position embedding
    # seq_length = length of the sentence for the positional
    #              embedding
    # num_words = size of the dictionary for the word embedding
    def __init__(self,vec_dim,num_heads,depth,seq_length, \
                 num_words,num_classes):
        super().__init__()

        self.num_words = num_words
        self.word_emb = nn.Embedding(num_words,vec_dim)
        self.pos_emb = nn.Embedding(seq_length,vec_dim)

        tblocks = []
        for i in range(depth):
            tblocks.append(transformer(vec_dim=vec_dim, \
                                       num_heads=num_heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(vec_dim,num_classes)

    def forward(self,x):
        # x: A (batch_size, sent_len) tensor of integer values 
        #    representing words (in some predetermined vocabulary).
        # output: A (batch_size, num_classes) tensor of 
        #         log probabilities over the classes

        # generate word embeddings
        words = self.word_emb(x)
        batch_size, sent_len, vec_dim = words.size()

        # generate position embeddings
        positions = torch.arange(sent_len)
        positions = self.pos_emb(positions)[None, :, :]. \
                    expand(batch_size,sent_len,vec_dim)

        x = words + positions
        x = self.tblocks(x)

        # Average-pool over the sent_len dimension and project 
        # to class probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
```

And that takes care of the network architecture! Now we'll turn to the data, which consists of 4,000 tweets together with the number of likes won by each tweet. We need to clean the data up a bit before we can tokenize it; we also need to label each tweet based on its number of likes.


```python
import pandas as pd
import re
import sys

tweets_df = pd.read_csv("my_tweets.csv")

#remove all columns except those containing tweets
tweets_df = tweets_df.drop(columns=['Date Created', \
            'Source of Tweet'],axis=1).sample(4000)
tweets_df.drop(['Unnamed: 0'],axis=1,inplace=True)

# remove punctuation
tweets_df['Tweets'] = \
    tweets_df['Tweets'].map(lambda x: \
    re.sub('[,\.!?]','', x))

# remove all the mentions (@)
tweets_df['Tweets'] = \
    tweets_df['Tweets'].map(lambda x: \
    re.sub('@([a-zA-Z0-9_]{1,50})', '', x))

# convert the titles to lowercase
tweets_df['Tweets'] = \
    tweets_df['Tweets'].map(lambda x: x.lower())
tweets_df.to_csv("my_tweets_cleaned.csv",index=False)
```


```python
tweets = pd.read_csv("my_tweets_cleaned.csv")
threshold = 3
good_tweets = (tweets['Number of Likes']>threshold).astype(int)
bad_tweets  = (tweets['Number of Likes']<=threshold).astype(int)

tweets.drop(['Number of Likes'],axis=1,inplace=True)
tweets['good'] = good_tweets
tweets['bad']  = bad_tweets
tweets_df.to_csv("my_tweets_cleaned.csv",index=False)
```

Now that the tweets are cleaned up and labeled, it's time to tokenize them. We'll create a dictionary that maps each word to an integer; then we'll convert each tweet into a vector of integers. Our architecture already includes an embedding layer that will map each word into a vector, so each tweet with be represented by a series of vectors. We're also using positional embedding so that the network can keep track of how the words in each tweet are ordered.


```python
import pickle
import numpy as np

tweets = pd.read_csv('labeled_tweets.csv')

# convert each word into a number; covert each tweet into a
# vector with a predefined length. empty positions will be
# set to zero

# build up a set containing all the tweet words; also find the
# maximum tweet length
vocab = set()
max_len = 0
for i in range(tweets.shape[0]):
    word_list = tweets.iloc[i,0].split()
    if len(word_list)>max_len:
        max_len = len(word_list)
    vocab = vocab.union(set(word_list))
# make a dictionary pairing words to integers
words_to_nums = {word:index for index,word in enumerate(vocab)}

# save the dictionary to a file
with open('words_to_nums.pkl', 'wb') as fp:
    pickle.dump(words_to_nums, fp)
    print('dictionary saved to file!')

tweets_as_vecs = np.zeros((tweets.shape[0],max_len))
for i in range(tweets.shape[0]):
    word_list = tweets.iloc[i,0].split()
    tweet_vec = [words_to_nums[word] for word in word_list]
    tweets_as_vecs[i,0:len(tweet_vec)]=tweet_vec

tweets_as_vecs = pd.DataFrame(tweets_as_vecs)
tweets_as_vecs.to_csv('tweets_as_vecs.csv',index=False)
# convert the tweet_labels such that 0 = bad tweet, 
# 1 = good tweet, and there is a single integer label for each
# tweet
tweet_labels = tweets.iloc[:,1:3]
true_class = np.zeros((1,tweet_labels.shape[0]))
true_class = pd.Series(true_class.ravel())
for i in range(tweet_labels.shape[0]):
    true_class[i] = 1 - tweet_labels.iloc[i].argmax()
true_class = true_class.astype(int)
true_class.to_csv('tweet_labels.csv',index=False)
```

    dictionary saved to file!


Now we can train our predictor. We'll also save the model weights to a file afterward.


```python
from sklearn.model_selection import train_test_split
from torch import optim

tweets = pd.read_csv("tweets_as_vecs.csv")
labels = pd.read_csv("tweet_labels.csv")

# create the training and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(tweets,labels,test_size=0.33)

X_train = np.array(X_train)
X_test  = np.array(X_test)
y_train = np.array(y_train)
y_test  = np.array(y_test)
X_train = torch.tensor(X_train)
X_test  = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test  = torch.tensor(y_test)
batch_size = 20
vec_dim = 100
sent_len = X_train.shape[1]
num_heads = 4
depth = 3
seq_length = sent_len
num_words = int(tweets.max().max())+1
num_classes = 2
learning_rate = 0.1
err_tol = 0.1
max_epoch = 1000
epoch = 0
err = 1.e5
X_train = X_train.view(batch_size,int(X_train.shape[0]/batch_size), \
                       X_train.shape[1]).int()
y_train = y_train.view(batch_size,int(y_train.shape[0]/batch_size))
model = tweet_classifier(vec_dim,num_heads,depth,seq_length, \
                         num_words,num_classes)
loss_func = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(),learning_rate)
while ((epoch<max_epoch)&(err>err_tol)):
    err = 0.0
    for i in range(batch_size):
        outputs = model(X_train[i])
        targets = y_train[i]
        loss = loss_func(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        err += loss.item()

    err /= batch_size
    epoch += 1
    if (epoch%10==0):
        print("epoch: %d, loss: %f" % (epoch, err))

if (epoch<=max_epoch):
    print("converged at epoch: ",epoch)
    print("with error: ",err)

torch.save(model.state_dict(), 'model_weights.pth')
torch.save(X_test,"X_test.pt")
torch.save(y_test,"y_test.pt")
```

    epoch: 10, loss: 0.475420
    epoch: 20, loss: 0.473031
    epoch: 30, loss: 0.470421
    epoch: 40, loss: 0.466993
    epoch: 50, loss: 0.459974
    epoch: 60, loss: 0.445342
    epoch: 70, loss: 0.430670
    epoch: 80, loss: 0.415124
    epoch: 90, loss: 0.394816
    epoch: 100, loss: 0.366840
    epoch: 110, loss: 0.333874
    epoch: 120, loss: 0.316484
    epoch: 130, loss: 0.275802
    epoch: 140, loss: 0.250228
    epoch: 150, loss: 0.237210
    epoch: 160, loss: 0.230245
    epoch: 170, loss: 0.148748
    epoch: 180, loss: 0.127036
    converged at epoch:  185
    with error:  0.09630416203290224


Let's see how well the model performs out of sample. 


```python
tweets = pd.read_csv("tweets_as_vecs.csv")
X_test = torch.load("X_test.pt").int()
y_test = torch.load("y_test.pt")
num_sents = y_test.shape[0]

batch_size = 20
vec_dim = 100
sent_len = X_test.shape[1]
num_heads = 4
depth = 3
seq_length = sent_len
num_words = int(tweets.max().max())+1
num_classes = 2
model = tweet_classifier(vec_dim,num_heads,depth,seq_length, \
                         num_words,num_classes)
model.load_state_dict(torch.load('model_weights.pth'))
X_test = X_test.view(batch_size,int(X_test.shape[0]/batch_size), \
                     X_test.shape[1]).int()
y_test = y_test.view(batch_size,int(y_test.shape[0]/batch_size))
num_correct = 0
for i in range(batch_size):
    outputs = model(X_test[i])
    outputs = torch.exp(outputs).round().int().argmax(axis=1)

    targets = y_test[i]
    num_correct += (outputs==targets).sum().item()

frac_correct = num_correct/num_sents
print("percent correct OOS: ",frac_correct*100)
```

    percent correct OOS:  69.46969696969697


For a training set of about 3,000 tweets (i.e. 67% of the full 4,000 tweet data set), near 70% accuracy out of sample isn't too bad!

[Discuss on Twitter](https://twitter.com/Estimatrix/status/1555693184977600512?s=20&t=YFPoxpEQ2Qp14U4FliD7fA)