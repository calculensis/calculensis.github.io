Title: predicting review sentiment with a neural net
Date: 2023-04-09
Category: machine learning
Tags: review sentiment
Slug: predicting review sentiment with a neural net
Authors: Kayla Lewis
Summary: we construct a neural net for sentiment prediction from scratch using numpy

<img align=right src="./images/star-rating.jpg" width="180"/>

In a previous post we used scikit learn to predict customer sentiment from product reviews; in this post we'll do the same but from scratch using Numpy. First we load the reviews and one-hot encode them.

```python
import numpy as np
import pandas as pd
import string
import sys

with open('/home/kayla/pylearn/sentiment_numpy/train.ft.txt') as file:
    reviews = file.readlines()

sentence_list = reviews[0:1000]
sentences     = pd.DataFrame(sentence_list,columns=['review'])

def one_hot(sentences):
    # get a list of common words to exclude
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))

    # takes a data frame where each row is a sentence and one-hot
    # encodes the words of each sentence
    encoded = sentences.drop('review',axis=1)

    # first remove all punctuation, make lower case, and collect the
    # words
    for i in range(sentences.shape[0]):
        words_list = []
        current_sentence = \
                sentences.iloc[i,0].translate(str.maketrans('','', \
                                              string.punctuation))
        current_sentence = current_sentence.lower()
        filtered = [word for word in current_sentence.split() \
                    if word not in stops]
        words_list = words_list + filtered
        words_list = list(set(words_list))
        encoded.loc[i,words_list] = 1

    encoded = encoded.fillna(0)
    encoded = encoded.iloc[:,0:].astype('int')
    label1_col = encoded.pop('label1')
    label2_col = encoded.pop('label2')
    encoded.insert(0,'label2',label2_col)
    encoded.insert(0,'label1',label1_col)
    return encoded

reviews = one_hot(sentences)
reviews.to_csv('amazon_reviews.csv',index=False)
```

Next, let's create training and test sets as well as set up a neural net to feed the data into.


```python
reviews = pd.read_csv('amazon_reviews.csv')
labels  = reviews[['label1','label2']]
reviews = reviews.iloc[:,2:reviews.shape[1]]

# set up the training and test sets
train_frac = 0.8
num_train = int(train_frac*reviews.shape[0])
num_test  = reviews.shape[0]-num_train
Xtrain = reviews[0:num_train]
Xtest  = reviews[num_train:num_train+num_test]
ytrain = labels[0:num_train]
ytest  = labels[num_train:num_train+num_test]

# define activation functions and their derivatives
def activ_func(type_,layer):
    if (type_==1):
        return np.tanh(layer)
    elif (type_==2):
        return np.exp(layer)/np.exp(layer).sum()

def activ_deriv(type_,layer):
    if (type_==1):
        return 1.0-np.tanh(layer)**2.0
    elif (type_==2):
        softmax = np.exp(layer)/np.exp(layer).sum()
        return softmax*(1.0-softmax)

# set up neural net weights and layers
num_layers = 6
num_input  = reviews.shape[1]
num_output = 2
num_hidden = 50
layers  = []
weights = []
np.random.seed(2)
layers.append(np.zeros((1,num_input)))
weight_scale = 0.2
for i in range(num_layers-1):
    layer_num = i+1
    layers.append(np.zeros((1,num_hidden)))
    if (layer_num==1):
        weights.append(weight_scale* \
                np.random.random((num_input,num_hidden))- \
                weight_scale/2.0)
    elif (layer_num==(num_layers-1)):
        weights.append(weight_scale* \
                np.random.random((num_hidden,num_output))- \
                weight_scale/2.0)
    else:
        weights.append(weight_scale* \
                np.random.random((num_hidden,num_hidden))- \
                weight_scale/2.0)
```

Now let's train the model.


```python
max_iter = 200
err_tol  = 0.2
alpha    = 0.1
err      = 1.e5
iter_num = 0
dropout  = 1
while ((err>err_tol)&(iter_num<max_iter)):
    err = 0.0
    for i in range(Xtrain.shape[0]):
        # forward propagation step
        if (dropout==1):
            dropout_mask = \
                np.random.randint(0,2,(num_layers-2,num_hidden))
        layers[0] = Xtrain.iloc[i].to_numpy().reshape(1,num_input)
        for j in range(1,len(layers)):
            if (j<(len(layers)-1)):
                if (dropout==1):
                    layers[j] = 2.0*dropout_mask[j-1]* \
                        activ_func(1,np.matmul(layers[j-1], \
                        weights[j-1]))
                else:
                    layers[j] = \
                        activ_func(1,np.matmul(layers[j-1], \
                        weights[j-1]))
            else:
                layers[j] = \
                    activ_func(2,np.matmul(layers[j-1],weights[j-1]))

        # backprop step
        # form the diagonal matrices for the derivatives of the
        # activations for each layer
        diag_derivs = []
        for j in range(1,len(layers)):
            if (j<(len(layers)-1)):
                type_ = 1
                if (dropout==1):
                    derivs = \
                        np.diag(dropout_mask[j-1]* \
                        activ_deriv(type_,layers[j]).ravel())
                else:
                    derivs = \
                        np.diag(activ_deriv(type_,layers[j]).ravel())
            else:
                type_ = 2
                derivs = \
                    np.diag(activ_deriv(type_,layers[j]).ravel())
            diag_derivs.append(derivs)

        # update the weights
        correct_ans = ytrain.iloc[i,0:num_output].to_numpy(). \
                      reshape(1,num_output)
        delta = layers[num_layers-1]-correct_ans
        mat_prod = np.matmul( \
                             delta,diag_derivs[num_layers-2])
        weights[num_layers-2] += -alpha* \
                np.outer(layers[num_layers-2],mat_prod)
        for j in range(0,len(weights)-1):
            # update the weights behind each layer working from
            # output toward input
            weight_index = num_layers-j-2
            diag_index   = len(weights)-j-2
            #print(weight_index,diag_index)
            next_pair = np.matmul(weights[weight_index].T, \
                                  diag_derivs[weight_index-1])
            mat_prod = np.matmul(mat_prod,next_pair)
            weights[diag_index] += -alpha* \
                np.outer(layers[weight_index-1],mat_prod)
        err += (delta**2.0).sum()

    err /= Xtrain.shape[0]
    print(iter_num+1,err)
    iter_num += 1

if (iter_num<max_iter):
    print('converged at iteration: ',iter_num)
    print('training set error: ',err)
```

    1 0.49980698697558884
    2 0.49883427876339753
    3 0.501143717639368
    4 0.4961312915885812
    5 0.4974398386496625
    6 0.48694445301040384
    7 0.47873258603941676
    8 0.4627638083564027
    9 0.45554736881128927
    10 0.43554726678092753
    11 0.4160133901300509
    12 0.3905081397950724
    13 0.34602970606645256
    14 0.31516613543381433
    15 0.26634745838598434
    16 0.26187804345625254
    17 0.23263605939285184
    18 0.18831707208296627
    converged at iteration:  18
    training set error:  0.18831707208296627


Finally let's check the model's out-of-sample accuracy.


```python
# check test set accuracy
num_correct = 0
for i in range(Xtest.shape[0]):
        # forward propagation step
        layers[0] = Xtest.iloc[i].to_numpy().reshape(1,num_input)
        for j in range(1,len(layers)):
            if (j<(len(layers)-1)):
                layers[j] = \
                    activ_func(1,np.matmul(layers[j-1],weights[j-1]))
            else:
                layers[j] = \
                    activ_func(2,np.matmul(layers[j-1],weights[j-1]))

        correct_ans = ytest.iloc[i,0:num_output].to_numpy(). \
                      reshape(1,num_output)
        if (layers[num_layers-1].round()==correct_ans).all():
            num_correct += 1

percent_correct = (num_correct/Xtest.shape[0])*100.0
print("percent test set accuracy: ",percent_correct)
```

    percent test set accuracy:  79.0


79% accuracy on the test set isn't bad considering that we only used 1000 reviews to train the model. We kept the number low because our simple one-hot encoding module isn't optimized for handling large datasets.

[Discuss on Twitter](https://twitter.com/Estimatrix/status/1555693184977600512?s=20&t=YFPoxpEQ2Qp14U4FliD7fA)
