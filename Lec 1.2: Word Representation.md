# Word Representation for NLP [Word Embedding]

- Machines cannot understand words. Therefore, when we want to train a machine learning model on text, we must first represent this text as vectors.
- Vectors are derived from textual data, in order to reflect various linguistic properties of the 
text.

 This process of representing text as a vector is called **word embedding**.

### **The 2 main approaches for word embedding are:**
   - **Frequency Based Embedding**
     - One-Hot Encoded vector
     - Bag of Word (BOW) Count Vector
     -  Term Frequency- Inverse Document frequency (TF-IDF) Vector
     -  Co-Occurrence Vector
   - **Prediction Based Embedding**
     -  Word2Vec
        - CBOW
        - Skipgram
     - Glove

## 1. One-Hot Encoding
- It is a method of representing words as vectors, where each word in the vocabulary is represented by a unique binary vector.

- All elements in the vector are 0 except for a single element that is 1, indicating the position of the word in the vocabulary.

- The length of the vector is the length of the unique words in the vocabulary.

### Example
Assume we have a small vocabulary consisting of the words $\{\text{cat}, \text{dog}, \text{fish}\}$.

$$
\begin{array}{ccc}
\text{Word} & \text{One-Hot Encoding} \\
\hline
\text{cat} & [1, 0, 0] \\
\text{dog} & [0, 1, 0] \\
\text{fish} & [0, 0, 1] \\
\end{array}
$$

Now, let's consider a sentence with the words: "cat dog fish cat".

The one-hot encoded representations of the words in the sentence are as follows:

$$
\begin{aligned}
&\text{"cat"} \quad \rightarrow \quad [1, 0, 0] \\
&\text{"dog"} \quad \rightarrow \quad [0, 1, 0] \\
&\text{"fish"} \quad \rightarrow \quad [0, 0, 1] \\
&\text{"cat"} \quad \rightarrow \quad [1, 0, 0] \\
\end{aligned}
$$

Thus, the sentence "cat dog fish cat" can be represented by the sequence of one-hot encoded vectors:

$$
\begin{aligned}
&\text{"cat"} \quad \rightarrow \quad [1, 0, 0] \\
&\text{"dog"} \quad \rightarrow \quad [0, 1, 0] \\
&\text{"fish"} \quad \rightarrow \quad [0, 0, 1] \\
&\text{"cat"} \quad \rightarrow \quad [1, 0, 0] \\
\end{aligned}
$$

So, the entire sentence can be represented as:

$$
\begin{bmatrix}
[1, 0, 0] \\
[0, 1, 0] \\
[0, 0, 1] \\
[1, 0, 0]
\end{bmatrix}
$$

This matrix shows the one-hot encoded vectors for each word in the sentence, capturing the presence and order of the words.

### Code
```py
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
Words = ['car', 'plane', 'car', 'truck', 'truck', 'plane', 'ship', 'ship', 'car']
Words_as_array = array(Words)
print(Words_as_array)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Words_as_array) # List 9 values
print("Integer Encoded\n==============\n",integer_encoded)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1) # Array 9x1
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(“One Hot\n=======\n”,onehot_encoded)
```
```
Output:
['car' 'plane' 'car' 'truck' 'truck' 'plane' 'ship' 'ship' 'car']

Integer Encoded
==============
[0 1 0 3 3 1 2 2 0]

One Hot
=======
[[1. 0. 0. 0.]
[0. 1. 0. 0.]
[1. 0. 0. 0.]
[0. 0. 0. 1.]
[0. 0. 0. 1.]
[0. 1. 0. 0.]
[0. 0. 1. 0.]
[0. 0. 1. 0.]
[1. 0. 0. 0.]]
```






