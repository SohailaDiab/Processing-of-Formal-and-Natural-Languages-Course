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

**Note:**
- **Distance** between two vectors of two words that are **One-Hot Encoded** is the 
**same** (either “2” for different words and “0” for same words)
- **Distance** between two vectors of two words that are **Integer Encoded** (vector 
length=1) **differs** based on integer values given to both words.

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

- Distance between “plane” and “truck”
  - Integer encoded: (3-1)^2 = 4
  - one Hot Encoded: 1+1=2
- Distance between “plane” and “car”
  - Integer encoded: (1-0)^2 = 1
  - one Hot Encoded: 1+1=2

## 2. Bag of Words (BOW)
- The bag of words (BOW) model is a representation that turns arbitrary text into fixed-length vectors by counting how many times each word appears.
-  It is called a “bag” of words, because any information about the order or structure of words in the document is discarded.

BOW involves:
- A vocabulary of known words. (AKA Vocab)
- A measure of the presence of known words (count of how many times a word appears). (Histogram)

### Steps to represent Documents using BOW
1. Take the corpus (full set of documents) and tokenize the documents (sentences) into words... basically break it into single words: [["hello","world"],["i","love","pizza"]]
2. Filter out stop words (eg. the, in, and, of, an, ...). These are words that appear very frequently in the language but do not give us a lot of information, so we can remove these words without losing any information.
3. Take the remaining tokens (words) and stem/lemmatize them. This basically gets the root word for each word.
4. Design the vocabulary (Dictionary of UNIQUE words in the corpus).
5. For each word, count how many times it appears in each document(sentence).

NOTE: For documents not considered during Vocab design (were not in the training process), they may contain some words not in vocabulary (Out of Vocab). Those words are ignored.

### Example
**Corpus:**
1. "The cat sits on the mat"
2. "The dogs bark at the cat"
3. "The cat chases the mouse and the cat sits on the mat"

**Vocabulary:** (after removing stop words and stemming)
["cat", "sit", "mat", "dog", "bark", "chase", "mouse"]

**Bag of Words Embeddings:**

| Sentence                                     | cat | sit | mat | dog | bark | chase | mouse |
|----------------------------------------------|-----|-----|-----|-----|------|-------|-------|
| The cat sits on the mat                      | 1   | 1   | 1   | 0   | 0    | 0     | 0     |
| The dogs bark at the cat                     | 1   | 0   | 0   | 1   | 1    | 0     | 0     |
| The cat chases the mouse and the cat sits on the mat | 2   | 1   | 1   | 0   | 0    | 1     | 1     |

### Code
```py
from sklearn.feature_extraction.text import CountVectorizer
# text documents
text_1="Hany love going to school"
text_2="The school is far from Sara home"
text_3="Hany likes apple more than banana"
text_4="Sara likes apple too"
corpus=[text_1+" "+text_2+" "+text_3+" "+text_4]
# create Vectorizer
vectorizer = CountVectorizer()
# tokenize, build vocab
vectorizer.fit(corpus)
print("Vocab \n ========")
print(vectorizer.vocabulary_)

# encode document
vector = vectorizer.transform([text_1])
print(text_1," ",vector.toarray())
vector = vectorizer.transform([text_2])
print(text_2," ",vector.toarray())
vector = vectorizer.transform([text_3])
print(text_3," ",vector.toarray())
vector = vectorizer.transform([text_4])
print(text_4," ",vector.toarray())
vector = vectorizer.transform(corpus)
print(corpus," ",vector.toarray())
```
```
Output:

Vocab 
========
{'hany': 5, 'love': 9, 'going': 4, 'to': 15, 'school': 12, 'the': 14, 'is': 7, 'far': 2, 'from': 3, 'sara': 11, 'home': 6, 'likes': 8, 'apple': 
0, 'more': 10, 'than': 13, 'banana': 1, 'too': 16}

Hany love going to school [[0 0 0 0 1 1 0 0 0 1 0 0 1 0 0 1 0]]

The school is far from Sara home [[0 0 1 1 0 0 1 1 0 0 0 1 1 0 1 0 0]]

Hany likes apple more than banana [[1 1 0 0 0 1 0 0 1 0 1 0 0 1 0 0 0]]

Sara likes apple too [[1 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1]]

'Hany love going to school The school is far from Sara home Hany likes apple more than banana Sara likes apple too'] 
 [[2 1 1 1 1 2 1 1 2 1 1 2 2 1 1 1 1]
```

## 3. TF-IDF
Term Frequency - Inverse Document Frequency

- TF-IDF gives more importance to words that are rare and important in a document, unlike Bag-of-Words (BoW) that just counts how many times each word appears. _Sometimes we call it “Term” instead of "Word"_.
- This helps TF-IDF capture the unique aspects of each document better.

- TF-IDF focuses on words that are actually meaningful in a document, ignoring common words like "the" or "and" that don't tell us much _(their large counts
means LOW discrimination power between documents)_. By doing this, TF-IDF reduces the noise in the document representation

### **What is Term Frequency (TF)?**
- This summarizes how often a given word appears within a document.
- Calculated as the ratio of the number of times a term appears in a document to the total number of terms in the document.
- Often normalized to prevent bias towards longer documents.

### **What is Inverse Document Frequency (IDF)?** 
- This down-scales words that appear a lot across documents.
- Calculated as the logarithm of the ratio of the total number of documents to the number of documents containing the term.
- Terms that appear in many documents will have a lower IDF, while terms that appear in fewer documents will have a higher IDF.

### **What does HIGH TF-IDF value for a word mean?**
**TF*IDF**
- **TF:** Word appears a lot <ins>in this document</ins>.
- **IDF:** Word rarely appears across <ins>all documents</ins>.
*TF-IDF defines the importance of a keyword or phrase within a document
[Same Word has different TF-IDF values in different documents]*

### 1. Calculate Term Frequency (TF)
There are 3 ways:
a.  Normalized Term Frequency
b. Logarithmically scaled Term Frequency
c. Augmented Term Frequency

#### a. Normalized Term Frequency

### $TF_{(t, d)} = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$
TF = (Number of times term t appeared in document d)/(Total number of terms in document d, counting repeated terms)

Where:

$f_{t,d}$ is the raw count of a term in a document, i.e., the number of times that term $t$ occurs in document $d$.

NOTE: 
- Dividing by denominator normalizes the value of TF irrespective of document length

#### b. Logarithmically scaled Term Frequency

### $TF_{(t, d)} = \log(1 + f_{t,d})$
TF = log(1 + Number of times term t appeared in document d)

Where:

$f_{t,d}$ is the raw count of a term in a document, i.e., the number of times that term $t$ occurs in document $d$.

NOTE:
- Logarithmic scale of $f_{t,d}$ value compensates different documents length
- Adding "1" to prevent log calculation error if $f_{t,d}$ is zero (word does not occur in document

#### b. Augmented Term Frequency

### $TF_{(t, d)} = 0.5 + 0.5 * \frac{f_{t,d}}{\max\{f_{t',d}: t' \in d\}}$

TF = 0.5 + 0.5 * (Number of times term t appeared in document d / Number of times the most frequently occurring term in the document appeared)

Where:

$f_{t,d}$ is the raw count of a term in a document, i.e., the number of times that term $t$ occurs in document $d$.

NOTE:
- This prevents bias towards longer documents
-  Ranges from 0.5 to 1


# Resources Used
- Lecture Slides
- [Bag of Words | Video by Quantopian](https://www.youtube.com/watch?v=IRKDrrzh4dE)
