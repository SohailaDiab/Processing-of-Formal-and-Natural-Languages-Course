# FastText Word Embedding

- We need to represent words by numerical vectors to be able to pass the text to the model.

## Some examples of word embedding techniques
- One-hot encoding
- Bag of Words
- Skip-gram
  - Slow since Softmax is used, since softmax contains a vector with the size of the WHOLE vocab
- Negative sampling skip gram
  - Same as skip-gram, but with a workaround that makes it faster than skip-gram. Instead of using the whole vocab in softmax
- FastText (what will be covered here)
