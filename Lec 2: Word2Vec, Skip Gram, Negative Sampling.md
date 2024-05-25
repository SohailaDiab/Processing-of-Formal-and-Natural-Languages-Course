
## Continuous bag-of-words (CBOW)
- Predicts the middle word based on surrounding context words. The context consists of a 
few words before and after the current (middle) word.
- This architecture is called a bag-of-words model as the order of words in the context is not important.
- Several times faster to train than the skip-gram, slightly better accuracy for the frequent 
words

![image](https://github.com/SohailaDiab/Processing-of-Formal-and-Natural-Languages-Course/assets/70928356/013303c3-2980-47f6-ac57-f5793fae7f20)

**a bag-of-words model predicts a word given the neighboring context**

## Skip-Gram
- Predicts words within a certain range before and after the current word in the same 
sentence.
- Works well with small amount of the training data, represents rare words or phrases well

![image](https://github.com/SohailaDiab/Processing-of-Formal-and-Natural-Languages-Course/assets/70928356/45324536-37be-443d-ac6d-b199b573783e)

**a skip-gram model predicts the context (or neighbors) of a word, given the word itself**

## CBOW vs. Skip-Gram
- Skip-Gram model is a better choice most of the time due to its ability to predict infrequent 
words, but this comes at the price of increased computational cost.
- If training time is a big concern, and you have large enough data to overcome the issue of predicting infrequent words, CBOW model may be a more viable choice. 
