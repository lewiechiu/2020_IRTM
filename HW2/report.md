# HW2 Report

## Author 
B06705023 資管四 邱廷翔
## Environment
* python >= 3.6
* Linux >= 16.04

## Requirments
* nltk

To install the required libraries, run the following command.
```shell
pip install -r requirements.txt
```

## Executing the code
```
python main.py
```
* Before running the code, make sure the directory *IRTM* is present.
* Output will be saved in *dictionary.txt* and *tfidf* folder.

## Program descriptions
The program can be broke into several phases.

1. Traverse the folder

    * For each file:

        1. Use *tokenization*, which is from HW1, to preprocess each line of the document.
        2. After we acquired a set of tokens used in the document, update the *doc_freq* dictionary.
        3. *doc_freq* is a dictionary with *key* being a corpus and *value* being the corpus's document-wise frequency.

2. Write the *doc_freq* to file and name it *dictionary.txt*.
3. Calculate the *idf* dictionary given *doc_freq* dictionary.
    * $idf=\log_{10}\frac{N}{df}$
4. For each file:
    1. Use *tokenization* to tokenize the whole document.
    2. For each term, calculate its *tf-idf* score.
    3. Generate a vector with size equal to the dictioanry size, and fill the tfidf scores into the corresponding position of corpus.