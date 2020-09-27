# HW1 Report
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
python pa1.py
```
* Before running the code, make sure the *input.txt* is present.
* Output will be saved in *result.txt*.

## Program descriptions
The program can be broke into several phases.

1. Read the text file, convert it to all lower cases and split the string by space.
2. Use Porter's algorithm to stem the word, and remove stop words if it is in the stopword list.
3. Write file to disk.