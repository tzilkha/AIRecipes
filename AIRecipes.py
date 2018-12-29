import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split as tts
from collections import Counter
import string
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

class text_preprocessor:
    data_x = []
    data_y = []
    word_count = {}
    data_x_prep = []
    data_y_prep = []
    clf = None
    def __init__(self, x, y):
        print("Initializing text_preprocessor object.")
        self.data_x = x
        self.data_y = y
        self.count()
    def count(self):
        print("Counting...")
        words = []
        for entry in self.data_x:
            for w in entry.split(" "):
                words+=[w.strip(string.punctuation).lower()]
        self.word_count=Counter(words)
        self.word_count = {k:v for k,v in dict(self.word_count).items() if k.isalpha()}
        self.word_count = Counter(self.word_count).most_common(3000)
    def get_count(self):
        return self.word_count
    def preprocess(self):
        print("Preprocessing data...")
        train_content_vec = []
        for entry in self.data_x:
            entry_l = [x.strip(string.punctuation) for x in entry.split()]
            entry_vector = []
            for word in self.word_count:
                entry_vector.append(entry_l.count(word[0]))
            train_content_vec.append(entry_vector)
        self.data_x_prep = train_content_vec
        self.data_y_prep = self.data_y
    def shuffle(self):
        print("Shuffling...")
        temp = list(zip(self.data_x_prep, self.data_y_prep))
        feats = []
        labels = []
        for sample in range(len(temp)):
            element = random.choice(temp)
            feats.append(element[0])
            labels.append(element[1])
            temp.remove(element)
        self.data_x_prep, self.data_y_prep = feats, labels
    def NaiveBayes(self, p=0.2):
        print("Creating NaiveBayes...")
        x_train, x_test, y_train, y_test = tts(self.data_x_prep, self.data_y_prep, test_size = p)
        clf = MultinomialNB()
        clf.fit(x_train, y_train)
        preds = clf.predict(x_test)
        self.clf = clf
        print("Checking accuracy...")
        print("Accuracy:", accuracy_score(y_test, preds))
    def predict(self, entry, prob=False):
        if self.clf==None:
            print("Error - NaiveBayes not created.")
            return 0
        if type(entry) != list: entry = [entry]
        res = []
        probs = []
        for sample in entry:
            words = [x.strip(string.punctuation) for x in sample.split()]
            vec = []
            for word in self.word_count:
                vec.append(words.count(word[0]))
            res.append(self.clf.predict([vec])[0])
            if (prob): probs.append(self.clf.predict_proba([vec])[0])
        if (prob): return res, probs
        return res
    def get_labels(self):
        return list(set(self.data_y))
    def info(self):
        if len(self.data_x) == 0:
            print("No data initialized.")
            return 0
        print("Samples:", len(self.data_x))
        print("Label Sample Space:", len(self.get_labels()))
        if len(self.data_x_prep) == 0:
            print("Data not preprocessed.")
            return 0
        print("NaiveBayes created:", self.clf != None)



if __name__ == "__main__":
    data = pd.read_csv("tweet_data.csv", sep=",", names=["Label", "Entry", "Date", "Flag", "User", "Tweet"],
                       encoding="latin 1")
    data = data[["Label", "Tweet"]]
    data = data.sample(frac=1).reset_index(drop=True)
    x = list(data["Tweet"][0:10000])
    y = list(data["Label"][0:10000])

    # Feed to lists, corresponding text samples and labels
    c = text_preprocessor(x, y)
    c.preprocess()
    c.shuffle()
    c.NaiveBayes()
    res = c.predict(["i hate you", "i love it", ], True)
    c.get_labels()
    c.info()