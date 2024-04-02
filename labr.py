import codecs
import numpy as np
import pandas as pd
import re

class LABR:
    def __init__(self, word_emb=False):
        self.REVIEWS_PATH = "EP4ASA/data/"
        self.CLEAN_REVIEWS_FILE = "reviews.tsv"
        self.word_emb = word_emb

    # Read the reviews file. Returns a tuple containing these lists:
    #   rating: the rating 1 -> 5
    #   body: the text of the review
    def read_review_file(self, file_name):
        reviews = codecs.open(file_name, 'r', 'utf-8').readlines()

        # remove comment lines and newlines
        reviews = [r.strip() for r in reviews if r[0] != u'#']

        # parse
        rating = list()
        body = list()
        for review in reviews:
            # split by <tab>
            parts = review.split(u"\t")

            # rating is first part and body is last part
            rating.append(int(parts[0]))
            if len(parts) > 4:
                body.append(parts[4])
            else:
                body.append(u"")

        return (rating, body)

    def read_clean_reviews(self):
         return self.read_review_file(self.REVIEWS_PATH + self.CLEAN_REVIEWS_FILE)

    # Reads a training or test file. The file contains the indices of the
    # reviews from the clean reviews file.
    def read_train_test_file(self, file_name):
        ins = open(file_name).readlines()
        ins = [int(i.strip()) for i in ins]

        return ins

    # A helpter function.
    def set_binary_klass(self, ar):
        ar[(ar == 1) + (ar == 2)] = 0
        ar[(ar == 4) + (ar == 5)] = 1

    # Returns (train_x, train_y, test_x, test_y)
    # where x is the review body and y is the rating (1->5 or 0->1)
    def get_train_test(self, klass = "2", balanced = "balanced"):
        (rating, body) = self.read_clean_reviews()
        rating = np.array(rating)
        body = pd.Series(body)

        train_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced+ "-train.txt")
        test_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced+ "-test.txt")

        train_ids = self.read_train_test_file(train_file)
        test_ids = self.read_train_test_file(test_file)

        train_x = body[train_ids]
        test_x = body[test_ids]
        train_y = rating[train_ids]
        test_y = rating[test_ids]

        if klass == "2":
            self.set_binary_klass(train_y)
            self.set_binary_klass(test_y)

        if self.word_emb: return list(train_x)+list(test_x)
        else: return (train_x, train_y, test_x, test_y)
