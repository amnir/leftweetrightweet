import csv
from datetime import datetime
import io
import os
import re
import string
import numpy as np
import gensim.models.keyedvectors as word2vec
from keras.utils import np_utils
from keras_preprocessing import sequence
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, mean_squared_error, mean_absolute_error)
import hebrew_tokenizer as ht
import tweepy
import pandas as pd
from gensim import corpora
from keras import Sequential, callbacks
from keras.callbacks import CSVLogger
from keras.layers import Embedding, SimpleRNN, Dense, Activation, LSTM
from sklearn.model_selection import train_test_split

from tweet_dumper import get_all_tweets, get_all_tweets2

consumer_key = os.environ['consumer_key']
consumer_secret = os.environ['consumer_secret']
access_key = os.environ['access_key']
access_secret = os.environ['access_secret']

UNK = "UNK"

def main():
    print("Maor")
    # authorize twitter, initialize tweepy
    download_tweets()


def download_tweets():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    right_wing = ["naftalibennett", "ronisassover", "bezalelsm", "gWBnMgXPN6kCe6a", "Likud_Party", "erane67",
                  "Ayelet__Shaked", "netanyahu", "oren_haz", "bezalelsm", "YehudahGlick", "dudiamsalem", "giladerdan1",
                  "gidonsaar", "KahlonMoshe", "YinonMagal", "ErelSegal", "davidbitan", "moshefeiglin", "NirBarkat", ]
    left_wing = ["zehavagalon", "tamarzandberg", "MeravMichaeli", "MeravMichaeli", "Syechimovich", "mkgilon",
                 "GabbayAvi",
                 "ishmuli", "cabel_eitan", "EldadYaniv", "yarivop", "amirperetz", "StavShaffir", "dovhanin",
                 "Isaac_Herzog",
                 "NitzanHorowitz", "machanetzioni"]
    print(len(right_wing))
    print(len(left_wing))
    # nafatali_tweets = get_all_tweets2("naftalibennett", auth, 1)
    # 0 - left wing, 1 - right wing
    [get_all_tweets(screen_name, auth, 0) for screen_name in left_wing]
    [get_all_tweets(screen_name, auth, 1) for screen_name in right_wing]


#

def clean_tweet(tweet):
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'https…', '', tweet)
    # remove hashtags - only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'@\S+', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    return tweet


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def clean_tweets():
    with open('clean_tweets.csv', 'a', newline='', encoding='utf-8') as clean_tweets_csv:
        writer = csv.writer(clean_tweets_csv, quoting=csv.QUOTE_ALL)
        with open('tweets.csv', 'r', encoding='utf-8') as tweets:
            tweets = csv.reader(tweets)
            rows = [[clean_tweet(tweet[0]), tweet[1]] for tweet in tweets]
            writer.writerows(rows)


def build_dict():
    with open('clean_tweets.csv', 'r', encoding='utf-8') as tweets:
        tweets = csv.reader(tweets)
        all_words = [UNK]
        for tweet in tweets:
            tokenized = ht.tokenize(tweet[0])
            words = [token for grp, token, token_num, (start_index, end_index) in tokenized]
            all_words.append(words)

        return corpora.Dictionary([word for word in all_words])

if __name__ == '__main__':
    # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    # auth.set_access_token(access_key, access_secret)
    # get_all_tweets2("zehavagalon", auth, 0)
    #


    dict = build_dict()
    # print(len(dict))
    # print(dict.token2id["במדינה"])
    # print(dict[3075])

    data = pd.read_csv('tweets.csv')
    print(data.head())
    X = data.text
    y = data.drop('text', axis=1)
    X_train, X_test, y_train, orig_y_test = train_test_split(X, y, test_size=0.2)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(orig_y_test)

    train_tweets = []
    for train in X_train:
        tweet = clean_tweet(train)
        words = [token for grp, token, token_num, (start_index, end_index) in ht.tokenize(tweet)]
        vector = [dict.token2id[word] for word in words]
        train_tweets.append(vector)

    train_tweets = sequence.pad_sequences(np.array(train_tweets), maxlen=150)

    test_tweets = []
    for test in X_test:
        tweet = clean_tweet(test)
        words = [token for grp, token, token_num, (start_index, end_index) in ht.tokenize(tweet)]
        vector = [dict.token2id[word] for word in words]
        test_tweets.append(vector)

    test_tweets = sequence.pad_sequences(np.array(test_tweets), maxlen=150)

    # LSTM
    model = Sequential()
    model.add(Embedding(len(dict.keys()), 256, dropout=0.2))
    model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # model.load_weights("lstm_model.hdf5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1,
                                             save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('training_set_iranalysis1.csv', separator=',', append=False)

    model.fit(train_tweets, y_train, epochs=3, batch_size=256, validation_data=(test_tweets, y_test),
              verbose=1, callbacks=[checkpointer, csv_logger])
    model.save("lstm_model.hdf5")

    test_pred = model.predict_classes(test_tweets)
    accuracy = accuracy_score(orig_y_test, test_pred)
    print("real tweets acc" + str(accuracy))
