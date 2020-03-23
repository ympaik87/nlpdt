import pathlib
import string
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam
from tqdm import tqdm
import pandas as pd
from nltk.tokenize import word_tokenize
import swifter


class TwNlp:
    def __init__(self):
        self.data_p = pathlib.Path(
            'D:/kaggle/disaster_tweets/nlp-getting-started/')

    def run(self):
        self.read_data()

    def read_data(self):
        self.test = pd.read_csv(self.data_p/'test.csv')
        self.train = pd.read_csv(self.data_p/'train.csv')

    def clean_text(self, text):
        url = re.compile(r'https?://\S+|www\.\S+')
        trimmed1 = url.sub(r'', text)

        html = re.compile(r'<.*?>')
        trimmed2 = html.sub(r'', trimmed1)

        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        trimmed3 = emoji_pattern.sub(r'', trimmed2)

        table = str.maketrans('', '', string.punctuation)
        trimmed4 = trimmed3.translate(table)
        return trimmed4

    def create_corpus_new(self, txt_srs):
        corpus = list()
        for tw in tqdm(txt_srs):
            words = [word.lower() for word in word_tokenize(tw)]
            corpus.append(words)
        return corpus

    def read_glove(self):
        p = self.data_p.parent/'glove.twitter.27B/glove.twitter.27B.100d.txt'
        embedding_dict = dict()
        with open(p, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vectors = np.asarray(values[1:], 'float32')
                embedding_dict[word] = vectors
        return embedding_dict

    def prep_data(self):
        df = pd.concat([self.train, self.test])
        df['text'] = df['text'].swifter.apply(self.clean_text)
        return self.create_corpus_new(df['text'])

    def glove(self):
        max_len = 50
        tokenizer_obj = Tokenizer()
        corpus = self.prep_data()
        tokenizer_obj.fit_on_texts(corpus)
        sequences = tokenizer_obj.texts_to_sequences(corpus)

        tweet_pad = pad_sequences(sequences, maxlen=max_len,
                                  truncating='post', padding='post')

        word_ind = tokenizer_obj.word_index
        num_words = len(word_ind) + 1
        embedding_matrix = np.zeros((num_words, 100))

        for word, i in tqdm(word_ind.items(), total=num_words-1,
                            desc='emb_vec'):
            if i < num_words:
                emb_vec = embedding_matrix.get(word)
                if emb_vec is not None:
                    embedding_matrix[i] = emb_vec
        return embedding_matrix, num_words

    def model(self):
        embedding_matrix, num_wrods = self.glove()
        model = Sequential()
        embedding = Embedding(
            num_words, 100, embeddings_initializer=Constant(embedding_matrix),
            input_length=MAX_LEN, trainable=False
        )
        model.add(embedding)
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        optimzer = Adam(learning_rate=3e-4)

        model.compile(loss='binary_crossentropy',
                      optimizer=optimzer, metrics=['accuracy'])



if __name__ == "__main__":
    tn = TwNlp()
    tn.run()
