import pathlib
import string
import re
# import numpy as np
import pandas as pd
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

    def trim_text(self, text):
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

    def clean(self):
        df = pd.concat([self.train, self.test])
        df['text'] = df['text'].swifter.apply(self.trim_text)


if __name__ == "__main__":
    tn = TwNlp()
    tn.run()
