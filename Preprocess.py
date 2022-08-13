import numpy as np
import pandas as pd
import torch
from datetime import datetime, date
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from emoji import demojize
from nltk.tokenize import TweetTokenizer

user_relevant_cols = ['user_created_at', 'user_protected',
                      'user_followers_count', 'user_friends_count', 'user_listed_count',
                      'user_favourites_count', 'user_statuses_count', 'user_default_profile']
tweet_tokenizer = TweetTokenizer()
bertweet = AutoModel.from_pretrained("vinai/bertweet-large")
auto_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large")
# sentence_transformer_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# def encode_sentence_transformer(normalized_tweet):
#     return sentence_transformer_model.encode(normalized_tweet)


def discrete_hour(h):
    if 0 <= h < 6: return 0
    elif 6 <= h < 12: return 1
    elif 12 <= h < 18: return 2
    elif 18 <= h < 24: return 3

'''
Implementation taken from:
https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
'''
def normalize_token(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalize_tweet(tweet):
    tokens = tweet_tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalize_token(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


def encode_bertweet(normalized_tweet):
    input_ids = torch.tensor([auto_tokenizer.encode(normalized_tweet)])
    with torch.no_grad():
        features = bertweet(input_ids)  # Models outputs are now tuples
    return features.last_hidden_state[0].mean(axis=0).numpy()


def encode_user_description(user_description):
    return encode_bertweet(user_description)


def encode_user_covariates(df_row):
    user = df_row[user_relevant_cols]
    user_age = date(2022, 7, 24) - datetime.strptime(user['user_created_at'], '%a %b %d %H:%M:%S +0000 %Y').date()
    user_age = user_age.days
    numerical_covariates = [user_age]
    for col in user_relevant_cols[1:]:
        numerical_covariates.append(int(user[col]))
    return np.array(numerical_covariates)

