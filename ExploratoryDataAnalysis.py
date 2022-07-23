import pandas as pd
import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud,STOPWORDS


def explode(df, col):
    s = df[col].apply(pd.Series).add_prefix(col + '_')
    return pd.concat([df.drop([col], axis=1), s], axis=1)


def create_date_columns(original_df):
    df = original_df.copy()
    df['created_at'] = df['created_at'] \
        .apply(lambda x: datetime.datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y'))
    df['year'] = df['created_at'].apply(lambda x: x.year)
    df['month'] = df['created_at'].apply(lambda x: x.month)
    df['day'] = df['created_at'].apply(lambda x: x.day)
    df['hour'] = df['created_at'].apply(lambda x: x.hour)
    return df


def filter_tweets_by_user_count(df, lb=10, ub=30):
    n_rows_before = df.shape[0]
    users_df = df[['id', 'user_id']].groupby('user_id').count()
    users_df = users_df[(users_df.id > lb) & (users_df.id < ub)]
    res = df[df.user_id.isin(users_df.index)]
    n_rows_after = res.shape[0]
    print(f'Removed {n_rows_before - n_rows_after} out of {n_rows_before} rows.')
    return res


def create_scaled_favorite_count(df_original):
    df = df_original.copy()
    user_stats = df[['user_id', 'favorite_count']].groupby('user_id').agg(['mean', 'std', 'count'])
    mean = user_stats[('favorite_count',  'mean')]
    std = user_stats[('favorite_count',  'std')]
    mu = lambda uid: mean[mean.index == uid].values[0]
    sigma = lambda uid: std[std.index == uid].values[0]
    df['scaled_favorite_count'] = df.apply(lambda row: (row['favorite_count'] - mu(row['user_id'])) / sigma(row['user_id']), axis = 1)
    return df


def filter_tweets(tweets):
    df = pd.DataFrame([t._json for t in tweets])
    df = df.drop_duplicates(subset='id')
    df = explode(df, 'user')
    df = create_date_columns(df)
    df = filter_tweets_by_user_count(df, lb=1, ub=100)
    df = create_scaled_favorite_count(df)
    df = df[df['scaled_favorite_count'].notna()]
    df = df[~df.duplicated('text')]
    df = df[~df.text.str.contains('Calculus')]
    return df


def plot_user_counts_distribution(df, title, path=None):
    user_tweet_counts = df[['id', 'user_id']].groupby('user_id').count().values
    user_tweet_counts = sorted([x[0] for x in user_tweet_counts])
    plt.hist(user_tweet_counts, bins=20)
    plt.title(title)
    plt.xlabel('Number of tweets')
    plt.ylabel('Number of users')
    if path:
        plt.savefig(path)
    plt.show()


def plot_hours_distribution(df, title, path=None):
    hour_tweet_counts = df[['id', 'hour']].groupby('hour').count()
    hours, counts = hour_tweet_counts['id'].index, hour_tweet_counts['id'].values
    plt.bar(hours, counts)
    plt.title(title)
    plt.xlabel('Tweeting hour')
    plt.ylabel('Number of tweets')
    if path:
        plt.savefig(path)
    plt.show()


def plot_days_distribution(df, title, path=None):
    days_tweet_counts = df[['id', 'day']].groupby('day').count()
    days, counts = days_tweet_counts['id'].index, days_tweet_counts['id'].values
    days = [str(d) for d in days]
    plt.bar(days, counts)
    plt.title(title)
    plt.xlabel('Tweeting day')
    plt.ylabel('Number of tweets')
    if path:
        plt.savefig(path)
    plt.show()


def plot_days_hours_distribution(df, title, path=None):
    days_hours_tweet_counts = df[['id', 'day', 'hour']].groupby(['day', 'hour']).count()
    days_hours, counts = days_hours_tweet_counts['id'].index, days_hours_tweet_counts['id'].values
    days_hours = [str(t) for t in days_hours]
    plt.bar(days_hours, counts)
    plt.title(title)
    plt.xlabel('Tweeting day and hour')
    plt.ylabel('Number of tweets')
    plt.xticks([], [])
    if path:
        plt.savefig(path)
    plt.show()


def plot_scaled_favorite_count(df, title, path=None):
    plt.hist(df['scaled_favorite_count'])
    plt.title(title)
    plt.xlabel('scaled favorite count')
    plt.ylabel('number of tweets')
    if path:
        plt.savefig(path)
    plt.show()


def wordcloud(df, title, additional_stopwords=("https", "t", "co", "S", "U","amp"), path=None):
    text = " ".join(tweet for tweet in df.text)
    stopwords = set(STOPWORDS)
    stopwords.update(additional_stopwords)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.title(title)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    if path:
        plt.savefig(path)
    plt.show()