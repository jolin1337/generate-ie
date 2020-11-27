''' Read wikidata article '''
import sys
import re
import pandas
import numpy as np

import matplotlib
matplotlib.use('agg')


def split_by_special_token(summary):
    str_tokens = re.sub(r'[^a-zåäöÅÄÖ\- ]', ' ', summary['summary'].lower())
    tokens = ' '.join(w for w in str_tokens.split(' ') if w != '')
    summary['summary'] = tokens
    return summary


def remove_stop_words(summary):
    stop_words = open('stopwords.txt', 'r').read().split('\n')
    for word in stop_words:
        summary['summary'] = summary['summary'].replace(' ' + word + ' ', ' ')
    return summary


def ngram(summary, g=3):
    tokens = summary['summary'].split(' ')
    res = []
    for n in range(1, min(len(tokens), g + 1), 1):
        sub_tokens = []
        for i in range(0, len(tokens) - n + 1, 1):
            sub_tokens.append(' '.join(tokens[i:i+n]))
        res.append(sub_tokens)
    return res


def main(articles_file):
    articles = pandas.read_csv(articles_file)
    article_special = split_by_special_token(articles.loc[1])
    article_no_stop_words = remove_stop_words(article_special)
    article_ngram = ngram(article_no_stop_words, g=10)
    print(article_ngram)
    return
    article_lengths = articles['summary'].apply(lambda x: len(x))
    print(article_lengths.mean())
    print(article_lengths.max())
    print(article_lengths.min())
    max_length = article_lengths.max()
    print(articles.iloc[article_lengths.idxmax()])
    print(articles.iloc[article_lengths.idxmin()])
    print(articles.iloc[300]['summary'])
    frequency, bins = np.histogram(article_lengths, bins=300)

    print(articles)
    for i, (b, f) in enumerate(zip(bins[1:], frequency)):
        if f == 1:
            print(round(b, 1), '1/one')
        else:
            print(round(b, 1), ' '.join(np.repeat('*', 20 * f / max(frequency))))
        # print(round(b, 1), ' '.join(np.repeat('*', f)))


if __name__ == '__main__':
    args = ['articles.csv']
    args[0] = sys.argv[1] or args[0]
    main(args[0])
