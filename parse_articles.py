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
    #args = ['articles.csv']
    #args[0] = sys.argv[1] or args[0]
    #main(args[0])

    import random
    import math
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3
    import pickle
    with open('data/validation.csv') as val:
        content = val.read()
        sets = []
        correct_sets = []
        sources = ['openie', 'generateie', 'minie']
        for source in ['all', *sources, 'overlap']:
            if source == 'all':
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n'))]
            elif source == 'overlap':
                openie_triples = [v.split('\t')[-1] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'openie']
                genie_triples = [v.split('\t')[-1] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'generateie']
                minie_triples = [v.split('\t')[-1] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'minie']
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[-1] in openie_triples and v.split('\t')[-1] in genie_triples and v.split('\t')[-1] in minie_triples]
                continue
            else:
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == source]
                sets.append(set([v.split('\t')[-1] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == source]))
                correct_sets.append(set([v.split('\t')[-1] for v in content.split('\n')[1:] if v.startswith('TRUE') and v.split('\t')[1] == source]))
            vals.sort(key=lambda x: random.random())
            batches = 10
            batch = int(len(vals)/batches)
            scores = []
            for x in range(batches):
                current = vals[x*batch:(x+1)*batch]
                score = sum(current) / len(current)
                scores.append(score)
            avg_score = sum(scores)/batches
            print("%s count %s" % (source, len(vals)))
            print("%s scores %s" % (source, scores))
            print("%s avg score %s" % (source, avg_score))
            variance = 0
            for s in scores:
                variance += (s - avg_score) ** 2
            print("%s variance %s" % (source, variance/(batches-1)))
            print("%s std %s" % (source, math.sqrt(variance/(batches-1))))
            print()
        # Make the diagrams
        venn3(sets, set_labels=sources)
        plt.savefig('results/venn3-ie-manual-triples.png')
        plt.clf()
        venn3(correct_sets, set_labels=sources)
        plt.savefig('results/venn3-ie-manual-correct-triples.png')
        plt.clf()

        qsets = []
        for source in sources:
            with open(f'results/matchie_{source}_eng.pkl', 'rb') as ie:
                counters = pickle.load(ie)
                triples = set(counters.triple_counter.keys())
                qsets.append(triples)
        venn3(qsets, set_labels=sources)
        plt.savefig('results/venn3-ie-quantitative.png')