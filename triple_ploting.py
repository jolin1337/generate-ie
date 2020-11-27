
import random
import math
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np
import pickle
import pandas
import collections

def plot_venn_triples(sources = ['openie', 'generateie', 'minie', 'generateie_swe', 'generateie_crossref']):
    with open('data/validation.csv', 'r') as val:
        content = val.read()
        sets = []
        correct_sets = []
        for source in ['all', *sources, 'overlap']:
            if source == 'all':
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] in ['openie', 'generateie', 'minie']]
            elif source == 'overlap':
                openie_triples = [v.split('\t')[2] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'openie']
                genie_triples = [v.split('\t')[2] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'generateie']
                minie_triples = [v.split('\t')[2] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == 'minie']
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[2] in openie_triples and v.split('\t')[2] in genie_triples and v.split('\t')[2] in minie_triples]
                vals = vals[::3]
            else:
                vals = [1 if v.startswith('TRUE') else 0 for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == source]
                sets.append(set([v.split('\t')[2] for v in content.split('\n')[1:] if (v.startswith('TRUE') or v.startswith('n')) and v.split('\t')[1] == source]))
                correct_sets.append(set([v.split('\t')[2] for v in content.split('\n')[1:] if v.startswith('TRUE') and v.split('\t')[1] == source]))
            vals.sort(key=lambda x: random.random())
            batches = 1 #max(1, math.ceil(len(vals)/33))
            print(batches)
            batch = int(len(vals)/batches)
            while batch == 0:
                batches = int(batches/2)
                batch = int(len(vals)/batches)
            scores = []
            if batch != 0:
                for x in range(batches):
                    current = vals[x*batch:(x+1)*batch]
                    score = sum(current) / len(current)
                    scores.append(score)
            leftovers = len(vals) % batches
            if leftovers > 0:
                current = vals[-leftovers:]
                score = sum(current) / len(current)
                scores.append(score)
            avg_score = sum(scores)/len(scores)
            print("%s count %s" % (source, len(vals)))
            print("%s count correct %s" % (source, len([v for v in vals if v == 1])))
            print("%s count error %s" % (source, len([v for v in vals if v == 0])))
            print("%s scores %s" % (source, scores))
            print("%s avg score %s" % (source, avg_score))
            variance = 0
            for s in scores:
                variance += (s - avg_score) ** 2
            print("%s variance %s" % (source, variance/max(1, len(scores)-1)))
            print("%s std %s" % (source, math.sqrt(variance/max(1, len(scores)-1))))
            print("%s pvariance %s" % (source, scores[0] * (1-scores[0])/len(vals)))
            print("%s pstd %s" % (source, math.sqrt(scores[0] * (1-scores[0])/len(vals))))
            print()
        # Make the diagrams
        venns = [
            None,
            None,
            venn2,
            venn3,
        ]
        print(min(3, len(sets)))
        venns[min(3, len(sets))](sets[:3], set_labels=sources)
        plt.savefig('results/venn3-ie-manual-triples.png')
        plt.clf()
        venns[min(3, len(correct_sets))](correct_sets[:3], set_labels=sources)
        plt.savefig('results/venn3-ie-manual-correct-triples.png')
        plt.clf()

def plot_bar_acc(sources = ['OIE', 'GIE', 'GIE', 'MIE']):

    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')

    scores = []
    stds = []
    counts = []
    count_corrects = []
    names = []
    def add_data_to_plot(name, sub_df):
        names.append(name)
        count = sub_df['class'].count()
        correct_count = sub_df['class'].sum()
        score = correct_count / count
        counts.append(count)
        count_corrects.append(correct_count)
        scores.append(score)
        stds.append(math.sqrt(score * (1-score)/count))
        print(name, scores[-1], count_corrects[-1], counts[-1])

    df = pandas.read_csv('data/validation.csv', delimiter='\t', header=0)#.drop_duplicates(['source', 'joint_triple'])
    class_map = {
        'TRUE': 1,
        'n': 0
    }
    df['class'] = df.apply(lambda x: class_map.get(x['Correct']), axis=1)
    df = df.dropna()
    comp_df = df[(df.source == 'openie') | (df.source == 'generateie_crossref') | (df.source == 'generateie') | (df.source == 'minie')]
    triples = comp_df[['joint_triple', 'class']].drop_duplicates()
    print(triples)
    print(df[df.source == 'openie'])
    print(df[df.source == 'generateie_crossref'])
    print(df[df.source == 'minie'])
    add_data_to_plot("All", comp_df)
    for source in ['openie', 'generateie_crossref', 'minie']:
        name = f'{source[0].upper()}IE'
        match_df = comp_df[comp_df.source == source]
        add_data_to_plot(name, match_df)
    for a1, a2 in [('openie', 'generateie'), ('openie', 'minie'), ('generateie', 'minie')]:
        name = f'Match\n{a1[0].upper()}IE/{a2[0].upper()}IE'
        tmp_df = comp_df[(df.source == a1) | (df.source == a2)].drop_duplicates(['source', 'joint_triple'])
        count_df = tmp_df.groupby('joint_triple').count()
        match_df = tmp_df[tmp_df.joint_triple.isin(count_df[count_df['Correct'] > 1].index)] #.drop_duplicates(['joint_triple'])
        add_data_to_plot(name, match_df)
    tmp_df = comp_df.drop_duplicates(['source', 'joint_triple'])
    count_df = tmp_df.groupby('joint_triple').count()
    match_df = tmp_df[tmp_df.joint_triple.isin(count_df[count_df['Correct'] > 2].index)] #.drop_duplicates(['joint_triple'])
    add_data_to_plot("Match all", match_df)


    width = 0.35  # the width of the bars
    ind = np.arange(len(scores))
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, counts, width,
                    color='SkyBlue', label='Counts')
    rects2 = ax.bar(ind + width/2, count_corrects, width,
                    color='IndianRed', label='Correct counts')
    ax.set_ylabel('Nr of triples')
    ax.set_title('Algorithm comparision')
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.legend()
    autolabel(rects1, "left")
    autolabel(rects2, "right")
    # TODO autolabel rects1, rects2
    plt.savefig('results/bar-ie-counts.png')
    plt.clf()
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, scores, width, yerr=stds,
                    color='SkyBlue', label='Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_title('Algorithm comparision')
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    #ax.legend()
    # TODO autolabel rects1, rects2
    plt.savefig('results/bar-ie-acc.png')
    plt.clf()

def plot_matching_triples(sources = ['openie', 'generateie', 'minie']):
    qsets = []
    df = pandas.read_csv('results/comparisions_eng.csv', header=0, delimiter='\t')
    for source in sources:
        df_source = df[df.source == source].apply(lambda x: f'{x["object"]}::{x["relation"]}::{x["subject"]}', axis=1)
        triples = set(df_source.values.tolist())
        qsets.append(triples)
    venn3(qsets, set_labels=sources)
    plt.savefig('results/venn3-ie-quantitative.png')
    plt.clf()

def plot_triple_document_length_correlation(sources = ['openie', 'generateie', 'minie']):
    df = pandas.read_csv('results/comparisions_eng.csv', header=0, delimiter='\t')
    df['len'] = df.apply(lambda x: len(x['text'].split(' ')), axis=1)
    docs = df.groupby(['id', 'len', 'text', 'source'])
    data = docs.count().reset_index()
    #data.plot(x = 'len', y = 'subject')
    print(data)
    ax = data[data.source == 'openie'].plot(x = 'len', y = 'subject', kind='scatter',  color="C2", legend=True, s=1)
    data[data.source == 'generateie'].plot(x = 'len', y = 'subject', ax=ax, kind='scatter', color="C3", legend=True, s=1)
    data[data.source == 'minie'].plot(x = 'len', y = 'subject', ax=ax, kind='scatter', color="C1", legend=True, s=1)
    plt.savefig('results/ie-quantitative-triple-document-ratio.png')
    plt.clf()


def plot_triple_matches(sources = ['openie', 'generateie', 'minie']):
    df = pandas.read_csv('results/comparisions_eng.csv', header=0, delimiter='\t')
    df['triple'] = df.apply(lambda x: f"{x['subject']}::{x['relation']}::{x['object']}", axis=1)
    fig, ax = plt.subplots()
    colors = ['pink', 'green', 'blue', 'purple', 'brown', 'yellow']
    i = 0
    for source in sources:
        alg_df = df[df['source'] == source].merge(df, on='id', how='right', suffixes=('', '_other'))

        #docs = alg_df.groupby(['id'])
        quantities_df = pandas.DataFrame({'id': np.arange(alg_df.id.nunique())})
        quantities_df['relations'] = (~alg_df.sort_values('id').duplicated('relation')).cumsum().groupby(alg_df.id).last().reset_index()[0]
        quantities_df['triples'] = (~alg_df.sort_values('id').duplicated('triple')).cumsum().groupby(alg_df.id).last().reset_index()[0]
        #quantities_df['triples'] = docs['triple'].nunique().cumsum().reset_index()['triple']
        print(source, quantities_df)
        quantities_df.plot(x = 'id', y = 'relations', ax=ax, kind='scatter', label=source[0].upper() + source[1:-2] + 'IE', color=colors[i], legend=True, s=1)
        i = (i + 1) % len(colors)
    plt.savefig('results/ie-quantitative-triple-iter-counts.png')
    plt.clf()
    fig, ax = plt.subplots()
    for j, a1 in enumerate(sources):
        for a2 in sources[j+1:]:
            a1_df = df[df['source'] == a1]
            a2_df = df[df['source'] == a2]
            alg_df = df[df.triple.isin(a1_df.triple) & df.triple.isin(a2_df.triple)].merge(df, on='id', how='right', suffixes=('', '_other'))
            #docs = alg_df.groupby(['id'])
            quantities_df = pandas.DataFrame({'id': np.arange(min(alg_df.id.nunique(), alg_df.id.nunique()))})
            #quantities_df['relations'] = docs['relation'].nunique().cumsum().reset_index()['relation']
            #quantities_df['triples'] = docs['triple'].nunique().cumsum().reset_index()['triple']
            quantities_df['relations'] = (~alg_df.sort_values('id').duplicated('relation')).cumsum().groupby(alg_df.id).last().reset_index()[0]
            quantities_df['triples'] = (~alg_df.sort_values('id').duplicated('triple')).cumsum().groupby(alg_df.id).last().reset_index()[0]
            quantities_df.dropna()
            print(a1, a2, quantities_df)
            quantities_df.plot(x = 'id', y = 'relations', ax=ax, kind='scatter', label=f'{a1[0].upper() + a1[1:-2]}IE/{a2[0].upper() + a2[1:-2]}IE', color=colors[i], legend=True, s=1)
            i = (i + 1) % len(colors)
    plt.savefig('results/ie-quantitative-triple-iter-match.png')
    plt.clf()
    #for id, doc in docs:
    #    print(doc['subject'].nunique())
    #data = docs.count().reset_index()
    #data.plot(x = 'len', y = 'subject')
    #ax = data[data.source == 'openie'].plot(x = 'len', y = 'subject', kind='scatter',  color="C2", legend=True, s=1)
    #data[data.source == 'generateie'].plot(x = 'len', y = 'subject', ax=ax, kind='scatter', color="C3", legend=True, s=1)
    #data[data.source == 'minie'].plot(x = 'len', y = 'subject', ax=ax, kind='scatter', color="C1", legend=True, s=1)
    #plt.savefig('results/ie-quantitative-triple-document-ratio.png')
    #plt.clf()



if __name__ == '__main__':
    #plot_venn_triples()
    plot_bar_acc()
    plot_triple_matches()
    #plot_triple_document_length_correlation()
    #plot_matching_triples()