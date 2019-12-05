""" Information extraction on sample sentences """
import time
from collections import defaultdict
import pandas
from stanfordcorenlp import StanfordCoreNLP
from information_extraction.english_parser import get_relations
# from information_extraction.swedish_parser import get_relations
from information_extraction.parser import print_tree
import re


class StanfordCoreNLPEx(StanfordCoreNLP):
    def word_tokenize(self, sentence, words=False, span=False):
        r_dict = self._request(self.url, 'ssplit,tokenize', sentence)
        tokens = ([token['index']
                   for s in r_dict['sentences']
                   for token in s['tokens']],)

        # Whether return token span
        if span:
            spans = [(token['characterOffsetBegin'], token['characterOffsetEnd'])
                     for s in r_dict['sentences']
                     for token in s['tokens']]
            tokens = (*tokens, spans)

        # Whether return token words
        if words:
            word_tokens = [(token['originalText'])
                           for s in r_dict['sentences']
                           for token in s['tokens']]
            tokens = (*tokens, word_tokens)

        if len(tokens) == 1:
            return tokens[0]
        else:
            return tokens

    def parse(self, sentence):
        r_dict = self._request(self.url, 'pos,parse', sentence)
        return [s['parse'] for s in r_dict['sentences']]

    def _check_args(self):
        self._check_language(self.lang)
        if not re.match('\d+g', self.memory):
            raise ValueError('memory=' + self.memory + ' not supported. Use 4g, 6g, 8g and etc. ')


def get_nlp_connection(url=None, memory='2g', port=8001, print_startuptime=True):
    """ Open up a connection to stanford corenlp api """
    if url is None:
        url = './stanford-corenlp-full-2018-10-05.zip'
    nlp = None
    start_time = time.time()
    try:
        nlp = StanfordCoreNLPEx(url, memory=memory, port=port)
    except:
        nlp = StanfordCoreNLPEx('http://localhost', memory=memory, port=port)
    # nlp.switch_language('sv')
    if print_startuptime:
        print('Startuptime:', (time.time() - start_time))
    return nlp


def get_sentence_tree(nlp, sentence):
    """ Parse stanford corenlp api dependency tree """
    entities = nlp.ner(sentence)
    pos_tags = nlp.pos_tag(sentence)
    tokens, token_names = nlp.word_tokenize(sentence, words=True)
    tree = nlp.dependency_parse(sentence)
    dep_tree = []
    for i, node in enumerate(tree):
        passed_roots = [j for j, t in enumerate(
            tree) if t[0] == 'ROOT' and j <= i]
        sentence_offset = max(passed_roots)
        name = token_names[tokens.index(
            node[2], sentence_offset)] if node[2] > 0 else '_'
        node_dict = {
            'dep_rel': node[0],
            'token': tokens.index(node[2], sentence_offset) if name != 'ROOT' else 0,
            'parent': tokens.index(node[1], sentence_offset) if node[1] > 0 else -1,
            'pos': pos_tags[tokens.index(node[2], sentence_offset)][1],
            'token_name': name,
            'entity': entities[tokens.index(node[2], sentence_offset)][1],
            'children': []
        }
        if node_dict['dep_rel'] == 'ROOT':
            dep_tree.append(defaultdict(lambda: defaultdict(list)))
            dep_tree[-1][-1] = {'children': [], 'dep_rel': 'START_DOC',
                                'token_name': 'START_DOC', 'pos': '', 'token': -1}
        node_dict['children'] = dep_tree[-1][node_dict['token']]['children']
        dep_tree[-1][node_dict['token']] = node_dict
        dep_tree[-1][node_dict['parent']
                     ]['children'].append(dep_tree[-1][node_dict['token']])
    return dep_tree

def load_data(f, start=0, end=-1):
    df = pandas.read_csv(f, header=0)
    return df.loc[start:end]

class Counter:
    def __init__(self):
        self.values = []
        self.total = 0

    def count(self, value):
        self.total += 1
        if value in self.values:
            return
        self.values.append(value)

    def count_if(self, value, f):
        if f(value):
            self.count(value)

    def get_avg(self):
        return len(self.values) / max(1, self.total)

    def get_total(self):
        return self.total

    def get_count(self):
        return len(self.values)

class Metrics:
    def __init__(self):
        self.entity_counter = Counter()
        self.relation_counter = Counter()

    def count(self, objs, f=lambda t: True):
        counts = 0
        for obj in objs:
            if f('object'):
                self.entity_counter.count(obj['object'])
                counts += 1
            if f('subject'):
                self.entity_counter.count(obj['subject'])
                counts += 1
            if f('relation'):
                self.relation_counter.count(obj['relation'])
                counts += 1
        return counts

    def count_if(self, objs1, objs2, f):
        matches = []
        for i, obj1 in enumerate(objs1):
            if i in matches:
                continue
            for j, obj2 in enumerate(objs2):
                if j in matches:
                    continue
                if self.count([obj1], f=lambda t: f(obj1, obj2, t)) > 0:
                    matches.append(i)
                    matches.append(j)
                    break

    def get_count(self):
        return {'entity': self.entity_counter.get_count(),
                'relation': self.relation_counter.get_count()}

    def get_total(self):
        return {'entity': self.entity_counter.get_total(),
                'relation': self.relation_counter.get_total()}
    def get_avg(self):
        return {'entity': self.entity_counter.get_avg(),
                'relation': self.relation_counter.get_avg()}


def main():
    """ Main function for this module """
    nlp = get_nlp_connection(url='http://localhost', memory='16g', port=9000)
    print("Loading dataset...")
    texts = pandas.read_csv('data/movie_dataset/summary-en.csv', header=0)
    # texts = pandas.read_csv('data/example_eng.csv', header=0)
    texts = texts.sample(n=min(100, len(texts)), random_state=0)
    print(texts)
    texts = texts.iterrows()
    # texts = load_data('data/example_eng.csv', start=0, end=600).iterrows()
    #texts = load_data('data/example_swe.csv', start=0, end=600).iterrows()
    print('parse sentences...')
    for i, text in texts:
        summary = text['summary']
        if summary.strip()[0] == '#':
            continue
        if '#' in summary or ':' in summary:
            continue
        # print('_____________________________')
        # print(i, ":", summary)
        # print('#---------------------------#')
        # print()
        tstart = time.time()
        sentences = summary.split('\n\n\n')[0]
        sentence = ', '.join([s for s in sentences.split('\n') if s != ''])

        generateie = get_sentence_tree(nlp, sentence)
        for gie_sentence in generateie:
            _, relations = get_relations(gie_sentence[-1])
            # print('---------------------------------------------')
            # print('Relations')
            for relation in relations:
                print('{:40} <=> {:^40} <=> {:>40} :: {}'.format(
                    relation['subject'].strip(),
                    relation['relation'].strip(),
                    relation['object'].strip(),
                    summary[:100]))
            print()
            print("Dependency tree")
            print_tree(gie_sentence[-1])
        # print('Tokenize:', nlp.word_tokenize(sentence, words=True))
        # print('Part of Speech:', nlp.pos_tag(sentence))
        # print('Named Entities:', nlp.ner(sentence))
        # print('Constituency Parsing:', '\n\n'.join(nlp.parse(sentence)))
        #print('Dependency Parsing:', nlp.dependency_parse(sentence))
        #print('Parsing time: ', (time.time() - tstart))
        #print('_____________________________')


def evaluate():
    """ Evaluate model agains openie """
    nlp = get_nlp_connection(url='http://localhost', memory='16g', port=9000)
    print("Loading dataset...")
    #texts = load_data('data/movie_dataset/summary-en.csv',
    #                  start=300, end=1600).iterrows()
    #texts = load_data('data/example_swe.csv', start=0, end=600).iterrows()
    texts = pandas.read_csv('data/movie_dataset/summary-en.csv', header=0)
    texts = texts.sample(n=100)
    texts = texts.iterrows()
    print('parse sentences...')
    openie_counter = Metrics()
    generateie_counter = Metrics()
    match_counter = Metrics()
    for _, text in texts:
        summary = text['summary']
        if summary.strip()[0] == '#':
            continue
        sentences = summary.split('\n\n\n')[0]
        sentence = ', '.join([s for s in sentences.split('\n') if s != ''])

        generateie = get_sentence_tree(nlp, sentence)
        openie = nlp._request(nlp.url, 'openie', sentence)['sentences']
        _, generateie = zip(*[get_relations(tree[-1]) for tree in generateie])
        for oie_sentence, gie_sentence in zip(openie, generateie):
            openie_counter.count(oie_sentence['openie'])
            generateie_counter.count(gie_sentence)
            match_counter.count_if(
                    oie_sentence['openie'],
                    gie_sentence,
                    lambda o1, o2, t: o1[t] in o2[t] or o2[t] in o1[t])
                    #lambda o1, o2, t: o1[t + 'Span'] in o2[t + 'Span'])
            print('---------------------------------------------')
            # print('Relations')
            # print('Generated IE:', generateie)
            # print('Open IE:', openie)
            print("Open IE avg:", openie_counter.get_avg())
            print("Open IE count:", openie_counter.get_count())
            print("Open IE total:", openie_counter.get_total())
            print("Generated IE avg:", generateie_counter.get_avg())
            print("Generated IE count:", generateie_counter.get_count())
            print("Generated IE total:", generateie_counter.get_total())
            print("Match IE avg:", match_counter.get_avg())
            print("Match IE count:", match_counter.get_count())
            print("Match IE total:", match_counter.get_total())

    import pickle
    with open('results/generateie.pkl', 'wb') as ie:
        pickle.dump(generateie_counter, ie)
    with open('results/openie.pkl', 'wb') as ie:
        pickle.dump(openie_counter, ie)
    with open('results/matchie.pkl', 'wb') as ie:
        pickle.dump(match_counter, ie)


if __name__ == '__main__':
    main()
    # evaluate()
