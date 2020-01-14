""" Information extraction on sample sentences """
import time
from collections import defaultdict
import pandas
from stanfordcorenlp import StanfordCoreNLP
from deeppavlov import configs, build_model
from information_extraction.parser import print_tree
import re
import csv


class StanfordCoreNLPEx(StanfordCoreNLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

    #def ner(self, sentence):
    #    return [(i, entity)
    #            for i, entity in enumerate(self.ner_model([sentence])[1][0])]

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
        if not re.match('\\d+g', self.memory):
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


def get_sentence_tree(nlp, sentence, use_alias=False):
    """ Parse stanford corenlp api dependency tree """
    tokens, token_names = nlp.word_tokenize(sentence, words=True)
    _, entities = zip(*nlp.ner(sentence)) # (0,['O'] * len(tokens)) #
    print("Entityies", entities)
    if use_alias:
        alias_entity_bank = {
            'LOCATION': 'Stockholm',
            'B-LOCATION': 'Stockholm',
            'I-LOCATION': 'Kommun',
            'PERSON': 'Adam',
            'B-PERSON': 'Adam',
            'I-PERSON': 'Andersson',
            'I-LOCATION': 'Kommun',
            'NUMBER': '1',
            'B-NUMBER': '1',
            'I-NUMBER': '000'
        }
        ignored_entities = [
            'B-DATE', 'I-DATE', 'DATE'
        ]
        alias_entities = []
        alias_sentence = []
        real_sentence = []
        for i, entity in enumerate(entities):
            if entity == 'O' or entity in ignored_entities:
                alias_entities.append(entity)
                alias_sentence.append(token_names[i])
                real_sentence.append(token_names[i])
            elif entity.startswith('B-') or '-' not in entity:
                alias_entities.append(entity)
                alias_sentence.append(token_names[i]) # alias_entity_bank.get(entity, 'object'))
                real_sentence.append(token_names[i])
            else:
                real_sentence[-1] += ' ' + token_names[i]
        tokens, _ = nlp.word_tokenize(' '.join(alias_sentence), words=True)
        token_names = real_sentence
        entities = alias_entities
        sentence = ' '.join(alias_sentence)
    print("Parsed sentence", sentence)
    tree = nlp.dependency_parse(sentence)
    pos_tags = nlp.pos_tag(sentence)
    dep_tree = []
    for i, node in enumerate(tree):
        passed_roots = [j for j, t in enumerate(tree) if t[0] == 'ROOT' and j <= i]
        sentence_offset = max(passed_roots)
        name = token_names[tokens.index(node[2], sentence_offset)] if node[2] > 0 else '_'
        node_dict = {
            'dep_rel': node[0],
            'token': tokens.index(node[2], sentence_offset) if name != 'ROOT' else 0,
            'parent_token': tokens.index(node[1], sentence_offset) if node[1] > 0 else -1,
            'pos': pos_tags[tokens.index(node[2], sentence_offset)][1],
            'token_name': name,
            'entity': entities[tokens.index(node[2], sentence_offset)],
            'children': []
        }
        if node_dict['dep_rel'] == 'ROOT':
            dep_tree.append(defaultdict(lambda: defaultdict(list)))
            dep_tree[-1][-1] = {
                'children': [],
                'dep_rel': 'START_DOC',
                'token_name': 'START_DOC',
                'pos': '', 'token': -1
            }
        node_dict['children'] = dep_tree[-1][node_dict['token']]['children']
        dep_tree[-1][node_dict['token']] = node_dict
        dep_tree[-1][node_dict['parent_token']]['children'].append(dep_tree[-1][node_dict['token']])
        node_dict['parent'] = dep_tree[-1][node_dict['parent_token']]
    return dep_tree


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
        if not isinstance(objs, list):
            objs = [objs]
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


def load_data(dataset, max_samples=100):
    print("Loading dataset...")
    texts = pandas.read_csv(dataset, header=0)
    # texts = pandas.read_csv('data/example_swe.csv', header=0)
    # texts = pandas.read_csv('data/movie_dataset/summary-sv.csv', header=0)
    if max_samples <= 0:
        max_samples = len(texts)
    texts = texts.sample(n=min(max_samples, len(texts)), random_state=0)
    print(texts)
    texts = texts.iterrows()
    # texts = load_data('data/example_eng.csv', start=0, end=600).iterrows()
    #texts = load_data('data/example_swe.csv', start=0, end=600).iterrows()
    for _, text in texts:
        summary = text['summary']
        if summary.strip()[0] == '#':
            continue
        if '#' in summary or ':' in summary:
            continue
        # print('_____________________________')
        # print(text.get('qid', i), ":", summary)
        # print('#---------------------------#')
        # print()
        sentences = summary.split('\n\n\n')[0]
        sentence = ', '.join([s for s in sentences.split('\n') if s != ''])
        text['sentence'] = sentence
        yield text

def parse_relations(nlp, sentence, lang='swe', use_openie=False, debug=False):
    #if lang == 'swe':
    from information_extraction.swedish_parser import get_relations
    #else:
    #    from information_extraction.english_parser import get_relations
    relations = []
    tree = get_sentence_tree(nlp, sentence)
    if debug:
        print("Dependency tree for", "'{sentence}'".format(sentence=sentence))
        for sent_tree in tree:
            print_tree(sent_tree[-1])
    generateie = get_relations(nlp, sentence, lang=lang)
    #generateie = [triple for s in tree for triple in get_relations(s[-1], lang=lang)[1]]
    relations += list(zip(['generateie'] * len(generateie), generateie))
    if use_openie:
        openie = [triple for s in nlp._request(nlp.url, 'openie', sentence)['sentences'] for triple in s['openie']]
        relations += list(zip(['openie'] * len(openie), openie))
    print("parse sentences...")
    for source, relation in relations:
        yield source, relation

def evaluate_comparison(dataset, lang, **kwargs):
    nlp = get_nlp_connection(url='http://localhost', memory='16g', port=9000)
    all_texts_relations = []
    for i, text in enumerate(load_data(dataset, max_samples=100)):
        data = []
        for source, relation in parse_relations(nlp, text['sentence'], lang, use_openie=True, debug=kwargs.get('debug')):
            row = {
                'id': text.get('qid', i),
                'source': source,
                'subject': relation['subject'],
                'relation': relation['relation'],
                'object': relation['object'],
                'text': text['sentence']
            }
            if len(data) == 0:
                data.append(list(row.keys()))
            data.append(list(row.values()))
            # print('{:40} <=> {:^40} <=> {:>40} :: {}'.format(
            #     relation['subject'].strip(),
            #     relation['relation'].strip(),
            #     relation['object'].strip(),
            #     summary[:100]))
        if data:
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                resulting_pandas = pandas.DataFrame(
                                        data=data[1:],
                                        columns=data[0])
                resulting_pandas.to_csv('results/comparisions/{lang}_{ID}.csv'.format(lang=lang, ID=text.get('qid', i)), sep='\t', encoding='utf-8')
                # print("Sentence:", text.get('qid', i))
                print(resulting_pandas)
            if len(all_texts_relations) == 0:
                all_texts_relations.append(data[0])
            all_texts_relations += data[1:]
        else:
            print("No relations found for", text['sentence'])
        print()
    resulting_pandas = pandas.DataFrame(
                            data=all_texts_relations[1:],
                            columns=all_texts_relations[0])
    resulting_pandas.to_html('results/comparisions_' + lang + '.html')
    resulting_pandas.to_csv('results/comparisions_' + lang + '.csv', sep='\t', encoding='utf-8')


def evaluate_matches(dataset, lang, **kwargs):
    """ Evaluate model agains openie """
    nlp = get_nlp_connection(url='http://localhost', memory='16g', port=9000)
    openie_counter = Metrics()
    generateie_counter = Metrics()
    match_counter = Metrics()
    relations = open('results/relations_life_' + lang + '.csv', 'w')
    relations_csv = csv.writer(relations, quoting=csv.QUOTE_NONNUMERIC)
    entities = open('results/entities_life_' + lang + '.csv', 'w')
    entities_csv = csv.writer(entities, quoting=csv.QUOTE_NONNUMERIC)
    row_columns = [
        "openie avg",
        "openie count",
        "openie total",
        "generateie avg",
        "generateie count",
        "generateie total",
        "match avg",
        "match count",
        "match total"
    ]
    entities_csv.writerow(row_columns)
    relations_csv.writerow(row_columns)
    for text in load_data(dataset, max_samples=100):
        oie_rels = []
        gie_rels = []
        for source, relation in parse_relations(nlp, text['sentence'], lang, use_openie=True):
            if source == 'openie':
                openie_counter.count(relation)
                oie_rels.append(relation)
            else:
                generateie_counter.count(relation)
                gie_rels.append(relation)
        match_counter.count_if(
                oie_rels,
                gie_rels,
                lambda o1, o2, t: o1[t] in o2[t] or o2[t] in o1[t])
                #lambda o1, o2, t: o1[t + 'Span'] in o2[t + 'Span'])
        row = [
            openie_counter.get_avg(),
            openie_counter.get_count(),
            openie_counter.get_total(),
            generateie_counter.get_avg(),
            generateie_counter.get_count(),
            generateie_counter.get_total(),
            match_counter.get_avg(),
            match_counter.get_count(),
            match_counter.get_total()
        ]
        entities_csv.writerow([c['entity'] for c in row])
        relations_csv.writerow([c['relation'] for c in row])
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
    with open('results/generateie_' + lang + '.pkl', 'wb') as ie:
        pickle.dump(generateie_counter, ie)
    with open('results/openie_' + lang + '.pkl', 'wb') as ie:
        pickle.dump(openie_counter, ie)
    with open('results/matchie_' + lang + '.pkl', 'wb') as ie:
        pickle.dump(match_counter, ie)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'method',
        choices=['matches', 'comparison'],
        help="matches or comparison")
    parser.add_argument(
        '--dataset',
        default="data/example_{language}.csv",
        help="File to csv dataset file (must have a summary column in it)" +
             ", default (data/example_{language}.csv)")
    parser.add_argument(
        '--language',
        default="swe",
        help="Three lettters of the language to process (swe or eng)")
    parser.add_argument(
        '--debug',
        action="store_true",
        help="Print detailed information about each sentence")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.dataset = args.dataset.format(language=args.language)
    import sys
    methods = {'matches': evaluate_matches, 'comparison': evaluate_comparison}
    method = methods.get(args.method, None)
    method(dataset=args.dataset, lang=args.language, debug=args.debug)
