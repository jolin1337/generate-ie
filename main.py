""" Information extraction on sample sentences """
import time
import pandas
from information_extraction.part_of_speech import set_language
from information_extraction.parser import print_tree
import csv
import collections
from py2neo import Graph, Node, Relationship
from py2neo.ogm import GraphObject, Property, RelatedTo
from information_extraction.sentence_structure import StanfordCoreNLPEx, get_sentence_tree
from information_extraction.triple_evaluators import Metrics

from jnius import autoclass

CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
#AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
MinIE = autoclass('de.uni_mannheim.minie.MinIE')
StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
String = autoclass('java.lang.String')
Properties = autoclass('java.util.Properties')
parser = CoreNLPUtils.StanfordDepNNParser()

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

def parse_relations(nlp, sentence, lang='swe', use_openie=False, use_minie=False, debug=False):
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
    set_language(lang)
    generateie = list(get_relations(nlp, sentence))
    #generateie = [triple for s in tree for triple in get_relations(s[-1], lang=lang)[1]]
    relations += list(zip(['generateie'] * len(generateie), generateie))
    if use_openie:
        openie = [triple for s in nlp._request('openie', sentence)['sentences'] for triple in s['openie']]
        relations += list(zip(['openie'] * len(openie), openie))
    if use_minie:
        model = MinIE(String(sentence), parser, 2)
        model.getPropositions().elements()
        triple_names = ['subject', 'relation', 'object']
        minie = [
            {
                triple_names[i]: val.toString() if val is not None else None
                for i, val in enumerate(ap.getTriple().elements())
                if i < 3
            } for ap in model.getPropositions().elements()
            if ap is not None
        ]
        minie = [triple for triple in minie if all(triple.get(n, None) is not None for n in triple_names)]
        relations += list(zip(['minie'] * len(openie), minie))
    print("parse sentences...")
    for source, relation in relations:
        yield source, relation

def evaluate_comparison(dataset, lang, **kwargs):
    nlp = get_nlp_connection(url='http://corenlp', memory='16g', port=9000)
    all_texts_relations = []
    for i, text in enumerate(load_data(dataset, max_samples=100)):
        data = []
        for source, relation in parse_relations(nlp, text['sentence'], lang, use_openie=True, use_minie=True, debug=kwargs.get('debug')):
            row = {
                'id': text.get('qid', i),
                'source': source,
                'subject': relation['subject'],
                'relation': relation['relation'],
                'object': relation['object'],
                'subjectSpan': f"{relation.get('subjectSpan', [-1, -1])[0]},{relation.get('subjectSpan', [-1, -1])[1]}",
                'relationSpan': f"{relation.get('relationSpan', [-1, -1])[0]},{relation.get('relationSpan', [-1, -1])[1]}",
                'objectSpan': f"{relation.get('objectSpan', [-1, -1])[0]},{relation.get('objectSpan', [-1, -1])[1]}",
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
    nlp = get_nlp_connection(url='http://corenlp', memory='16g', port=9000)
    ie_models = {
        model_name: {
            'metrics': Metrics(),
            'rels': [],
            'name': model_name
        }
        for model_name in ['openie', 'generateie', 'minie']
    }
    match_counters = [(m, Metrics()) for m in ie_models.keys() if m != 'generateie']
    triples = open('results/triples_life_' + lang + '.csv', 'w')
    triples_csv = csv.writer(triples, quoting=csv.QUOTE_NONNUMERIC)
    relations = open('results/relations_life_' + lang + '.csv', 'w')
    relations_csv = csv.writer(relations, quoting=csv.QUOTE_NONNUMERIC)
    entities = open('results/entities_life_' + lang + '.csv', 'w')
    entities_csv = csv.writer(entities, quoting=csv.QUOTE_NONNUMERIC)
    row_columns = [
        *[model + metric for model in ie_models.keys() for metric in [' avg', ' count', ' total']],
        *['match ' + model + metric for model in ie_models.keys() for metric in [' avg', ' count', ' total'] if model != 'generateie']
    ]
    entities_csv.writerow(row_columns)
    relations_csv.writerow(row_columns)
    triples_csv.writerow(row_columns)
    for i, text in enumerate(load_data(dataset, max_samples=10000)):
        for source, relation in parse_relations(nlp, text['sentence'], lang, use_openie=True, use_minie=True):
            ie_models[source]['rels'].append(relation)
            ie_models[source]['metrics'].count(relation)
        for match_model, match_counter in match_counters:
            match_counter.count_if(
                    ie_models[match_model]['rels'],
                    ie_models[source]['rels'],
                    lambda o1, o2, t: o1[t] == o2[t] or o2[t] == o1[t])
                    #lambda o1, o2, t: o1[t + 'Span'] in o2[t + 'Span'])
        row = [
            *[getattr(model['metrics'], metric)() for model in ie_models.values() for metric in ['get_avg', 'get_count', 'get_total']],
            *[getattr(model['metrics'], metric)() for model in ie_models.values() for metric in ['get_avg', 'get_count', 'get_total'] if model['name'] != 'generateie']
        ]
        entities_csv.writerow([c['entity'] for c in row])
        relations_csv.writerow([c['relation'] for c in row])
        triples_csv.writerow([c['relation'] for c in row])
        for name, val in zip(row_columns, row):
            print(name + ":", val)

        if i % 100 == 0:
            print("Saving data...")
            import pickle
            for model in ie_models.values():
                with open(f'results/{model["name"]}_{lang}.pkl', 'wb') as ie:
                    pickle.dump(model['metrics'], ie)
            for match_model, match_counter in match_counters:
                with open(f'results/matchie_{match_model}_{lang}.pkl', 'wb') as ie:
                    pickle.dump(match_counter, ie)

class Entity(GraphObject):
    __primarykey__ = "entity_name"
    __primarylabel__ = "Entity"
    algorithm = Property()
    source = Property()
    entity_name = Property()
    relates = RelatedTo("Entity")
    defined_by = RelatedTo("Article")

class Article(GraphObject):
    __primarykey__ = "id"
    __primarylabel__ = "Article"
    id = Property()
    sentence = Property()

def upload_triples(dataset, lang, **kwargs):
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))
    nlp = get_nlp_connection(url='http://localhost', memory='16g', port=9000)
    dataset_node = Node("Dataset", id=dataset)
    lang_node = Node("Language", id=lang)
    SOURCED = Relationship.type("SOURCED")
    DEFINED_BY = Relationship.type("DEFINED_BY")
    IS_LANG = Relationship.type("IS_LANGUAGE")
    graph.merge(dataset_node, "Dataset", "id")
    graph.merge(lang_node, "Language", "id")
    for i, text in enumerate(load_data(dataset, max_samples=10000)):
        tx = graph.begin()
        article = Node("Article", id=i, sentence=text['sentence'])
        tx.merge(article, "Article", "id")
        tx.merge(IS_LANG(article, lang_node))

        for source, relation in parse_relations(nlp, text['sentence'], lang, use_openie=True):
            algorithm_node = Node("Algorithm", id=source)
            subject = Node("Entity",
                           entity_name=relation['subject'])
            object = Node("Entity",
                           entity_name=relation['object'])
            REL = Relationship.type(relation['relation'])
            tx.merge(algorithm_node, "Algorithm", "id")
            subj_sourced = SOURCED(subject, algorithm_node)
            obj_sourced = SOURCED(object, algorithm_node)
            subj_def = DEFINED_BY(subject, article)
            obj_def = DEFINED_BY(object, article)
            rel = REL(subject, object)
            tx.merge(subj_sourced, "SOURCED", "entity_name")
            tx.merge(obj_sourced, "SOURCED", "entity_name")
            tx.merge(subj_def, "DEFINED_BY", "entity_name")
            tx.merge(obj_def, "DEFINED_BY", "entity_name")
            tx.merge(rel, "REL", "entity_name")
        tx.commit()




def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'method',
        choices=['matches', 'comparison', 'upload_triples'],
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
    methods = {
        'matches': evaluate_matches,
        'comparison': evaluate_comparison,
        'upload_triples': upload_triples
    }
    method = methods.get(args.method, None)
    method(dataset=args.dataset, lang=args.language, debug=args.debug)
