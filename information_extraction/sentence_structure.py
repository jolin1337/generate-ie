from stanfordcorenlp import StanfordCoreNLP
from deeppavlov import configs, build_model
from collections import defaultdict
import re


class StanfordCoreNLPEx(StanfordCoreNLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

    #def ner(self, sentence):
    #    return [(i, entity)
    #            for i, entity in enumerate(self.ner_model([sentence])[1][0])]

    def word_tokenize(self, sentence, words=False, span=False):
        r_dict = self._request('ssplit,tokenize', sentence)
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
        r_dict = self._request('pos,parse', sentence)
        return [s['parse'] for s in r_dict['sentences']]

    def corefs(self, sentence):
        r_dict = self._request('ssplit,tokenize,coref', sentence)
        tokens = [[(token['index'], token['originalText'])
                   for token in s['tokens']]
                   for s in r_dict['sentences']]

        for sent_idx, sent_tokens in enumerate(tokens):
            sentNum = sent_idx + 1
            for index, word in sent_tokens:
                coref = None
                for s in r_dict['corefs'].values():
                    for ref in s:
                        if ref['headIndex'] == index and ref['sentNum'] == sentNum and ref['isRepresentativeMention'] is False:
                            repr_ref = [r for r in s if r['isRepresentativeMention'] is True][0]
                            coref = {
                                'refWord': tokens[repr_ref['sentNum']-1][repr_ref['headIndex']-1][1],
                                **ref
                            }
                yield index, coref, word

    def _check_args(self):
        self._check_language(self.lang)
        if not re.match('\\d+g', self.memory):
            raise ValueError('memory=' + self.memory + ' not supported. Use 4g, 6g, 8g and etc. ')


def get_sentence_tree(nlp, sentence, use_alias=False):
    """ Parse stanford corenlp api dependency tree """
    tokens, token_names = nlp.word_tokenize(sentence, words=True)
    _, entities = (0,['O'] * len(tokens))
    if use_alias:
        _, entities = zip(*nlp.ner(sentence))
        #print("Entityies", entities)
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
    # print("Parsed sentence", sentence)
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

