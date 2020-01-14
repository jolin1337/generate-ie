""" Generall functions needed to parse a sentence """
import json


def get_pos_tags(lang='swe'):
    global POS, language
    if POS is not None and lang == language:
        return POS

    """ Load predefined pos tag mappings """
    with open('pos-tags.json', 'r') as pos_file:
        print("Loading pos-tags ({})".format(lang))
        pos_tags_file = json.load(pos_file)
        pos_tags = {c['key']: c['value'] for c in pos_tags_file[lang + '_pos']}
        POS = pos_tags
        language = lang
        return pos_tags


language='swe'
POS = None


def print_tree(tree, depth=0, lang='swe'):
    """ Print dependency tree"""
    POS = get_pos_tags(lang)
    # if depth > 10: return
    colors = {
        'punct': '1;30',  # GREEN
        'cc': '1;30',  # GREEN
        'ROOT': '0;31',  # RED
        'nsubj': '0;32',  # GREEN
        'cop': '0;33',  # ORANGE
        'acl': '1;33',  # ORANGE
        'aux': '1;33',  # ORANGE
        'det': '0;34',  # BLUE
        'dep': '0;34',  # BLUE
        'amod': '0;35',  # PINK
        'nmod': '0;35',  # PINK
        'advmod': '0;35',  # PINK
        'appos': '0;36',  # BLUE
        'dobj': '1;36',  # LIGHT_BLUE
        'nummod': '1;36',  # LIGHT_BLUE
        'conj': '1;33',  # YELLOW
        'case': '1;33',  # YELLOW
        'compound': '1;33',  # YELLOW
        'NO_COLOR': '0'
    }
    colors['NN'] = '0;31'
    colors['NNP'] = '0;31'
    colors['NNS'] = '0;31'
    colors['VB'] = '0;32'
    colors['VBG'] = '0;32'
    colors['VBZ'] = '0;32'
    colors['JJ'] = '0;33'
    colors['IN'] = '0;34'
    colors['TO'] = '0;34'
    colors['DT'] = '0;35'
    colors['MD'] = '0;36'
    colors['WDT'] = '0;37'
    pos_class = POS.get(tree['pos'])
    for k, value in colors.items():
        colors[k] = f'\033[{value}m'
    color_pos = colors.get(tree["pos"], colors['NO_COLOR'])
    color_dep_rel = colors.get(tree["dep_rel"], colors['NO_COLOR'])
    padding = depth * '--|-' + '->'
    print(f'{tree["token"]:3} |{padding} {tree["token_name"]} => {color_dep_rel}{tree["dep_rel"]} {color_pos}{tree["pos"]}{colors["NO_COLOR"]} {pos_class}')
    children = tree.get('children', [])
    # children.reverse()
    for child in children:
        print_tree(child, depth + 1, lang=lang)



def relation_score_of(relation):
    if POS.get(relation['pos']) == 'Verb':
        return 10
    if 'subj' in relation['dep_rel']:
        return 9
    if 'obj' in relation['dep_rel']:
        return 8
    if POS.get(relation['pos']) == 'Noun':
        return 7
    if POS.get(relation['pos']) == 'Adverb':
        return 6
    if POS.get(relation['pos']) == 'Adjective':
        return 5
    return 0

def split_relations(rel, lang='swe'):
    POS = get_pos_tags(lang)
    split_relations = []
    relations = [sorted(relation, key=lambda r: relation_score_of(r), reverse=True) for relation in rel]
    #print('\n'.join([','.join([merge_entities([e] + list(e.get('sub_info',{}).values())).get('token_name', 'NNN') for e in r]) for r in relations ]))
    for relation in relations:
        first = relation[:2]
        for r in relation[2:]:
            subject = relation = object = None
            r_triple = first + [r] # sorted(first + [r], key=lambda e: e['token'])
            if POS.get(r_triple[0]['pos']) == 'Verb' or any([POS.get(e['pos']) == 'Verb' for e in
                r_triple[0].get('sub_info', {}).values()]):
                relation = r_triple[0]
                subject = r_triple[1]
                object = r_triple[2]
            else:
                subject = r_triple[0]
                if POS.get(r_triple[1]['pos']) == 'Verb' or r_triple[1]['dep_rel'] == 'case' or any([POS.get(e['pos']) == 'Verb' or e['dep_rel'] == 'case' for e in r_triple[1].get('sub_info', {}).values()]):
                    relation = r_triple[1]
                    object = r_triple[2]
                elif POS.get(r_triple[2]['pos']) == 'Verb' or r_triple[2]['dep_rel'] == 'case' or any([POS.get(e['pos']) == 'Verb' or e['dep_rel'] == 'case' for e in r_triple[2].get('sub_info', {}).values()]):
                    relation = r_triple[2]
                    object = r_triple[1]
            if None not in [subject, relation, object]:
                subject_sub_info = [{**si, 'has_sub_info': True} for si in subject.get('sub_info', {}).values()]
                relation_sub_info = [{**si, 'has_sub_info': True} for si in relation.get('sub_info', {}).values()]
                object_sub_info = [{**si, 'has_sub_info': True} for si in object.get('sub_info', {}).values()]
                subject_merged = merge_entities([subject] + subject_sub_info, related_entities=[relation, object] + relation_sub_info + object_sub_info)
                relation_merged = merge_entities([relation] + relation_sub_info, related_entities=[subject, object] + subject_sub_info + object_sub_info)
                object_merged = merge_entities([object] + object_sub_info, related_entities=[subject, relation] + subject_sub_info + relation_sub_info)
                relation = {
                    'subject': subject_merged['token_name'],
                    'subjectSpan': subject_merged['token_span'],
                    'relation': relation_merged['token_name'],
                    'relationSpan': relation_merged['token_span'],
                    'object': object_merged['token_name'],
                    'objectSpan': object_merged['token_span']
                }
                if relation not in split_relations:
                    split_relations.append(relation)

    return split_relations


def entity_score_of(entity, sentence_length=100000):
    if entity['dep_rel'] == 'case':
        return sentence_length+1
    return entity['token']


def merge_entities(entities, related_entities=[]):
    sorted_entities = sorted(entities, key=lambda e: entity_score_of(e), reverse=True)
    max_token = sorted_entities[0]['token']
    res_entity = {}
    related_tokens = [e['token'] for e in related_entities]
    for entity in sorted_entities:
        if entity['dep_rel'] == 'case' and entity['parent'] not in related_tokens:
            continue
        if entity['token'] in related_tokens and entity.get('has_sub_info', None):
            continue
        token_name = entity['token_name'] + ' ' + res_entity.get('token_name', '')
        if entity['token'] > res_entity.get('token', 0):
            token_name = res_entity.get('token_name', '') + ' ' + entity['token_name']
        res_entity = {
            **res_entity,
            **entity,
            'token_name': token_name.strip(),
            'token_span': [
                min(entity['token'], res_entity.get('token', max_token)),
                max(entity['token'], res_entity.get('token', -1)) + 1
            ]
        }
    return res_entity

