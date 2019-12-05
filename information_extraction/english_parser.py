from .parser import get_pos_tags
import numpy as np


def manage_amod(parent, child, rel, propagation_entities):
    """ If amod we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'amod':
        propagation_entities.append({
            'propagation_count': 1,
            'type': 'concat',
            **child
        })


def manage_nummod(parent, child, rel, propagation_entities):
    """ If nummod we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'nummod':
        if parent['dep_rel'] == 'nmod:npmod' or parent['dep_rel'] == 'nmod':
            propagation_entities.append({
                'propagation_count': 1,
                'type': 'concat',
                **child
            })
        else:
            rel[0].append(child)


def manage_nmod_npmod(parent, child, rel, propagation_entities):
    """ If nmod:npmod we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'nmod:npmod':
        rel[0].append(child)
        # propagation_entities.append({
        #     'propagation_count': 1,
        #     'propagation_direction': 'left',
        #     'type': 'concat',
        #     **child
        # })


def manage_case(parent, child, rel, propagation_entities):
    """ If case we can concatinate it with parents parent as part of that entity/relation """
    if child['dep_rel'] == 'case':
        propagation_entities.append({
            'propagation_count': 2,
            'type': 'concat',
            **child
        })


def manage_cop(parent, child, rel, propagation_entities):
    """ If cop relation is a verb then direct relation to parent """
    if child['dep_rel'] == 'cop' and POS.get(child['pos'], None) == 'Verb' and child['pos'] == 'VBZ':
        rel[0].append(child)
        # propagation_entities.append({
        #     'propagation_count': 1,
        #     'propagation_direction': 'left',
        #     'type': 'concat',
        #     **child
        # })


def manage_compound(parent, child, rel, propagation_entities):
    """ If compound or compound:prt we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'compound' or child['dep_rel'] == 'compound:prt':
        propagation_entities.append({
            'propagation_count': 1,
            'type': 'concat',
            **child
            })


def manage_aux(parent, child, rel, propagation_entities):
    """ If aux we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'aux' and child['pos'] == 'MD':
        propagation_entities.append({
            'propagation_count': 1,
            'type': 'concat',
            **child
        })


POS = None
SUBJS = []


def get_relations(tree):
    global SUBJS, POS
    if POS is None:
        POS = get_pos_tags()

    if tree['dep_rel'] == 'ROOT':
        SUBJS = []
    # entities = nlp.ner(sentence)
    # pos_tags = nlp.pos_tag(sentence)
    # print(pos_tags)
    # tokens, token_names = nlp.word_tokenize(sentence, words=True)
    children = tree.get('children', [])
    rel = [[tree]]
    sub_rels = []
    propagation_entities = []
    for child in children:
        manage_amod(tree, child, rel, propagation_entities)
        manage_nummod(tree, child, rel, propagation_entities)
        manage_nmod_npmod(tree, child, rel, propagation_entities)
        manage_case(tree, child, rel, propagation_entities)
        manage_cop(tree, child, rel, propagation_entities)
        manage_compound(tree, child, rel, propagation_entities)
        manage_aux(tree, child, rel, propagation_entities)

        sub_prop_entities, sub_rels_tmp = get_relations(child)
        sub_rels += sub_rels_tmp
        for entity in sub_prop_entities[::-1]:
            entity['propagation_count'] -= 1
            if entity['propagation_count'] > 0:
                propagation_entities.append(entity)
                continue
            if entity['type'] == 'entity' and entity['propagation_direction'] == 'right':
                rel.append(rel[0] + [entity])
                continue
            if entity['type'] == 'entity' and entity['propagation_direction'] == 'left':
                rel.append([entity] + rel[0])
                continue
            if entity['token'] > child['token']:
                child['token_name'] += ' ' + entity['token_name']
                if child.get('sub_info', None) is None:
                    child['sub_info'] = {}
                child['sub_info'][entity['token']] = entity
                continue
            child['token_name'] = entity['token_name'] + ' ' + child['token_name']
            if child.get('sub_info', None) is None:
                child['sub_info'] = {}
            child['sub_info'][entity['token']] = entity

        # If nsubj direct relation to parent
        if child['dep_rel'] == 'cop' and POS.get(child['pos'], None) == 'Verb' and child['pos'] != 'VBZ':
            rel[0].append(child)
        elif child['dep_rel'] == 'nsubj':
            if POS.get(child['pos'], None) in ['Noun', 'Adjective']:
                SUBJS.append(child)
            else:
                child = sorted(SUBJS, key=lambda s: s['token'] - child['token'])[-1] if SUBJS else None
            if child:
                rel[0].append(child)
        elif child['dep_rel'] == 'nmod' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            rel[0].append(child)
        elif child['dep_rel'] == 'dobj' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            # SUBJS.append(child)
            rel[0].append(child)
        elif child['dep_rel'] == 'dep' and POS.get(child['pos'], None) in ['Verb']:
            rel[0].append(child)
        elif child['dep_rel'] == 'conj' and POS.get(child['pos'], None) in ['Noun']:
            #rel[0].append(child)
            propagation_entities.append({
                'propagation_count': 1,
                'propagation_direction': 'right',
                'type': 'entity',
                **child
            })
        elif child['dep_rel'] == 'appos' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            rel[0].append(child)
            propagation_entities.append({
                'propagation_count': 1,
                'propagation_direction': 'right',
                'type': 'entity',
                **child
            })

    split_relations = []
    for relation in rel:
        first = relation[:2]
        for r in relation[2:]:
            subject = relation = object = None
            r_triple = sorted(first + [r], key=lambda e: e['token'])
            if POS.get(r_triple[0]['pos']) == 'Verb' or any([POS.get(e['pos']) == 'Verb' for e in
                r_triple[0].get('sub_info', {}).values()]):
                relation = r_triple[0]
                subject = r_triple[1]
                object = r_triple[2]
            else:
                subject = r_triple[0]
                if POS.get(r_triple[1]['pos']) == 'Verb' or any([POS.get(e['pos']) == 'Verb' for e in r_triple[1].get('sub_info', {}).values()]):
                    relation = r_triple[1]
                    object = r_triple[2]
                elif POS.get(r_triple[2]['pos']) == 'Verb' or any([POS.get(e['pos']) == 'Verb' for e in r_triple[2].get('sub_info', {}).values()]):
                    relation = r_triple[2]
                    object = r_triple[1]
            if None not in [subject, relation, object]:
                subject_merged = merge_entities([subject] + list(subject.get('sub_info', {}).values()))
                relation_merged = merge_entities([relation] + list(relation.get('sub_info', {}).values()))
                object_merged = merge_entities([object] + list(object.get('sub_info', {}).values()))
                if object.get('sub_info', None) is not None:
                    print(object)
                split_relations.append({
                    'subject': subject_merged['token_name'],
                    'subjectSpan': subject_merged['token_span'],
                    #[
                    #    min([subject['token']] + [s['token'] for s in subject.get('sub_info', {}).values()]),
                    #    max([subject['token']] + [s['token'] for s in subject.get('sub_info', {}).values()]) + 1
                    #],
                    'relation': relation_merged['token_name'],
                    'relationSpan': relation_merged['token_span'],
                    #[
                    #    min([relation['token']] + [s['token'] for s in relation.get('sub_info', {}).values()]),
                    #    max([relation['token']] + [s['token'] for s in relation.get('sub_info', {}).values()]) + 1
                    #],
                    'object': object_merged['token_name'],
                    'objectSpan': object_merged['token_span']
                    #[
                    #    min([object['token']] + [s['token'] for s in object.get('sub_info', {}).values()]),
                    #    max([object['token']] + [s['token'] for s in object.get('sub_info', {}).values()]) + 1
                    #]
                })

    # split_relations = [r for r in split_relations if len(r) >= 3]
    return propagation_entities, sub_rels + split_relations


def merge_entities(entities):
    sorted_entities = sorted(entities, key=lambda e: -e['token'])
    max_token = sorted_entities[0]['token']
    res_entity = {}
    for entity in sorted_entities:
        res_entity = {
            'token_name': entity['token_name'] + ' ' + res_entity.get('token_name', ''),
            'token_span': [
                min(entity['token'], res_entity.get('token', max_token)),
                max(entity['token'], res_entity.get('token', -1)) + 1
            ],
            **entity,
            **res_entity
        }
    return res_entity
