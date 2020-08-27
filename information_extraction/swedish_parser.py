from .part_of_speech import get_pos
from .parser import split_relations
from collections import defaultdict
import json
import os

def add_entity(which_relations='all', **kwargs):
    relations = kwargs.get('relations')
    entity = kwargs.get('entity')
    parent = kwargs.get('parent', {})
    if parent.get('dep_rel') == 'punct':
        return

    if get_pos(entity['pos']) in ['Noun', 'Adjective']:
        OBJS.append(entity)
    for i, rel in enumerate(relations):
        if which_relations == 'all' or i in which_relations:
            rel.append(entity)


def add_nsubj_entity(**kwargs):
    new_child = kwargs.get('entity')
    children = new_child['children']
    if get_pos(new_child['pos']) in ['Noun', 'Adjective']:
        SUBJS.append(new_child)
    else:
        new_child = sorted(SUBJS, key=lambda s: s['token'] - new_child['token'])[-1] if SUBJS else None
    if new_child:
        entity = new_child.copy()
        entity['children'] = children
        kwargs['entity'] = entity
        add_entity(**kwargs)


def add_expl_entity(**kwargs):
    new_child = kwargs.get('entity')
    children = new_child['children']
    if get_pos(new_child['pos']) in ['Noun', 'Adjective']:
        OBJS.append(new_child)
    else:
        new_child = sorted(OBJS, key=lambda s: s['token'] - new_child['token'])[-1] if OBJS else None
    if new_child:
        entity = new_child.copy()
        entity['children'] = children
        kwargs['entity'] = entity
        add_entity(**kwargs)


def propagate_entity(count=1, line=True, type='entity', copy_relation=False, **kwargs):
    prop_entities = kwargs.get('propagation_entities')
    entity = kwargs.get('entity')
    parent = kwargs.get('parent')
    entity['propagation_count'] = count
    entity['propagation_direction'] = 'right'
    entity['type'] = type
    if copy_relation:
        entity['dep_rel'] = parent['dep_rel']
    prop_entities.append(entity)
    #     'propagation_count': count,
    #     'line': line,
    #     'type': type,
    #     **entity
    # })
action_method_mapping = {
    'propagate_entity': propagate_entity,
    'add_expl_entity': add_expl_entity,
    'add_nsubj_entity': add_nsubj_entity,
    'add_entity': add_entity
}
curr_dir = os.path.dirname(__file__)
prop_rules = json.load(open(os.path.join(curr_dir, 'prop_rules.json'), 'r'))['rules']
dep_rules = json.load(open(os.path.join(curr_dir, 'rules.json'), 'r'))['rules']


def conditions(conds, **kwargs):
    is_true = True
    entity = kwargs.get('entity')
    parent = kwargs.get('parent')
    if not isinstance(conds, list):
        conds = [conds]
    for condition in conds:
        for attr, value in condition.items():
            if attr == 'or':
                or_true = False
                for second_condition in value:
                    or_true = or_true or conditions(second_condition, **kwargs)
                is_true = is_true and or_true
            elif attr == 'and':
                and_true = True
                for second_condition in value:
                    and_true = and_true and conditions(second_condition, **kwargs)
                is_true = is_true and and_true
            elif attr == 'not':
                is_true = is_true and not conditions(value, **kwargs)
            elif attr == 'parent':
                kwargs_copy = kwargs.copy()
                kwargs_copy['entity'] = parent
                is_true = is_true and conditions(value, **kwargs_copy)
            elif attr == 'pos':
                is_true = is_true and (entity['pos'] == value or get_pos(entity['pos']) == value)
            elif attr in entity:
                is_true = is_true and entity[attr] == value
            else:
                raise Exception("Unkown condition {}=={}".format(attr, value))

    return is_true


def do_actions(actions, **kwargs):
    if not isinstance(actions, list):
        actions = [actions]
    for action in actions:
        action['method'](**action.get('params', {}), **kwargs)


def handle_propagations(prop_entities, rel, child):
    leftovers = []
    for entity in prop_entities[::-1]:
        entity['propagation_count'] -= 1
        if entity['propagation_count'] > 0:
            if get_pos(child['pos']) != 'Verb':
                leftovers.append(entity)
            continue
        if entity['type'] == 'entity' and entity['propagation_direction'] == 'right':
            rel.append([r for r in rel[0] if r['token'] != entity['parent_token']] + [entity])
            continue
        if entity['type'] == 'entity' and entity['propagation_direction'] == 'left':
            rel.append([entity] + [r for r in rel[0] if r['token'] != entity['parent_token']])
            continue

        if entity['token'] > child['token']:
            #child['token_name'] += ' ' + entity['token_name']
            if child.get('sub_info', None) is None:
                child['sub_info'] = {}
            child['sub_info'][entity['token']] = entity
        else:
            #child['token_name'] = entity['token_name'] + ' ' + child['token_name']
            if child.get('sub_info', None) is None:
                child['sub_info'] = {}
            child['sub_info'][entity['token']] = entity
    return leftovers


SUBJS = []
OBJS = []


def is_relation(dep, pos):
    return get_pos(pos) == 'Verb' or dep[0] == 'parataxis'

def is_entity(dep, pos):
    return get_pos(pos) == 'Noun' or \
           dep[0] == 'nsubj' or \
           (dep[0] == 'mark' and get_pos(pos) == 'Conjunction') or \
           dep[0] == 'amod' or \
           dep[0] == 'advmod'

def is_parent(e1, parent, deps, pos):
    e1 = e1[1]
    p = e1[1] - 1
    while p > -1:
        e1 = deps[p]
        if e1[2] == parent[1][2]:
            return True
        if is_relation(deps[p], pos[p]):
            break
        p = e1[1] - 1
    return False


def get_sentence_relations(tree, pos_tags):
    words, pos_tags = zip(*pos_tags)
    token_words = list(words)
    token_pos = list(pos_tags)
    #subtract_with1 = min([dep[1] for dep in tree])
    #subtract_with2 = min([dep[2] for dep in tree]) - 1
    #tree = [[dep[0], dep[1] - subtract_with1, dep[2] - subtract_with2] for dep in tree]
    tree = sorted(tree, key=lambda dep: dep[2])
    disabled = []
    token_conditions = defaultdict(list)
    for word, dep, pos in zip(token_words, tree, token_pos):
        kwargs = {
            'entity': {
                'dep_rel': dep[0],
                'pos': pos
            },
            'parent': {
                # -2 because it is the parent node
                'dep_rel': tree[dep[1] - 1][0] if dep[1] > 0 else -1,
                'pos': token_pos[dep[1] - 1] if dep[1] > 0 else -1
            }
        }
        for rule in prop_rules:
            if not isinstance(rule['actions'], list):
                rule['actions'] = [rule['actions']]
            if not any((action['method'] == propagate_entity or action['method'] == 'propagate_entity') and action.get('params', {}).get('type') == 'concat' for action in rule['actions']):
                continue
            if conditions(rule['conditions'], **kwargs) and dep[1] > 0:
                dep_entity = dep
                rules = []
                for action in rule['actions']:
                    method = action['method']
                    if isinstance(method, str):
                        method = action_method_mapping[method]
                    if method != propagate_entity:
                        continue
                    for _ in range(action.get('params', {}).get('count', 1)):
                        if dep_entity[1] - 1 > 0:
                            dep_entity = tree[dep_entity[1] - 1]
                            rules.append(dep_entity)
                if rules:
                    rules.pop() # Remove the actual node
                token_conditions[dep_entity[2] - 1].append({
                    'rules': rules,
                    'token': dep
                })
                if get_pos(token_pos[dep[2] - 1]) == 'Verb':
                    tmp_dep = list(dep)
                    tmp_dep_entity = list(dep_entity)
                    tree[dep[2] - 1] = (tmp_dep_entity[0], tmp_dep_entity[1], tmp_dep[2])
                    tree[dep_entity[2] - 1] = (tmp_dep[0], tmp_dep[1], tmp_dep_entity[2])
                    for i in range(len(tree)):
                        if tree[i][1] == tmp_dep_entity[2]: # If parent is the dependant
                            tree[i] = (tree[i][0], tmp_dep[2], tree[i][2])

                disabled.append(dep)
    relations = [(idx, dep) for idx, (dep, pos) in enumerate(zip(tree, token_pos)) if dep not in disabled and is_relation(dep, pos)]
    entities = [(idx, dep) for idx, (dep, pos) in enumerate(zip(tree, token_pos)) if dep not in disabled and is_entity(dep, pos)]
    triples = []
    print("relations:",[token_words[r[0]] for r in relations])
    print("entities:", [token_words[e[0]] for e in entities])
    for ridx, rel in relations:
        potential_entities = sorted(entities, key=lambda entity: abs(entity[0] - ridx))
        left_entities = []
        right_entities = []
        for eidx, entity in potential_entities:
            conflicting_relations = sorted(relations, key=lambda rel: abs(rel[0] - eidx))[1:]
            if conflicting_relations:
                if conflicting_relations[0][0] < eidx and conflicting_relations[0][0] > ridx:
                    break
                if conflicting_relations[0][0] > eidx and conflicting_relations[0][0] < ridx:
                    break
            if eidx < ridx:
                left_entities.append((eidx, entity))
            if eidx > ridx:
                right_entities.append((eidx, entity))
        for reidx, re in right_entities:
            for leidx, le in left_entities:
                triple = [(leidx, le), (ridx, rel), (reidx, re)]
                if (is_parent(triple[1], triple[0], tree, token_pos) and is_parent(triple[2], triple[0], tree, token_pos)) or \
                        (is_parent(triple[0], triple[1], tree, token_pos) and is_parent(triple[2], triple[1], tree, token_pos)) or \
                        (is_parent(triple[0], triple[2], tree, token_pos) and is_parent(triple[1], triple[2], tree, token_pos)):
                    cp_token_words = list(token_words)
                    triple_obj = make_triple(triple, cp_token_words)
                    for node, condition in token_conditions.items():
                        dep_entity = tree[node]
                        for rules in sorted(condition, key=lambda k: k['token'][2], reverse=True):
                            dep = rules['token']
                            for rule in rules['rules']:
                                if rule not in [le, rel, re]:
                                    break
                            else:
                                if dep_entity[2] <= dep[2]:
                                    cp_token_words[dep_entity[2] - 1] += ' ' + cp_token_words[dep[2] - 1]
                                else:
                                    cp_token_words[dep_entity[2] - 1] = cp_token_words[dep[2] - 1] + ' ' + cp_token_words[dep_entity[2] - 1]
                                triple_obj = make_triple(triple, cp_token_words)
                                if dep[0] == 'case' and triple_obj not in triples:
                                    triples.append(triple_obj)
                                continue
                            break
                    if triple_obj not in triples:
                        triples.append(triple_obj)
    return triples


def make_triple(triple, token_names):
    return {
        "subject": token_names[triple[0][0]],
        "subjectSpan": [triple[0][0], triple[0][0] + 1],
        "relation": token_names[triple[1][0]],
        "relationSpan": [triple[1][0], triple[1][0] + 1],
        "object": token_names[triple[2][0]],
        "objectSpan": [triple[2][0], triple[2][0] + 1],
    }


def get_relations(nlp, sentence, sub_call=False):
    rel_idx = 0
    pos_tags = nlp.pos_tag(sentence)
    print("Parsed sentence", sentence)
    tree = nlp.dependency_parse(sentence)
    splits = [idx for idx, dep in enumerate(tree) if dep[rel_idx] == 'ROOT'] + [len(tree)]# or dep[rel_idx] == 'parataxis'] + [len(tree)]
    sentences = [tree[splits[i]:splits[i+1]] for i in range(len(splits) - 1)]
    pos_sentences = [pos_tags[splits[i]:splits[i+1]] for i in range(len(splits) - 1)]

    for i, sent, pos in zip(range(len(sentences)), sentences, pos_sentences):
        for triple in get_sentence_relations(sent, pos):
            if i > 0:
                offset = sum(len(s) for s in pos_sentences[:i])
                triple['subjectSpan'][0] += offset
                triple['subjectSpan'][1] += offset
                triple['relationSpan'][0] += offset
                triple['relationSpan'][1] += offset
                triple['objectSpan'][0] += offset
                triple['objectSpan'][1] += offset
            yield triple
    return
    relations = [
        triple
        for sent, pos in zip(sentences, pos_sentences)
            for triple in get_sentence_relations(sent, pos)
    ]
    return relations




def get_relations2(tree, sub_call=False):
    global SUBJS, OBJS

    rel = [[tree]]
    if tree['dep_rel'] == 'START_DOC':
        SUBJS = []
        OBJS = []
        rel = [[]]
    # entities = nlp.ner(sentence)
    # pos_tags = nlp.pos_tag(sentence)
    # print(pos_tags)
    # tokens, token_names = nlp.word_tokenize(sentence, words=True)
    children = tree.get('children', [])
    sub_rels = []
    propagation_entities = []
    for child in children:
        for rule in dep_rules:
            kwargs = {
                'relations': rel,
                'entity': child,
                'propagation_entities': propagation_entities,
                'parent': tree
            }
            if conditions(rule['conditions'], **kwargs) is True:
                do_actions(rule['actions'], **kwargs)
        sub_prop_entities, sub_rels_tmp = get_relations(child, sub_call=True)
        sub_rels += sub_rels_tmp
        propagation_entities += handle_propagations(sub_prop_entities, rel, child)

    if tree['dep_rel'] == 'START_DOC':
        handle_propagations(propagation_entities, rel, tree)
    subjs = [r for r in rel[0] if r['dep_rel'] == 'nsubj']
    #if not subjs and tree['dep_rel'] != 'parataxis':
    #    closest_subj = sorted(SUBJS, key=lambda s: s['token'] - tree['token'])[-1] if SUBJS else None
    #    if closest_subj:
    #        # rel[0].append(closest_subj)
    #        add_entity(relations=rel, entity=closest_subj)
    from information_extraction.parser import print_tree, merge_entities
    # if tree['dep_rel'] == 'ROOT':
    #if tree['dep_rel'] == 'START_DOC':
    #     print("Child")
    #     print_tree(tree)
    # print([','.join([merge_entities([e] + list(e.get('sub_info', {}).values())).get('token_name', 'NNN') for e in r]) for r in rel])
    all_relations = sub_rels + rel
    if sub_call is False:
        all_relations = split_relations(all_relations)
    return propagation_entities, all_relations
