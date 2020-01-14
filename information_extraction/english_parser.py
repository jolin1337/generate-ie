from .parser import get_pos_tags
from information_extraction.parser import split_relations


def add_entity(relations, entity):
    for rel in relations:
        rel.append(entity)


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
        POS = get_pos_tags('eng')
        if parent['dep_rel'] in ['nmod:npmod', 'nmod'] or POS.get(parent['pos'], None) == 'Noun':
            propagation_entities.append({
                'propagation_count': 1,
                'type': 'concat',
                **child
            })
        else:
            # rel[0].append(child)
            add_entity(rel, child)


def manage_nmod_npmod(parent, child, rel, propagation_entities):
    """ If nmod:npmod we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] == 'nmod:npmod':
        # rel[0].append(child)
        add_entity(rel, child)
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
            'propagation_direction': 'right',
            'type': 'concat',
            **child
        })


def manage_cop(parent, child, rel, propagation_entities):
    """ If cop relation is a verb then direct relation to parent """
    POS = get_pos_tags('eng')
    if child['dep_rel'] == 'cop' and POS.get(child['pos'], None) == 'Verb' and child['pos'] == 'VBZ':
        # rel[0].append(child)
        add_entity(rel, child)
        propagation_entities.append({
            'propagation_count': 1,
            'type': 'concat',
            **child
        })


def manage_compound(parent, child, rel, propagation_entities):
    """ If compound or compound:prt we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] in ['compound', 'compound:prt']:
        propagation_entities.append({
            'propagation_count': 1,
            'type': 'concat',
            **child
            })


def manage_ccomp(parent, child, rel, propagation_entities):
    """ If compound or compound:prt we can concatinate it with parent as part of that entity/relation """
    if child['dep_rel'] in ['ccomp']:
        propagation_entities.append({
            'propagation_count': 1,
            'propagation_direction': 'left',
            'type': 'entity',
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


def handle_propagations(prop_entities, rel, child):
    leftovers = []
    for entity in prop_entities[::-1]:
        entity['propagation_count'] -= 1
        if entity['propagation_count'] > 0:
            leftovers.append(entity)
            continue
        if entity['type'] == 'entity' and entity['propagation_direction'] == 'right':
            rel.append([r for r in rel[0] if r['token'] != entity['parent']] + [entity])
            continue
        if entity['type'] == 'entity' and entity['propagation_direction'] == 'left':
            rel.append([entity] + [r for r in rel[0] if r['token'] != entity['parent']])
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


def get_relations(tree, sub_call=False):
    global SUBJS
    POS = get_pos_tags('eng')

    rel = [[tree]]
    if tree['dep_rel'] == 'START_DOC':
        SUBJS = []
        rel = [[]]
    # entities = nlp.ner(sentence)
    # pos_tags = nlp.pos_tag(sentence)
    # print(pos_tags)
    # tokens, token_names = nlp.word_tokenize(sentence, words=True)
    children = tree.get('children', [])
    sub_rels = []
    propagation_entities = []
    for child in children:
        if child['dep_rel'] == 'conj' and POS.get(child['pos'], None) in ['Noun']:
            #rel[0].append(child)
            child['propagation_count'] = 1
            child['propagation_direction'] = 'right'
            child['type'] = 'entity'
            child['dep_rel'] = tree['dep_rel']
            propagation_entities.append(child)
        manage_amod(tree, child, rel, propagation_entities)
        manage_nummod(tree, child, rel, propagation_entities)
        manage_nmod_npmod(tree, child, rel, propagation_entities)
        manage_case(tree, child, rel, propagation_entities)
        manage_cop(tree, child, rel, propagation_entities)
        manage_compound(tree, child, rel, propagation_entities)
        manage_ccomp(tree, child, rel, propagation_entities)
        manage_aux(tree, child, rel, propagation_entities)

        # If nsubj direct relation to parent
        if child['dep_rel'] == 'cop' and POS.get(child['pos'], None) == 'Verb' and child['pos'] != 'VBZ':
            # rel[0].append(child)
            add_entity(rel, child)
        elif child['dep_rel'] == 'nsubj' and tree['dep_rel'] != 'parataxis': # or child['dep_rel'] == 'dobj': # TODO
            new_child = child
            if POS.get(child['pos'], None) in ['Noun', 'Adjective']:
                SUBJS.append(child)
            else:
                new_child = sorted(SUBJS, key=lambda s: s['token'] - child['token'])[-1] if SUBJS else None
            if new_child:
                #rel[0].append(child)
                children = child['children']
                child = new_child.copy()
                child['children'] = children
                add_entity(rel, child)
        elif child['dep_rel'] == 'nmod' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            #rel[0].append(child)
            add_entity(rel, child)
        elif child['dep_rel'] == 'dobj' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            # SUBJS.append(child)
            #rel[0].append(child)
            add_entity(rel, child)
        elif child['dep_rel'] == 'dep' and POS.get(child['pos'], None) in ['Verb']:
            #rel[0].append(child)
            add_entity(rel, child)
        elif child['dep_rel'] == 'appos' and POS.get(child['pos'], None) in ['Noun', 'Adjective']:
            #rel[0].append(child)
            add_entity(rel, child)
            propagation_entities.append({
                'propagation_count': 1,
                'propagation_direction': 'right',
                'type': 'entity',
                **child
            })
        sub_prop_entities, sub_rels_tmp = get_relations(child, sub_call=True)
        sub_rels += sub_rels_tmp
        propagation_entities += handle_propagations(sub_prop_entities, rel, child)

    if tree['dep_rel'] == 'START_DOC':
        handle_propagations(propagation_entities, rel, tree)
    subjs = [r for r in rel[0] if r['dep_rel'] == 'nsubj']
    if not subjs and tree['dep_rel'] != 'parataxis':
        closest_subj = sorted(SUBJS, key=lambda s: s['token'] - tree['token'])[-1] if SUBJS else None
        if closest_subj:
            # rel[0].append(closest_subj)
            add_entity(rel, closest_subj)
    #from information_extraction.parser import print_tree, merge_entities
    # if tree['dep_rel'] == 'ROOT':
    #if tree['dep_rel'] == 'START_DOC':
    #     print("Child")
    #     print_tree(tree, lang='eng')
    #print([','.join([merge_entities([e] + list(e.get('sub_info',{}).values())).get('token_name', 'NNN') for e in r]) for r in rel ])
    all_relations = sub_rels + rel
    if sub_call is False:
        all_relations = split_relations(all_relations, lang='eng')
    return propagation_entities, all_relations
