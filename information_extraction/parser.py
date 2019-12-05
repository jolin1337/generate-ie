""" Generall functions needed to parse a sentence """
import json


def get_pos_tags():
    """ Load predefined pos tag mappings """
    with open('pos-tags.json', 'r') as pos_file:
        pos_tags_file = json.load(pos_file)
        pos_tags = {c['key']: c['value'] for c in pos_tags_file['pos']}
        return pos_tags


POS = None


def print_tree(tree, depth=0):
    """ Print dependency tree"""
    global POS
    if POS is None:
        POS = get_pos_tags()

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
        print_tree(child, depth + 1)
