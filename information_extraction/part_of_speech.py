import json
__all__ = ['set_language', 'get_pos_tags']

LANG = 'swe'
POS = None

def set_language(lang):
    global LANG, POS
    LANG = lang
    POS = None


def get_pos_tags(lang=None):
    global POS, LANG
    if lang is None:
        lang = LANG
    if POS is not None and lang == LANG:
        return POS
    """ Load predefined pos tag mappings """
    with open('pos-tags.json', 'r') as pos_file:
        print("Loading pos-tags ({})".format(lang))
        pos_tags_file = json.load(pos_file)
        pos_tags = {c['key']: c['value'] for c in pos_tags_file[lang + '_pos']}
        POS = pos_tags
        return pos_tags

def get_pos(pos_tag, default=None, lang=None):
    pos = get_pos_tags(lang)
    return pos.get(pos_tag, default)