import json
import pandas as pd
import re
import main

nlp = main.get_nlp_connection(url='http://localhost', memory='16g', port=9000)

def get_offset_from_word(text, word_idx):
    global nlp
    # return word_idx
    _, words = nlp.word_tokenize(text, words=True)
    # words = [c for w in word_tokenize(text) for c in word_tokenize(w)] # re.sub(r'\W', '  ', text).split(' ')
    offset = 0
    for i, w in enumerate(words[:int(word_idx)]):
        offset += len(w)
        if offset + 1 < len(text) and text[offset + 1] == ' ':
            offset += 1
    return offset
    return sum(len(w) + 1 for i, w in enumerate(words[:int(word_idx)]))

docs = pd.read_csv('results/comparisions_eng.csv', delimiter='\t')
output = [{
    'text': list(doc['text'])[0],
    'meta': {
        'id': id,
        'testar': 'foobar'
    },
    'project_type': 'triple-labeling',
    # 'labels': list(set(json.dumps({
    #     'source': src,
    #     'object': {
    #         'id': 'Object',
    #         'label': objSpan.split(',')[0],
    #         'start_offset': get_offset_from_word(text, int(objSpan.split(',')[0])),
    #         'end_offset': get_offset_from_word(text, int(objSpan.split(',')[1])),
    #         'text': obj,
    #         'labelType': 1
    #     },
    #     'relation': {
    #         'id': 'Relation',
    #         'label': relSpan.split(',')[0],
    #         'start_offset': get_offset_from_word(text, int(relSpan.split(',')[0])),
    #         'end_offset': get_offset_from_word(text, int(relSpan.split(',')[1])),
    #         'text': rel,
    #         'labelType': 2
    #     },
    #     'subject': {
    #         'id': 'Subject',
    #         'label': subjSpan.split(',')[0],
    #         'start_offset': get_offset_from_word(text, int(subjSpan.split(',')[0])),
    #         'end_offset': get_offset_from_word(text, int(subjSpan.split(',')[1])),
    #         'text': subj,
    #         'labelType': 1
    #     }
    # }) for src, text, obj, subj, rel, objSpan, subjSpan, relSpan in zip(doc['source'], doc['text'], doc['object'], doc['subject'], doc['relation'], doc['objectSpan'], doc['subjectSpan'], doc['relationSpan'])
    #    if src == 'generateie'))
} for id, doc in docs.groupby('id')]

outfile = open('results/doccano_format.json', 'w')
for line in output:
    outfile.write(json.dumps(line) + '\n')

outfile.close()
