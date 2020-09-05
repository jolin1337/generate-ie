from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_text
from sklearn.model_selection import cross_val_score, train_test_split

from main import parse_relations, get_nlp_connection, print_tree, get_sentence_tree

dataset = pd.read_csv('data/movie_dataset/evaluated_triples.csv').dropna()

nlp = get_nlp_connection(url='http://corenlp', memory='16g', port=9000)

pos_tokens = {}
rel_type_tokens = {}
def get_feature_vector(example):
  sentence = example['text']
  #print_tree(get_sentence_tree(nlp, sentence)[0][-1])
  tokens, token_names = nlp.word_tokenize(sentence, words=True)
  tree = nlp.dependency_parse(sentence)
  pos_tags = nlp.pos_tag(sentence)
  mapped = []
  for i, node in enumerate(tree):
    passed_roots = [j for j, n in enumerate(tree) if n[0] == 'ROOT' and j <= i]
    sentence_offset = max(passed_roots)
    pos = pos_tags[tokens.index(node[2], sentence_offset)][1]
    rel_type = node[0]

    if pos not in pos_tokens:
      pos_tokens[pos] = len(pos_tokens)
    pos_token = pos_tokens[pos]

    if rel_type not in rel_type_tokens:
      rel_type_tokens[rel_type] = len(rel_type_tokens)
    rel_type_token = rel_type_tokens[rel_type]

    node_vec = [rel_type_token, node[2], pos_token]
    mapped += node_vec
  return np.array(mapped)[:9]

dataset['feature_vector'] = dataset.apply(get_feature_vector, axis=1)
dataset['rel_type'] = dataset.apply(lambda x: x['feature_vector'][0::3], axis=1)
dataset['node'] = dataset.apply(lambda x: x['feature_vector'][1::3], axis=1)
dataset['pos'] = dataset.apply(lambda x: x['feature_vector'][2::3], axis=1)
dataset['target'] = dataset.apply(lambda d: np.array([int(d['Valid'] == 1), int(d['Valid'] == 2), 0]), axis=1)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, min_samples_leaf=1)
#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
features = list(dataset.feature_vector)
X_train, X_test, y_train, y_test = train_test_split(features, list(dataset.target), test_size=0.33, random_state=42)
dt.fit(X_train, y_train)
feature_names = ['rel_type', 'node', 'pos'] * int(len(features[0]) / 3)
feature_names = [f'{fn}_{i}' for i, fn in enumerate(feature_names)]
tree_rules = export_text(dt, feature_names=feature_names)
print(tree_rules)
print(dt.predict_proba(X_test))
# print(cross_val_score(dt, features, list(dataset.target), cv=10)))
