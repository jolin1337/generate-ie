# Read more about dep relations in swedish here: https://cl.lingfil.uu.se/~nivre/swedish_treebank/
# Source: https://cl.lingfil.uu.se/~nivre/swedish_treebank/dep.html
# Source2: https://universaldependencies.org/sv/dep/
tree_to_uud = {
  '++': 'cc', # Coordinating conjunction
  '+A': 'sconj', # Should be sconj but it's non existing?
  '+F': 'conj', # Coordination at main clause level
  'AA': 'obj', # Other adverbial
  'AG': 'nmod:agent',
  'AN': '', # Apposition
  'SS': 'nsubj',
  'IF': 'mark',
  'AT': 'amod',
  'ET': 'advmod',
  'VG': 'parataxis', # Verb group
  'OO': 'obj',
  'PA': 'case',
  'TA': 'nmod',
  'CJ': 'conj', # Conjunct (in coordinate structure)
  'SP': 'ccomp',
  'ROOT': 'ROOT'
}