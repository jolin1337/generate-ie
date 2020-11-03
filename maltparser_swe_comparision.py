import pandas
from main import collect_relations, parse_relations
from maltparser import MaltParserEx

maltparser = MaltParserEx()
maltparser.read_conll_to_buffer('data/talbanken-stanford-1.2/talbanken-stanford-train.conll')

data = []
for idx, tags, deps in maltparser:
    sentence = ' '.join(tag[0] for tag in tags)
    relations = parse_relations(
        maltparser,
        sentence,
        'swe',
        use_openie=False,
        use_minie=False,
        debug=False
    )
    for source, relation in relations:
        collect_relations(source, relation, idx, sentence, data)


resulting_pandas = pandas.DataFrame(
    data=data[1:],
    columns=data[0]
)
resulting_pandas.head(4000).to_csv('results/comparisions_swe.csv', sep='\t', encoding='utf-8')
