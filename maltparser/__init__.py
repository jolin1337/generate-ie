import os
import tempfile
import subprocess

from nltk.parse.malt import MaltParser
from maltparser.dep_rel_mappings import tree_to_uud


maltdir = os.path.join(os.path.dirname(__file__), '../../maltparser')


class MaltParserEx(MaltParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{
            **kwargs,
            'parser_dirname': maltdir,
            'model_filename': os.path.join(maltdir, 'swemalt-1.7.2.mco'),
        })
        self._tags_buff = {}
        self._meta_buff = {}
        self._dep_buff = []
        #self.ner_model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)

    def __iter__(self):
        for sentence, meta in self._meta_buff.items():
            idx = meta.get('sent_idx')
            dep = self._dep_buff[idx]
            tags = self._tags_buff[sentence]
            yield idx, tags, dep


    #def ner(self, sentence):
    #    return [(i, entity)
    #            for i, entity in enumerate(self.ner_model([sentence])[1][0])]

    def read_conll_to_buffer(self, file, word_idx=1, tag_idx=4, parent_idx=6, dep_idx=7):
        with open(file, 'r') as fh:
            sentences = fh.read().split('\n\n')
            for i, sentence in enumerate(sentences):
                words = [word.split('\t') for word in sentence.split('\n') if len(word) > 0]
                sentence_text = ' '.join(word[word_idx] for word in words)
                tags = [(word[word_idx], word[tag_idx]) for word in words]
                self._tags_buff[sentence_text] = tags
                self._meta_buff[sentence_text] = {
                    'sent_idx': i
                }
                # (dep, parent, node)
                deps = [(word[dep_idx], int(word[parent_idx]), int(word[0])) for word in words]
                deps.sort(key=lambda x: (x[1], 0 if x[0].upper() == 'ROOT' else 1))
                self._dep_buff.append(deps)

    def buffer(self, sentences, meta=None):
        if meta is None:
            meta = [{} for _ in sentences]
        if hasattr(sentences, '__iter__') and not isinstance(sentences, str):
            sentences = '. H §§. '.join(sentences)
        if not isinstance(meta, list):
            meta = [meta]
        # Parse the sentences with malt tokenization
        tag_sents = [self.pos_tag(sentences)]
        meta_idx = 0
        i_offset = len(self._tags_buff)
        for i, tag_sent in enumerate(tag_sents):
            sent_words = [tag[0] for tag in tag_sent]
            if ''.join(sent_words) == 'H§§':
                meta_idx += 1
                continue
            word_key = ' '.join(sent_words)
            self._tags_buff[word_key] = tag_sent
            self._meta_buff[word_key] = {
                'sent_idx': i_offset + i,
                **meta[meta_idx],
                **self._meta_buff.get(word_key, {})
            }
        self._dep_buff += [list(d) for d in self.parse_tagged_sents(self._tags_buff.values(), verbose=True)]

    def _find_sentence_meta(self, sentence, meta=None):
        meta = self._meta_buff.get(sentence)
        if meta is None:
            self.buffer(sentence, meta=meta)
            meta = self._meta_buff.get(sentence)
        return meta

    def word_tokenize(self, sentence, words=False, span=False, meta=None):
        meta = self._find_sentence_meta(sentence, meta)
        tag = self._tags_buff.get(sentence)
        tokens = ([i+1 for i, _ in enumerate(tag)],)
        if words:
            tokens = (*tokens, [w[0] for i, w in enumerate(tag)])
        #if span: #TODO: Need th span returned from pos_tag java call
        #    tokens = (*tokens, [w[2] for i, w in enumerate(tag)])
        return tokens


    def dependency_parse(self, sentence, meta=None):
        meta = self._find_sentence_meta(sentence, meta=meta)
        return self._dep_buff[meta['sent_idx']]
        nodes = dep[0].nodes
        # (dep, parent, node)
        return [(
            tree_to_uud.get(node['rel'], node['rel']),
            node['head'] if node['head'] is not None else None,
            node['address']
        ) for idx, node in nodes.items() if node['head'] is not None]

    def _check_args(self):
        pass

    def pos_tag(self, sentence):
        if self._tags_buff.get(sentence) is not None:
            return self._tags_buff.get(sentence)
        pos_tags = []
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".txt") as tmpfile:
            tmpfile.write(sentence.encode('utf-8'))
            tmpfile.flush()
            staggerjar = os.path.join(maltdir, 'stagger.jar')
            staggermodel = os.path.abspath(os.path.join(maltdir, 'swedish.bin'))
            cmd = ['java','-jar', staggerjar, '-modelfile', staggermodel, '-tag', tmpfile.name]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = p.communicate(b"input data that is passed to subprocess' stdin")
            rc = p.returncode
            for sub_sent in output.decode('utf-8').split('\n\n'):
                words = sub_sent.split('\n')
                pos = []
                if len(words) <= 1:
                    continue
                for word in words:
                    word_data = word.split('\t')
                    if len(word_data) <= 1:
                        continue
                    POS_IDX = 3
                    WORD_IDX = 1
                    pos.append([word_data[WORD_IDX], word_data[POS_IDX]])
                pos_tags += pos
        return pos_tags



if __name__ == '__main__':
    sentence = "Jag ler bra. Hur ler du? H §§. Pappa äter korv under bron"
    import time
    t = time.time()
    parser = MaltParserEx()
    parser.buffer(sentence, {'test':"yeo"})
    print(zip(*parser.word_tokenize('Pappa äter korv under bron', words=True)))
    print(parser._tags_buff['Pappa äter korv under bron'])
    print(parser._meta_buff['Pappa äter korv under bron'])
    print(parser.dependency_parse('Pappa äter korv under bron'))
    #sentences = MaltParserEx.pos_tag(sentence)
    #print("Time", time.time()-t)
    #t = time.time()
    #malt = MaltParser(
    #    parser_dirname=maltdir,
    #    model_filename=os.path.join(maltdir, 'swemalt-1.7.2.mco'),
    #    tagger=MaltParserEx.pos_tag)
    #print([list(g)[0].nodes for g in malt.parse_tagged_sents(sentences, verbose=True)])
    print("Time", time.time()-t)