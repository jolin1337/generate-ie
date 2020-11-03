import os

    CoreNLPUtils = None
    AnnotatedProposition = None
    MinIE = None
    MinIE_Mode = None
    StanfordCoreNLP = None
    String = None
    Integer = None
    Properties = None
    parser = None
    jnius_detatch = None

    def initialize_jnius():
        global (
            CoreNLPUtils,
            AnnotatedProposition,
            MinIE,
            MinIE_Mode,
            StanfordCoreNLP,
            String,
            Integer,
            Properties,
            parser,
            jnius_detatch
        )
        from jnius import autoclass, detatch
        try:
            jnius_detatch = detatch

            CoreNLPUtils = autoclass('de.uni_mannheim.utils.coreNLP.CoreNLPUtils')
            AnnotatedProposition = autoclass('de.uni_mannheim.minie.annotation.AnnotatedProposition')
            MinIE = autoclass('de.uni_mannheim.minie.MinIE')
            MinIE_Mode = autoclass('de.uni_mannheim.minie.MinIE$Mode')
            StanfordCoreNLP = autoclass('edu.stanford.nlp.pipeline.StanfordCoreNLP')
            String = autoclass('java.lang.String')
            Integer = autoclass('java.lang.Integer')
            Properties = autoclass('java.util.Properties')
            parser = CoreNLPUtils.StanfordDepNNParser()
        except:
            print("Warning: Java was not found!!")
            parser = None
            CoreNLPUtils = None
            AnnotatedProposition = None
            MinIE = None
            MinIE_Mode = None
            StanfordCoreNLP = None
            String = None
            Integer = None
            Properties = None

    __nr_of_docs = 0
    def get_relations(sentence):
        global __nr_of_docs
        if __nr_of_docs > 100:
            __nr_of_docs = 0
        if __nr_of_docs == 0:
            if jnius_detatch is not None:
                jnius_detatch()
            initialize_jnius()
        model = MinIE(String(sentence), parser, MinIE_Mode.SAFE)
        model.getPropositions().elements()
        triple_names = ['subject', 'relation', 'object']
        minie = [
            {
                triple_names[i]: val.toString() if val is not None else None
                for i, val in enumerate(ap.getTriple().elements())
                if i < 3
            } for ap in model.getPropositions().elements()
            if ap is not None
        ]
        minie = [triple for triple in minie if all(triple.get(n, None) is not None for n in triple_names)]
        __nr_of_docs += 1
        return minie