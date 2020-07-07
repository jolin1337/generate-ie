
section=${1:-all}
STANFORD_VERSION=${$1:-"stanford-corenlp-full-2018-10-05"}

trainGensim () {
    echo "Training Gensim word2vec model"
    git clone https://github.com/klintan/wiki-word2vec
    cd wiki-word2vec

    pyenv local 3.6.4
    pip install gensim
    make LANGUAGE=sv
    cd -
}


trainDependencyParser () {
    echo "Clone swedish depparse model repo"
    git clone https://github.com/klintan/corenlp-swedish-depparse-model
    cd corenlp-swedish-depparse-model
    if [ -d "$STANFORD_VERSION" ]; then
	    echo "Skipping download of $STANFORD_VERSION, since it already exists"
    else
        echo "Downloading $STANFORD_VERSION"
        wget -O $STANFORD_VERSION.zip http://nlp.stanford.edu/software/$STANFORD_VERSION.zip
        unzip $STANFORD_VERSION.zip -d ./
    fi

    echo "Download Part of speech tagger"
    wget -O swedish.tagger https://raw.githubusercontent.com/klintan/corenlp-swedish-pos-model/master/swedish.tagger

    echo "Download conllu dataset for dependency parsing"
    [ ! -f ../data/sv_talbanken-ud-train.conllu ] && wget -O ../data/sv_talbanken-ud-train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-train.conllu
    [ ! -f ../data/sv_talbanken-ud-test.conllu ] && wget -O ../data/sv_talbanken-ud-test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-test.conllu
    [ ! -f ../data/sv_talbanken-ud-dev.conllu ] && wget -O ../data/sv_talbanken-ud-dev.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/master/sv_talbanken-ud-dev.conllu
    [ ! -f ../data/sv_lines-ud-train.conllu ] && wget -O ../data/sv_lines-ud-train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-LinES/master/sv_lines-ud-train.conllu
    [ ! -f ../data/sv_lines-ud-test.conllu ] && wget -O ../data/sv_lines-ud-test.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-LinES/master/sv_lines-ud-test.conllu
    [ ! -f ../data/sv_lines-ud-dev.conllu ] && wget -O ../data/sv_lines-ud-dev.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-LinES/master/sv_lines-ud-dev.conllu
    [ ! -f ../data/sv_pud-ud-train.conllu ] && wget -O ../data/sv_pud-ud-train.conllu https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-PUD/master/sv_pud-ud-test.conllu
    tail -n +2 ../wiki-word2vec/data/sv/model_sv.word2vec.model.txt > model_sv.word2vec.model.txt
    echo "Train dependency parser"
    ../trainDependencyParser.sh $STANFORD_VERSION model_sv.word2vec.model.txt
    cd -
}


case $section in
    "all")
        trainGensim
        trainDependencyParser
        ;;
    "gensim")
        trainGensim
        ;;
    "dependencyParser")
        trainDependencyParser
        ;;
    *)
        echo "Unkown model \"$section\""
        ;;
esac
