
section=${1:-all}

trainGensim () {
	echo "Training Gensim word2vec model"
	git clone https://github.com/klintan/wiki-word2vec
	cd wiki-word2vec

	pyenv local 3.6.4
	pip install gensim
	#make LANGUAGE=sv
}


cleanConllFiles () {
conllFile=$1
	python -c "
with open('$conllFile', 'r') as inp:
	with open('$conllFile.clean', 'w') as out:
		for line in inp:
			if not line.startswith('#'):
				out.write(line + '\\n')
"
}

trainDependencyParser () {
	echo "Training dependency parser"
	git clone https://github.com/klintan/corenlp-swedish-depparse-model
	cd corenlp-swedish-depparse-model
    cleanConllFiles ../data/sv_talbanken-ud-train.conllu
	cleanConllFiles ../data/sv_talbanken-ud-dev.conllu
	cleanConllFiles ../data/sv_talbanken-ud-test.conllu
	tail -n +2 ../wiki-word2vec/data/sv/model_sv.word2vec.model.txt > model_sv.word2vec.model.txt
	java -mx16g -cp "../stanford-corenlp-full-2018-10-05/*" \
         edu.stanford.nlp.parser.nndep.DependencyParser \
         -trainFile ../data/sv_talbanken-ud-train.conllu.clean \
         -devFile ../data/sv_talbanken-ud-dev.conllu.clean \
         -embedFile	model_sv.word2vec.model.txt \
         -embeddingSize 400 \
         -model swe-dependency-model.txt.gz

	echo "Removing invalid embedding tokens"
	gunzip -c swe-dependency-model.txt.gz | \
       sed -e 's/ ([^0-9\\-])/\1.\2/g' | \
      gzip -c > swe-dependency-model.txt.gz

	java -mx16g -cp "../stanford-corenlp-full-2018-10-05/*" \
		edu.stanford.nlp.parser.nndep.DependencyParser \
		-model swe-dependency-model.txt.gz
		-testFile ../data/sv_talbanken-ud-test.conllu.clean
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
