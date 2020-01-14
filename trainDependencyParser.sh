
cleanConllFiles () {
    inConllFile=$1
    outConllFile=$2
    mode=$3
    [ -f $inConllFile ] && python -c "
with open('$inConllFile', 'r') as inp:
    with open('$outConllFile.clean', '$mode') as out:
        for line in inp:
            if not line.startswith('#'):
                out.write(line)
"
}
removeInvalid () {
    modelIn=$1
    modelOut=$2
    python -c "
import gzip
vec_len = 400
with gzip.open('$modelIn', 'rb') as inp:
    with gzip.open('$modelOut', 'wb') as out:
        for line in inp:
          line = line.decode('utf-8')
          if ' ' not in line or '.' not in line:
            out.write(line.encode('utf-8'))
            continue
          line_split = line.split(' ')
          line_len = len(line_split)
          if line_len > vec_len:
            split_idx = line_len - vec_len - 1
            new_line = '.'.join(line_split[:split_idx]) + ' '.join(line_split[split_idx:])
            out.write(new_line.encode('utf-8'))
          else:
            out.write(line.encode('utf-8'))
"
}

STANFORD_VERSION=$1
embedFile=$2
source=lines
source=talbanken
source='all'
mode='w'
for conll_source in 'talbanken' 'pud'
do
    cleanConllFiles ../data/sv_${conll_source}-ud-train.conllu sv_${source}-ud-train.conllu $mode
    cleanConllFiles ../data/sv_${conll_source}-ud-dev.conllu sv_${source}-ud-dev.conllu $mode
    cleanConllFiles ../data/sv_${conll_source}-ud-test.conllu sv_${source}-ud-test.conllu $mode
    mode='a'
done
java -mx16g -cp "$STANFORD_VERSION/*" \
     edu.stanford.nlp.parser.nndep.DependencyParser \
     -trainFile sv_${source}-ud-train.conllu.clean \
     -devFile sv_${source}-ud-dev.conllu.clean \
     -embedFile $embedFile \
     -embeddingSize 400 \
     -model swe-dependency-model-${source}-orig.txt.gz

echo "Removing invalid embedding tokens"
#gunzip -c swe-dependency-model-${source}-orig.txt.gz | \
#   sed -re 's/ ([^0-9\\-])/.\1/g' | \
#  gzip -c > swe-dependency-model-${source}.txt.gz
removeInvalid swe-dependency-model-${source}-orig.txt.gz swe-dependency-model-${source}.txt.gz

echo "Evaluate model"
java -mx16g -cp "$STANFORD_VERSION/*" \
     edu.stanford.nlp.parser.nndep.DependencyParser \
     -model swe-dependency-model-${source}.txt.gz \
     -testFile sv_${source}-ud-test.conllu.clean

mkdir --parents "../models/${source}"
cp "server-${source}.properties" "../models/${source}/"
cp "swedish.tagger" "../models/${source}/"
cp "swe-dependency-model-${source}.txt.gz" "../models/${source}/"
echo "
annotators = tokenize, ssplit, pos, depparse
tokenize.language = se
pos.model = $( pwd )/../models/${source}/swedish.tagger
depparse.model = $( pwd )/../models/${source}/swe-dependency-model-${source}.txt.gz
" > server-${source}.properties
echo "
java -mx16g -cp \"$( pwd )/$STANFORD_VERSION/*\" \
      edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
      -serverProperties "$( pwd )/server-${source}.properties"
" > run_server.sh
