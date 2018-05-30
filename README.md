# UWT
unsupervised word translation

To run the experiment, see ./run.ipynb for detail. 

Before run the experiment, download the fastText word embedding data:
```bash
## English:
$mkdir data
$curl -Lo data/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
## Spanish:
$curl -Lo data/wiki.es.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
```

and download the evaluation data:
```bash
## Word Translation Task:
$mkdir -p data/crosslingual/dictionaries
$curl -Lo data/crosslingual/dictionaries/en-es.5000-6500.txt https://s3.amazonaws.com/arrival/dictionaries/en-es.5000-6500.txt
## Cross-lingual Word Translation Task:
$mkdir -p data/crosslingual/wordsim
$curl -Lo semeval2017-task2.zip http://alt.qcri.org/semeval2017/task2/data/uploads/semeval2017-task2.zip
$unzip semeval2017-task2.zip
$paste SemEval17-Task2/test/subtask2-crosslingual/data/en-es.test.data.txt SemEval17-Task2/test/subtask2-crosslingual/keys/en-es.test.gold.txt > data/crosslingual/wordsim/$lg_pair-SEMEVAL17.txt
```

Python library needed:
- Numpy
- Pytorch


