# PharmaCoNER

[Linguistically Informed Relation Extraction and Neural Architectures for Nested Named Entity Recognition in BioNLP-OST 2019
](https://arxiv.org/abs/1910.03385)

![BiLSTM-CRF Architecture](models/model-PharmaCoNER.PNG?raw=true "BiLSTM-CRF with Multi-Tasking")

## Setting-up

Install requirements.

``
pip install -r requirements.txt
``

### Data

Install [docker](https://docs.docker.com/compose/install/), for Ubuntu.

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo apt install docker.io
```

Download [MedTagger](https://github.com/medtagger/MedTagger.git) and run the docker daemon.

```
sudo service docker start
cd PATH_TO/MedTagger
sudo docker-compose up
```

You can skip the following steps if you are using the pre-processed version of dataset, included in this repository.

Download `train`, `development` and `test` set from PharmaCoNER's 
[webpage](http://temu.bsc.es/pharmaconer/index.php/datasets/).
Extract the content of zip files in `PharmaCoNER/dataset/raw/` directory (e.g. path of `S0004-06142005000500011-1.txt`
 relative to this project would be `PharmaCoNER/dataset/raw/train/S0004-06142005000500011-1.txt`).
 
Prepare the data by running this command.
 
``
python data.py
``

### Embeddings

Download Spanish Medical embeddings from [here](https://drive.google.com/open?id=1e4HwAsEfSpzS3ZcoU3yK0ADdsk1AtaZ4) and
store them under `resources/spanish-embs/`

## Train

```
python train.py --train dataset/data-strategy=1/train.txt --dev dataset/data-strategy=1/dev.txt --word_lstm_dim 200 --word_dim 100 --pos_dim 50 --ortho_dim 50 --pre_emb resources/spanish-embs/fasttext-100d-train_dev_test.vec --pre_emb_1 resources/spanish-embs/fasttext-100d-train_dev.vec --ranking_loss 1
```

## Evaluation

```
python evaluate.py -model models/data-strategy=1/best_model/ -data dataset/raw/test/ -span dataset/test-token_span.pkl
```

```
python ensemble.py -model1 models/data-strategy=1/best_model/ -model2 models/data-strategy=2/best_model/ -model3 models/data-strategy=3/best_model/ -data dataset/raw/test/ -span dataset/test-token_span.pkl
```


## Prediction

```
python predict.py -model models/data-strategy=1/best_model/ -text 'consideró el diagnostico de metástasis de carcinoma de mama y el Servicio de Oncología Médica inició tratamiento con epirubicina y ciclofosfamida.'
```

```
consideró	 O
el	         O
diagnostico	 O
de	         O
metástasis	 O
de	         O
carcinoma	 O
de	         O
mama	     O
y	         O
el	         O
Servicio	 O
de	         O
Oncología	 O
Médica	     O
inició	     O
tratamiento	 O
con	         O
epirubicina	 B-NORMALIZABLES
y	         O
ciclofosfamida	B-NORMALIZABLES
.	         O

```

## Results

*Test Set*

| Model                       | P          | R      | F1     |
| --------------------------- | ---------- | ------ | ------ |
| --------------------------- | ---------- | ------ | ------ |
| Data Split 1                | 0.8894     | 0.8165 | 0.8514 |
| Data Split 2                | 0.9046     | 0.8186 | 0.8595 |
| Data Split 3                | 0.8905     | 0.8175 | 0.8525 |
| Ensemble (DS1 + DS2 + DS3)  | **0.9060** | **0.8273** | **0.8649** |


## Acknowledgements

LSTM-CRF implementation is based on [https://github.com/glample/tagger](https://github.com/glample/tagger).

## Citation

If you found this codebase useful than please consider citing our work.
```
@inproceedings{DBLP:conf/bionlp/GuptaYS19,
  author    = {Usama Yaseen and
               Pankaj Gupta and
               Hinrich Sch{\"{u}}tze},
  editor    = {Jin{-}Dong Kim and
               Claire N{\'{e}}dellec and
               Robert Bossy and
               Louise Del{\'{e}}ger},
  title     = {Linguistically Informed Relation Extraction and Neural Architectures
               for Nested Named Entity Recognition in BioNLP-OST 2019},
  booktitle = {Proceedings of The 5th Workshop on BioNLP Open Shared Tasks, BioNLP-OST@EMNLP-IJNCLP
               2019, Hong Kong, China, November 4, 2019},
  pages     = {132--142},
  publisher = {Association for Computational Linguistics},
  year      = {2019}
}
```
