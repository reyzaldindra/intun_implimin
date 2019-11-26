import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import bpjs_ner
from bpjs_ner import LSTMTagger
import re
import string
from flask import Flask, abort, jsonify, request
import spacy
from spacy.lang.id.stop_words import STOP_WORDS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

app = Flask(__name__)
def _removeNonAscii(s):
	return "".join(i for i in s if ord(i)<128)

def clean_text(text):
	stop_words = set(stopwords.words('indonesian'))
	nlp = spacy.blank('id')
	text = text.lower()
	text = re.sub('[^a-zA-Z-0-9 ]+', '', text)
	text = _removeNonAscii(text)
	text = text.split()
	text = [nlp(word)[0].lemma_ for word in text]
	return text

def clean_entity(text):
	text = text.replace(' ','')
	text = text.split(',')
	return text

def convertToDict(sequence):
	vocab = sorted(set(np.load('vocab.npy')))
	word_to_idx = {}
	idx_to_word = {}
	word_to_idx['<pad>'] = 0
	word_to_idx['<unk>'] = 1
	for index, word in enumerate(vocab):
		word_to_idx[word] = index+2 # +1 becaause of pad token
	for word, index in word_to_idx.items():
		idx_to_word[index] = word
	seq = [word_to_idx[s] if s in word_to_idx else word_to_idx['<unk>'] for s in sequence]
	return seq

def loadModel():
	state_ner = torch.load('bpjs_ner.pth.tar')
	model_ner = torch.load('bpjs_ner.pth')
	model_ner.load_state_dict(state_ner['state_dict'])
	return model_ner

@app.route('/',methods=['POST','GET'])
@app.route('/test',methods=['POST','GET'])
def prediksi():
	sentence =request.args.get('sentence')
	# sentence = 'Berapa iuran BPJS?'
	model = loadModel()
	ner_dict =['O','B-FIN','I-FIN', 'B-ORG','I-ORG', 'B-KELAS_RAWAT','I-KELAS_RAWAT',
	'B-BIAYA_KELAS','I-BIAYA_KELAS','B-NO_KTP','I-NO_KTP', 'B-HOSPITAL','I-HOSPITAL',
	'B-SEGMEN','I-SEGMEN', 'B-KECAMATAN_FASKES','I-KECAMATAN_FASKES', 'B-KELURAHAN_FASKES',
	'I-KELURAHAN_FASKES','B-KEPEMILIKAN_FASKES','I-KEPEMILIKAN_FASKES','B-JENIS_FASKES',
	'I-JENIS_FASKES', 'B-PROVINSI_FASKES','I-PROVINSI_FASKES','B-KABUPATEN','I-KABUPATEN',
	'B-KABUPATEN_FASKES','I-KABUPATEN_FASKES','B-DISEASE','I-DISEASE', 'B-STATUS_PULANG',
	'I-STATUS_PULANG', 'B-TIPE_FASKES','I-TIPE_FASKES','B-TINGKAT_LAYANAN','I-TINGKAT_LAYANAN',
	'B-JENIS_KUNJUNGAN','I-JENIS_KUNJUNGAN','B-TGL_DATANG','I-TGL_DATANG','B-TGL_PULANG',
	'I-TGL_PULANG', 'B-TGL_TINDAKAN','I-TGL_TINDAKAN', 'B-POLIKLINIK_RUJUKAN','I-POLIKLINIK_RUJUKAN']
	sentence = clean_text(sentence)
	sentence = convertToDict(sentence)
	output=model(torch.tensor(sentence).squeeze(0))
	output = torch.max(output, 1)[1].numpy()
	output = [ner_dict[a] for a in output]
	return jsonify(output)

if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server ...")
	print("Please wait until server has fully started")
	app.run()
