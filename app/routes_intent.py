import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from bpjs_intent import LSTM_Intent
import re
import string
from flask import Flask, abort, jsonify, request
import spacy
from spacy.lang.id.stop_words import STOP_WORDS
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))
nlp = spacy.blank('id')
app = Flask(__name__)


def _removeNonAscii(s):
	return "".join(i for i in s if ord(i)<128)

def clean_text(text):
	text = text.lower()
	text = re.sub('[^a-zA-Z-0-9 ]+', '', text)
	text = _removeNonAscii(text)
	text = text.strip()
	text = ''.join([nlp(word)[0].lemma_ for word in text])
	return text

def convertToDict(sequence):
	vocab = sorted(set(np.load('vocab intent.npy')))
	word_to_idx = {}
	idx_to_word = {}
	word_to_idx['<pad>'] = 0
	word_to_idx['<unk>'] = 1
	for index, word in enumerate(vocab):
		word_to_idx[word] = index+2 # +1 becaause of pad token
	for word, index in word_to_idx.items():
		idx_to_word[index] = word
	seq = [np.array([word_to_idx[s] if s in word_to_idx else word_to_idx['<unk>'] for s in es.split(' ')],dtype=np.int64) for es in sequence]
	return seq

def loadModel():
	state_intent = torch.load('bpjs_intent.pth.tar')
	model_intent = torch.load('bpjs_intent.pth')
	model_intent.load_state_dict(state_intent['state_dict'])
	return model_intent

@app.route('/',methods=['POST','GET'])
@app.route('/test',methods=['POST','GET'])
def prediksi():
	sentence =request.args.get('sentence')
	# sentence = 'Berapa biaya'
	model = loadModel()
	print(sentence)
	label_intent = np.array(['TRANSACTION', 'OTHERS', 'RECORD', 'PROFIL', 'GREETINGS','CLOSINGS'])
	inputs = [clean_text(sentence)]
	print(inputs)
	inputs = convertToDict(inputs)
	print(inputs)
	output=model(torch.tensor(inputs).permute(1,0))
	print(output)
	output=label_intent[torch.max(output, 1)[1]]
	return output
# print(prediksi())
if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server ...")
	print("Please wait until server has fully started")
	app.run()
