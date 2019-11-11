import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from bpjs_intent import LSTM_Intent
import re
import string
from flask import Flask, abort, jsonify, request
app = Flask(__name__)


def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
	text = text.lower()
	text = re.sub('[^a-zA-Z-0-9 ]+', '', text)
	text = _removeNonAscii(text)
	text = text.strip()
	return text

def convertToDict(sequence):
	idx_to_word = np.load('vocab.npy').tolist()
	word_to_idx = {}
	for i,(index,word) in enumerate(idx_to_word.items()):
		word_to_idx[word] = index
	seq = [np.array([word_to_idx[s] if s in word_to_idx else word_to_idx['<unk>'] for s in es.split(' ')],dtype=np.int64) for es in sequence]
	seq = [np.array(a) for a in seq]
	return seq

def loadModel():
	state_intent = torch.load('bpjs.pth.tar')
	model_intent = torch.load('checkpoint_intent_model.pth')
	model_intent.load_state_dict(state_intent['state_dict'])
	return model_intent

@app.route('/',methods=['POST','GET'])
@app.route('/index',methods=['POST','GET'])
def prediksi():
	sentence =request.form.get('sentence')
	model = loadModel()
	print(sentence)
	# sentence = 'halo nama saya joko'
	label_intent = np.array(['TRANSACTION', 'OTHERS', 'RECORD', 'PROFIL', 'GREETINGS','CLOSINGS'])
	inputs = [clean_text(sentence)]
	inputs = convertToDict(inputs)
	output=model(torch.tensor(inputs).permute(1,0))
	output=label_intent[torch.max(output, 1)[1]]
	return output

if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server ...")
	print("Please wait until server has fully started")
	app.run()