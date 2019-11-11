import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from intent_class import LSTM_Intent
import re
import string
from flask import Flask, abort, jsonify, request
app = Flask(__name__)


#Untuk NER
def clean_entity(text):
    text = text.replace(' ','') #menghapus semua spasi
    text = text.split(',') #memisahkan berdasarkan koma
    return text
def clean_sentence(text):
    text = text.lower() #mengubah menjadi lowercase
    text = re.sub('[^a-zA-Z0-9 ]+','',text)
#  text.translate(str.maketrans('','', string.punctuation)) #menghapus semua tanda baca
#  text = text.split(' ') #memisahkan berdasarkan spasi
    text = text.split()
    return text

#Untuk Intent
def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
	text = text.lower()
	text = re.sub('[^a-zA-Z-0-9 ]+', '', text)
	text = _removeNonAscii(text)
	text = text.strip()
	return text

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


def flow_pemesanan(brand,tipe):
	if len(brand)>0 and len(tipe)>0:
		return 'Berikut adalah daftar handphone dengan brand '+str(brand)+' dan tipe '+str(tipe)
	elif len(brand)==0 and len(tipe)>0:
		return 'Handphone tipe '+str(tipe)+' apa yang dimaksud?'
	elif len(brand)>0 and len(tipe)==0:
		return 'Handphone brand '+str(brand)+' tipe apa yang dimaksud?'
	else :return 'Handphonenya merk apa bro?'
	return 'pesan'

def flow_informasi(brand,tipe):
    if len(brand)>0 and len(tipe)>0:
        return 'Berikut adalah daftar handphone dengan brand '+str(brand)+' dan tipe '+str(tipe)
    elif len(brand)==0 and len(tipe)>0:
        return 'Handphone tipe '+str(tipe)+' apa yang dimaksud?'
    elif len(brand)>0 and len(tipe)==0:
        return 'Handphone brand '+str(brand)+' tipe apa yang dimaksud?'
    else :return 'Handphonenya merk apa bro?'
    return 'info'

def flow_pembelian(brand,tipe):
	if len(brand)>0 and len(tipe)>0:
		return 'Berikut adalah daftar handphone dengan brand '+str(brand)+' dan tipe '+str(tipe)
	elif len(brand)==0 and len(tipe)>0:
		return 'Handphone tipe '+str(tipe)+' apa yang dimaksud?'
	elif len(brand)>0 and len(tipe)==0:
		return 'Handphone brand '+str(brand)+' tipe apa yang dimaksud?'
	else :return 'Handphonenya merk apa bro?'
	return 'beli'

@app.route('/',methods=['POST','GET'])
@app.route('/index',methods=['POST','GET'])
def prediksi():
	state_ner = torch.load('checkpoint_ner.pth.tar')
	model_ner = torch.load('checkpoint_ner_model.pth')
	model_ner.load_state_dict(state_ner['state_dict'])
	state_intent = torch.load('checkpoint_intent.pth.tar')
	model_intent = torch.load('checkpoint_intent_model.pth')
	model_intent.load_state_dict(state_intent['state_dict'])
	idx_to_word = np.load('my_file.npy')
	idx_to_word = idx_to_word.tolist()
	dict_ner = {'B-LOCATION','I-LOCATION','B-BRAND','I-BRAND','B-TYPE','I-TYPE','B-PRICE','I-PRICE','B-SPEC','I-SPEC','B-N_SPEC','I-N_SPEC','O'}
	dict_intent = np.array(['INFORMASI', 'PEMESANAN', 'PEMBELIAN'])
	word_to_idx = {}
	for i,(index,word) in enumerate(idx_to_word.items()):
		word_to_idx[word] = index

	ner2id = np.load('ner_corpus.npy').tolist()
	id2ner = {}
	for i, a in enumerate(ner2id):
	    id2ner[i] =a
	print(ner2id,id2ner)
	sentence =request.form.get('sentence')
	inputs = [clean_sentence(sentence)]
	outputs = [word_to_idx[word] if word in word_to_idx else word_to_idx['<unk>'] for word in inputs[0]]
	outputs=torch.LongTensor(outputs)
	ner = model_ner(outputs)
	print(ner)
	ner = torch.max(ner, 1)[1].numpy()
	print(ner)
	ner = [id2ner[a] for a in ner]
	brand = []
	tipe = []
	price = []
	for i,a in enumerate(ner):
	    if 'BRAND' in a:
	        brand.append(inputs[0][i])
	    if 'TYPE' in a:
	        tipe.append(inputs[0][i])
	    if 'PRICE' in a:
	        price.append(inputs[0][i])
	query = "select * from phone where BRAND="+str(brand)+"AND TYPE="+str(tipe)+"AND PRICE="+str(price)
	inputsa = [clean_text(sentence)]
	input_test = [[word_to_idx[s] if s in word_to_idx else word_to_idx['<unk>'] for s in es.split(' ')] for es in inputsa]
	input_tensor = [pad_sequences(x,10) for x in input_test]
	output=model_intent(torch.tensor(input_tensor).permute(1,0))
	intent=dict_intent[torch.max(output, 1)[1]]
	brand = []
	tipe = []
	for i,a in enumerate(ner):
	    if 'BRAND' in a:
	        brand.append(inputs[0][i])
	    if 'TYPE' in a:
	        tipe.append(inputs[0][i])
	query = "select * from phone where BRAND="+str(brand)+"AND TYPE="+str(tipe)
	print(ner)
	switch = {
    'PEMESANAN': flow_pemesanan(brand,tipe),
    'INFORMASI': flow_informasi(brand,tipe),
    'PEMBELIAN': flow_pembelian(brand,tipe)}.get(intent)
	return jsonify(results=switch)
if __name__ == '__main__':
	print("Loading PyTorch model and Flask starting server ...")
	print("Please wait until server has fully started")
	app.run()