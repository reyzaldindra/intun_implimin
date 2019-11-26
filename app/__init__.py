from flask import Flask, abort, jsonify, request

app = Flask(__name__)

from app import route_ner
