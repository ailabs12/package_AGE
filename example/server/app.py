from flask import Flask, json, request
from copy import deepcopy
import asyncio
import time #замер времени
import threading
import queue
import base64

import os
import sys

TestPath = os.path.dirname(os.path.abspath(__file__)) #директория с текущим файлом
ImgPath = os.path.join(TestPath, 'man300x300_base64.txt') #картинка в той же директории

AgeGenderEmoPath = os.path.abspath(os.path.join(__file__,"../../../..")) #путь до директории, в которой находится пакет детектора 

sys.path.append(AgeGenderEmoPath) #добавляем в окружение путь до директории

from AgeGenderEmo.AgeGender.age_gender_recognition_retail_0013 import AgeGenderRecognition
from AgeGenderEmo.Emo.emotions_recognition_retail_0003 import EmoRecognition

def get_age_gender(img_b64,q):
	return q.put_nowait(AgeGenderRecognition(img_b64))

def get_emo(img_b64,q):
	return q.put_nowait(EmoRecognition(img_b64))

app = Flask(__name__)

@app.route("/", methods=['POST'])
def detectorAGE():
	if (not is_valid_request(request)):
		return json.jsonify(get_json_response(msg='Error! Check the validity of the JSON request and the existence of the field \'image\'')) #Invalid request!  Field \'image\' not found

	img_b64 = get_request_data(request) #(string img_b64) (tuple get_request_data(request))
	img_header = get_image_header(img_b64)
	img_body = get_image_body(img_b64)

	if (img_body is None):
		return json.jsonify(get_json_response(msg='Image not found'))

	prediction_result_age_gender = None
	prediction_result_emo = None

	start_time = time.time() #замер времени

	qe_age_gender = queue.Queue()
	qe_emo = queue.Queue()

	t_get_age_gender = threading.Thread(name = 'get_age_gender', target = get_age_gender, args = (img_b64,qe_age_gender))

	t_get_emo = threading.Thread(name = 'get_emo', target = get_emo, args = (img_b64,qe_emo))

	t_get_age_gender.start()
	t_get_emo.start()

	while t_get_age_gender.is_alive() or t_get_emo.is_alive(): # пока функция выполняется
		prediction_result_age_gender = qe_age_gender.get()
		if not t_get_emo.is_alive():
			prediction_result_emo = qe_emo.get()
		t_get_age_gender.join(2)
		t_get_emo.join(2)
	
	print("--- %s seconds ---" % (time.time() - start_time)) #конец замера времени



	todayold = time.time() #замер времени

	pred_result_age_gender = AgeGenderRecognition(img_b64) #list age gender
	pred_result_emo = EmoRecognition(img_b64) #list emo

	print("--- %s seconds old ---" % (time.time() - todayold)) #конец замера времени

	return json.jsonify(get_json_response(result_ag=prediction_result_age_gender,result_emo=prediction_result_emo,img_header=img_header))


def is_valid_request(request):
	try:
		request.get_json(force=True)
	except:
		return False
	return 'image' in request.json

def get_request_data(request):
	r = request.json
	image = r['image'] if 'image' in r else ''
	return image

def get_image_body(img_b64):
	if 'data:image' in img_b64:
		img_encoded = img_b64.split(',')[1]
		return base64.decodebytes(img_encoded.encode('utf-8'))
	else:
		return None

def get_image_header(img_b64):
	if 'data:image' in img_b64:
		#data:image/jpeg;base64,
		return img_b64.split(',')[0] + ','
	else:
		return None

def get_json_response(result_ag=None, result_emo=None, msg=None, img_header=None):
	json = {
		'success': False
	}

	if msg is not None:
		json['message'] = msg
		return json

	json['data'] = []

	if result_ag is None and result_emo is None:
		return json

	if result_ag is not None and result_emo is not None:
		result = {**result_ag, **result_emo}
		json['data'].append(deepcopy(result))

	if result_ag is not None and result_emo is None:
		json['data'].append(deepcopy(result_ag))

	if result_ag is None and result_emo is not None:
		json['data'].append(deepcopy(result_emo))

	json['success'] = True
	return json
