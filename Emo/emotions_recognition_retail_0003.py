
import os
from openvino.inference_engine import IENetwork, IEPlugin

import base64
import cv2
import numpy as np
import json

from AgeGenderEmo import DEVICE_CONST

ProjectPath = os.path.dirname(os.path.abspath(__file__))

JsonPath = os.path.join(ProjectPath, 'EmoClassNames.json')
try:
	with open(JsonPath, encoding='utf-8') as f:
		EmoClassNames = json.load(f)
except:
	print('ERROR! Could not read file EmoClassNames.json')
	raise

IRmodelPath = os.path.join(ProjectPath, 'IR')

weights = os.path.join(IRmodelPath, 'emotions-recognition-retail-0003.bin')
modelFile = os.path.join(IRmodelPath, 'emotions-recognition-retail-0003.xml')

if not os.path.exists(weights) or not os.path.exists(modelFile):
	if not os.path.isfile(weights) or not os.path.isfile(modelFile):
		print('Exiting: could not find model and weights')
		raise SystemExit(1)

#Чтение модели из xml и bin файлов промежуточного представления (IR-Intermediate Representation)
net = IENetwork(model=modelFile, weights=weights)

def EmoRecognition(imageBase64):

	imageBase64 = imageBase64.replace('data:image/jpg;base64','')
	imageBase64 = imageBase64.replace('data:image/jpeg;base64','')
	imageBase64 = imageBase64.replace('data:image/png;base64','')
	imageBase64 = imageBase64.replace('data:image/gif;base64','')

	try:
		imageBase64 = base64.b64decode(imageBase64) #bytes
	except:
		print('Exiting: picture must be in base64 format')
		raise SystemExit(1)

	nparr = np.fromstring(imageBase64, np.uint8) #(bytes --> numpy.ndarray)
	img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED) #numpy.ndarray #Input network: BGR color
	#cv2.imwrite("test.jpg", img) #Для тестирования корректности преобразования в массив numpy.ndarray(выше)

	#Модель работает с изображением 64 x 64
	imgResized = cv2.resize(img, (64, 64)) #numpy.ndarray

	#Сеть принимает изображение blob на вход
	inputBlob = cv2.dnn.blobFromImage(imgResized) #numpy.ndarray

	#Имена входных и выходных слоёв
	#print(net.inputs)
	#print(net.outputs)

	#Инициализация и настройка плагина
	plugin = IEPlugin(device=DEVICE_CONST) #plugin_dirs=...

	#Загрузка сети, которая была прочитана из IR, в плагин и создание исполняемую сеть
	exec_net = plugin.load(network=net)#, num_requests=1)

	#Асинхронный вывод для первого запроса
	#Входной слой data
	exec_net.requests[0].async_infer({'data': inputBlob})
	#Ждёт пока результат станет доступным
	exec_net.requests[0].wait()

	#Выходной слой prob_emotion
	res = exec_net.requests[0].outputs['prob_emotion']
	ArrayEmo = np.reshape(res, 5) #1-D массив длины 5
	MaxVal = np.argmax(ArrayEmo) #индекс максимального значения
	className = EmoClassNames[str(MaxVal)]['Eng'] #['Rus'] для русского

	#Формирование вывода
	Output = {"emotions": {"class": className, "confidence": round(ArrayEmo[MaxVal]*100, 2)}}

	return Output #dict
