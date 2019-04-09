
import os
from openvino.inference_engine import IENetwork, IEPlugin

import base64
import cv2
import numpy as np
import json

from AgeGenderEmo import DEVICE_CONST

ProjectPath = os.path.dirname(os.path.abspath(__file__))

JsonPath = os.path.join(ProjectPath, 'GenderClassNames.json')
try:
	with open(JsonPath, encoding='utf-8') as f:
		GenderClassNames = json.load(f)
except:
	print('ERROR! Could not read file GenderClassNames.json')
	raise

IRmodelPath = os.path.join(ProjectPath, 'IR')

weights = os.path.join(IRmodelPath, 'age-gender-recognition-retail-0013.bin')
modelFile = os.path.join(IRmodelPath, 'age-gender-recognition-retail-0013.xml')

if not os.path.exists(weights) or not os.path.exists(modelFile):
	if not os.path.isfile(weights) or not os.path.isfile(modelFile):
		print('Exiting: could not find model and weights')
		raise SystemExit(1)

#Чтение модели из xml и bin файлов промежуточного представления (IR-Intermediate Representation)
net = IENetwork(model=modelFile, weights=weights)

def AgeGenderRecognition(imageBase64):

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

	#Модель работает с изображением 62 x 62
	imgResized = cv2.resize(img, (62, 62)) #numpy.ndarray

	#Сеть принимает изображение blob на вход
	inputBlob = cv2.dnn.blobFromImage(imgResized) #numpy.ndarray
	#print(inputBlob.shape)

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

	#Выходные слои age_conv3 и prob
	ArrayAge = exec_net.requests[0].outputs['age_conv3']
	ArrayAge = np.reshape(ArrayAge, 1) #делаем одномерный массив для простоты работы
	ArrayAge *= 100 #умножаем на 100, так как сеть выводит расчётный возраст делённый на 100
	age = ArrayAge[0]
	age = int(age)

	ArrayGender = exec_net.requests[0].outputs['prob'] #сеть выводит два значения (вероятности принадлежности к определённому полу [female, male])
	ArrayGender = np.reshape(ArrayGender, 2) #1-D массив длины 2
	maxim = np.argmax(ArrayGender) #индекс максимального
	gender = GenderClassNames[str(maxim)]['Eng'] #['Rus'] для русского

	#Формирование вывода
	'''
	JsonOutput = json.dumps({"age": age, "gender": {"class": gender, "confidence": round(ArrayGender[maxim]*100, 2)}}, 
	sort_keys = True, indent = 4, separators = (',', ': '), ensure_ascii=False) #<class 'str'>
	JsonInput = json.loads(JsonOutput) #<class 'dict'>
	'''
	Output = {"age": age, "gender": {"class": gender, "confidence": round(ArrayGender[maxim]*100, 2)}}

	return Output #dict
