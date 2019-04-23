import os
import time #замер времени

TestPath = os.path.dirname(os.path.abspath(__file__)) #директория с текущим файлом
ImgPath = os.path.join(TestPath, 'man300x300_base64.txt') #картинка в той же директории
 
AgeGenderEmoPath = os.path.abspath(os.path.join(__file__,"../../../..")) #путь до директории, в которой находится пакет детектора 

import sys
sys.path.append(AgeGenderEmoPath) #добавляем в окружение путь до директории

from AgeGenderEmo.AgeGender.age_gender_recognition_retail_0013 import AgeGenderRecognition
from AgeGenderEmo.Emo.emotions_recognition_retail_0003 import EmoRecognition

with open(ImgPath, 'r') as myfile:
	imageBase64=myfile.read().replace('\n', '')

start_time = time.time() #замер времени

AgeGender = AgeGenderRecognition(imageBase64)
Emo = EmoRecognition(imageBase64)

print("--- %s seconds ---" % (time.time() - start_time)) #конец замера времени

print(AgeGender)
print(Emo)
