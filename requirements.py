import os, subprocess

AgeGenderEmoPath = os.path.dirname(os.path.abspath(__file__))
ReqPath = os.path.join(AgeGenderEmoPath, 'requirements.txt')
try:
	subprocess.call(['pip3', 'install', '-r', '{ReqPath}'.format(ReqPath=ReqPath)])
except OSError:
	print('Dependencies are not installed. Install dependencies (file "requirements.txt") from the package folder.')
