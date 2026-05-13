import json
import requests
import time
import shutil
import os
import os.path

###########################################
version = 'version 1.0'
#antivirus = 'template'
os.system('git config --global user.name "REFADE"')
os.system('git config --global user.email "smll@ecomp.poli.br"')
os.system('git config --global credential.helper store')
###########################################
incluidos = ["benign/benign", "benign/virustotal", "benign/cuckoobox", "malware/malware", "malware/virustotal", "malware/cuckoobox"]
cam = os.getcwd()
for ii in range(0, len(incluidos)):
#for ii in range(0, 1):
	os.chdir(cam + '/'+ antivirus + '/' + incluidos[ii])		
	os.system('ls > ' + cam + '/audit.txt')	
	os.chdir(cam +'/' + antivirus)	

	arq = open(cam + '/audit.txt', 'r')
	t = arq.readlines()
	for exe in t:
		exe = exe.strip() #retira o \n			
		print('===========================')			
		print('git add '+ incluidos[ii] + '/' + exe)		
		os.system('git add '+ incluidos[ii] + '/' + exe)	
		os.system('git commit -m \"'+ version +'\"')
		os.system('git push --force')
###########################################
