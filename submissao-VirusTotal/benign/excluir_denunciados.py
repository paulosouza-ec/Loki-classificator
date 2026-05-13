import json
import requests
import time
import shutil
import os
import os.path

###########################################
cam = os.getcwd() 		 	#Altere aqui, diretorio principal onde devem estar main.cpp e o main.py
directory = '/benign'			#Altere aqui, subdiretorio dentro do diretorio principal onde devem estar os arquivos investigados
virustotal = '/virustotal'		#Altere aqui, subdiretorio que armazenara os resultados do virustotal
analises = '/cuckoobox'		#Altere aqui, subdiretorio que armazenara os resultados das analises
denunciados = '_denunciados'
#extensao = '.exe'
###########################################
os.chdir(cam + directory)							
os.system('ls > ' + cam + '/audit.txt')	
os.chdir(cam)				
###########################################

line_true = "true"
arq = open(cam + "/audit.txt","r")	
t = arq.readlines()
for line in t:
	line = line.strip()
	#[nome,a] = line.split(extensao)
	nome = line	
	exe = cam + virustotal + "/" + nome + ".json"
	print('-------------------------------------------------')
	print(exe)
	fp = open(exe, "r")
	tt = fp.readlines()
	for virus in tt:
		virus = virus.strip()
		if (virus.find(line_true.lower())!=-1):
			#print('dentro\n\n\n\n\n\n')
			#------arquivo original--------------------------
			src = cam + directory + "/" + nome 
			dst = cam + directory + denunciados + "/" + nome 
			print(src)
			print(dst)
			shutil.move(src, dst)
		
			#------virustotal--------------------------------
			src = cam + virustotal + "/" + nome + ".json"
			dst = cam + virustotal + denunciados + "/" + nome + ".json"
			print(src)
			print(dst)			
			shutil.move(src, dst)

			#------Analise-----------------------------------
			src = cam + analises + "/" + nome + ".json"
			dst = cam + analises + denunciados + "/" + nome + ".json"
			print(src)
			print(dst)			
			shutil.move(src, dst)

			break	
	fp.close()
arq.close()




















