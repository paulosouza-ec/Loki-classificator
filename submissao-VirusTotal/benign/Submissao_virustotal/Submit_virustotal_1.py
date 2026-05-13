# -*- coding: utf-8 -*-
from scr.virustotal import virustotal
from scr.utils import (
    convert_json,
    estados
)
from pathlib import Path
import time
import json
import requests
import os
import os.path

"""
    Modo de uso:
    
    - Coloque o diretorio da pasta aonde esta a pasta que possui os arquivos
    - Digite o nome da pasta em "pasta"
    - Deixe o script rodar, alguns arquivos ficarão com um tamanho menor do que 1kb, isso significa que por 
        ele não existia é foi realizado o request dele, e durante isso ele esta sendo analisado, logo ele não foi
        inserido na pasta, para solucionar o problema da quantidade de arqs diferentes no final basta submeter o 
        script novamente. 
    
    
    DICA!!!!: deixe todos em execução, no final você submete novamente para colocar os que não existiam. 
    
    Editado por: Liosvaldo Santiago 
    Data: 28 de julho, 2020 
        
"""


############################
os.chdir('../benign')	
directory = os.getcwd()
path = Path(directory)
dir_entrada = Path(directory)

############################
os.chdir('..')	
directory = os.getcwd()
print(directory)

try:
    print('Criando a pasta de repositorio')
    Path(directory+'/virustotal').mkdir()


except FileExistsError:
    print("pasta ja existe, saindo")

dir_saida = Path(directory+'/virustotal')
print(dir_saida)
liosvaldo = "3be3f93c8c3d68cba0b7ded051f6efe19750a20faee1240b7a8a34353e0a178e"
projetodejavu = "ba9ce570c28153b04850faba210da5474e07a48871e913bf606dc8d2afbf401d"
igor = "cda142c4a220cdecc438699d99478ebb91bfb48c8bd6ca0cc4a5cb5e7478c003"
pablo = "3290127fd04642a00875135441bc487498483abee1173aa5a0e6c0c92cec26ed"
"""
 Notas de usuario: 
    - chaves projetodejavu e pablo são da nova geração, logo só podem submeter 1000 requests por dia!!
"""

apikey =liosvaldo
debug_time = True
debug = False
reqs = 0
virustotal = virustotal(apikey)

############################


for file in sorted(dir_entrada.iterdir()):
    time_init = time.localtime(time.time())[5]
    file_exist = dir_saida.joinpath(file.name + '.json').exists()

    print("File: ", file.name)

    if file_exist: 
        print('se ele existir na pasta de saida, não faça mais nada com ele')	
        continue #se ele existir na pasta de saida, não faça mais nada com ele

    try:
        response = ""

        response = virustotal.report(file)
        reqs = estados(reqs,time_init)

        print('Response_code', response.json()['response_code'])

        if response.json()['response_code'] == -2:
            print('Problema de -2, seguindo o arquivo')
            continue

    except json.decoder.JSONDecodeError:
        response = ""
        print('Problema no json, tentando novamente')
        response = virustotal.report(file)
        reqs = estados(reqs,time_init)


    print(response)
    try:
        if virustotal.status(response) and response.json()['response_code'] == 1: # arquivo já existente e podemos tratar ele aqui mesmo
            print("A mensagem foi recebida é existe repositorio.")
            convert_json(response.json(), file, dir_saida)

        elif virustotal.status(response) and (response.json()['response_code'] == 0):
            print("response_code_0!!!!")
            response = ""

            response = virustotal.scan(file)
            reqs = estados(reqs,time_init)
            print('Response_code Parte 2:', response.json()['response_code'])
            print(response.json())
            print(file.name, " Não existente no virustotal, scaneando ele")


            if virustotal.status(response): #True se deu tudo certo, False se tivemos quantidade excedida
                response = virustotal.report(file, response.json()['resource'])
                reqs = estados(reqs,time_init)
                print(file.name, " Reportado pelo virustotal, criando o json")
                print("Response_code ", response.json()['response_code'])
                print(response.json())

                if virustotal.status(response) and response.json()['response_code']==1: # Deu tudo certo ou não
                    convert_json(response.json(), file, dir_saida)


                else: # Error 204
                    print('Error 204!!')
                    print(response.json())
                    reqs = estados(3,time_init)

            else: # Error 204
                print('Error 204!!')
                print(response.json())
                reqs = estados(3,time_init)

        elif virustotal.status(response) and response.json()['response_code'] == -2:
            #Arquivo ainda em analise, com previsão de retorno para a proxima analise.
            continue

        elif not virustotal.status(response): #Quantidade de requisições execidas aguardar um determinado tempo
            print("Requisições excedidas, entrando em aguardo")
            reqs= estados(3,time_init)

        else:
            print("alguma coisa aconteceu, verifique!!")
            print(response.json())
            exit()

    except json.JSONDecodeError:
        print("JSON ERROR!!!")
        print(response)
        continue

    except requests.exceptions.ConnectionError:
        # Criado para tentar suprir erros de shutdown do servidor
        print("Erro de conexão... Aguardando um tempo para a retomada....")
        reqs = estados(3,time_init)
        continue


