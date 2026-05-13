from .utils import timeout
import requests
import pathlib
import hashlib

class virustotal:

    def  __init__(self,apikey):
        self.apikey = apikey

    def scan(self, file: pathlib.Path):
        try:
            url = 'https://www.virustotal.com/vtapi/v2/file/scan'

            params = {'apikey': self.apikey}

            files = {'file': (file.name, open(file, 'rb'))}

            return requests.post(url, files=files, params=params)

        except requests.Timeout as error:
            print("Tivemos um timeout!!")
            print(error)
            timeout()


    def report(self,file: pathlib.Path, resource = ""):
        try:
            with open(file, 'rb') as hash_file:
                bytes = hash_file.read()

            readable_hash = hashlib.sha256(bytes).hexdigest()

            if resource != "": readable_hash

            url = 'https://www.virustotal.com/vtapi/v2/file/report'

            params = {'apikey': self.apikey, 'resource': readable_hash}

            return requests.get(url, params=params)

        except requests.Timeout as error:
            print("Tivemos um timeout!!!")
            print(error)
            timeout()

    def status(self,response: requests.Response):

        if response.status_code == 200: # deu tudo certo
            return True

        elif response.status_code == 204: # excedemos o limite de arquivos no minuto
            print('Error 204')
            print("limite de arquivos excedidos por minuto")
            return False

        elif response.status_code == 400 or response.status_code == 403: # excedemos o limite de arquivos no minuto
            print('Alguma coisa no esta errada, favor verificar.')
            print("DICA: Resquest ou APIKEY")
            exit()


