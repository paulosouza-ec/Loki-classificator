from scr.ajuste_tempo import ajuste_tempo
from pathlib import Path
import json
import time

def convert_json(json_response ,file: Path, dir_saida):
    with open(dir_saida.joinpath(str(file.name) + '.json'), 'w') as json_file:
                json.dump(json_response, json_file, indent=4)
    print("Conversão concluida!")

def estados(reqs,time_init: time, debug: bool = False):

    if reqs<3:
        reqs+=1
    else:
        reqs = 0
        time_end = time.localtime(time.time())[5]
        ajuste_tempo(time_init, time_end)
        if debug: print("Quantidade de reqs ", reqs)

    return reqs

def timeout():
    #func criada para cuidar de problemas com timeout devido a internet!
    time.sleep(300)

