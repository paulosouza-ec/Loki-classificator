import time


def ajuste_tempo(time_init, time_end,debug_time = False):
    """
    Função criada para realizar atrasos no script.

    :param time_init: tempo de inicio do script
    :param time_end: tempo de finalização do script
    :param debug_time: por  defaul False função de desenvolvedor
    :return: Nada
    """
    print("Entrando no Ajuste de tempo")
    if time_init>=time_end:
        time_end+=60 # correcao de horarios pois o inicio chegou em 60
        if debug_time: print(time.asctime(time.localtime(time.time())))
    if(time_end - time_init <= 60): # significa que ele zerou
        if debug_time: print('faltam ', 60-(time_end-time_init), ' secs para  completar um minuto')
        time.sleep(60-(time_end-time_init))
    else:
        if debug_time: print('Completamos mais que 1 minuto')
    print("Saindo do ajuste de tempo")

if __name__ == "__main__":

    debug_time = False
    print('Horario inicial é: ', time.localtime())
    time_init = time.localtime()[5]
    if debug_time: print(time_init)
    for i in range(10000000): i+=1
    time_end = time.localtime()[5]
    if debug_time: print(time_end)
    ajuste_tempo(time_init, time_end)
    print('horario atual é: ', time.localtime())

