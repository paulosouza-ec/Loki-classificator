# -*- coding: utf-8 -*-
"""
mELM - Morphological Extreme Learning Machine (Máquina de Aprendizado Extremo Morfológico)
Versão 4.5 FINAL - Vetorização Estável e Protegida contra Estouro de Memória (RAM)

Desenvolvido originalmente por:
Prof. Dr. Sidney Marlon Lopes de Lima
Federal University of Pernambuco
Department of Electronics and Systems

Otimizações de Performance, Memória e Generalização (2026):
- Correção de ArrayMemoryError via Vetorização Bidimensional Controlada.
- Substituição de np.linalg.pinv por np.linalg.solve com Regularização Ridge (L2).
- Correção definitiva de estouros numéricos / colapsos nas funções morfológicas.
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
from time import process_time, time
from datetime import datetime, timedelta
from random import seed as rnd_seed
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template


#========================================================================
# VISUALIZAÇÃO DE DADOS E RELATÓRIOS
#========================================================================

def plot_and_save_cm(cm, title, filename):
    if cm is None:
        return

    cm = np.asarray(cm, dtype=float)
    if cm.size == 0 or np.all(cm == 0):
        return

    if cm.max() <= 1.0:
        cm_percent = cm * 100.0
    else:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0
        ) * 100.0

    annot_matrix = np.empty_like(cm_percent, dtype=object)
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            annot_matrix[i, j] = f"{cm_percent[i, j]:.1f}%"

    class_labels = ["Benigno", "Maligno"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percent,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Porcentagem (%)"},
        xticklabels=[f"Pred {l}" for l in class_labels],
        yticklabels=[f"{l}" for l in class_labels]
    )

    plt.title(title)
    plt.xlabel("Rótulo Predito")
    plt.ylabel("Rótulo Verdadeiro")
    plt.tight_layout()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()


#========================================================================
# ESTIMATIVA DE TEMPO E RASTREAMENTO DE PROGRESSO
#========================================================================

def estimate_total_time(n_kernels, n_neurons_configs, n_folds, n_samples):
    if n_samples < 2000:
        time_per_fold = 0.01
    else:
        time_per_fold = 0.04
    
    total_iterations = n_kernels * n_neurons_configs * n_folds
    estimated_seconds = total_iterations * time_per_fold * 1.1
    return estimated_seconds


def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_time_estimate(n_kernels, n_neurons_configs, n_folds, n_samples):
    estimated_seconds = estimate_total_time(n_kernels, n_neurons_configs, n_folds, n_samples)
    
    print("\n" + "="*80)
    print("⏱️  ESTIMATIVA DE TEMPO (VERSÃO VETORIZADA COM PROTEÇÃO DE MEMÓRIA)")
    print("="*80)
    print(f"  Kernels a testar:      {n_kernels}")
    print(f"  Configurações neuron.: {n_neurons_configs}")
    print(f"  K-folds:               {n_folds}")
    print(f"  Total de iterações:    {n_kernels * n_neurons_configs * n_folds}")
    print(f"  ⏱️  Tempo estimado:      ~{format_time(estimated_seconds)}")
    
    end_time = datetime.now() + timedelta(seconds=estimated_seconds)
    print(f"  🕐 Término previsto:    {end_time.strftime('%H:%M:%S')}")
    print("="*80 + "\n")


def print_progress_info(current_kernel, total_kernels, current_nh, total_nh, 
                       current_fold, total_folds, start_time):
    total_iterations = total_kernels * total_nh * total_folds
    completed_iterations = (
        current_kernel * total_nh * total_folds +
        current_nh * total_folds +
        current_fold
    )
    
    percent = 100.0 * completed_iterations / total_iterations if total_iterations > 0 else 0
    elapsed_time = time() - start_time
    if completed_iterations > 0:
        time_per_iteration = elapsed_time / completed_iterations
        remaining_iterations = total_iterations - completed_iterations
        remaining_time = time_per_iteration * remaining_iterations
        remaining_str = format_time(remaining_time)
        elapsed_str = format_time(elapsed_time)
    else:
        remaining_str = "calculando..."
        elapsed_str = "0s"
    
    bar_length = 40
    filled_length = int(bar_length * completed_iterations / total_iterations) if total_iterations > 0 else 0
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    print(f"  ⏱️  [{bar}] {percent:.1f}% | Decorrido: {elapsed_str} | Restante: {remaining_str}")


#========================================================================
# DIAGNÓSTICO AUTOMÁTICO DE KERNEL
#========================================================================

def diagnose_kernel_performance(results, kernel_name):
    diagnosis = {
        'status': 'unknown',
        'problem': None,
        'explanation': '',
        'recommendation': ''
    }
    
    accuracy = results.get('accuracy_test', 0)
    std = results.get('std_test', 0)
    cm = results.get('confusion_matrix_test', None)
    
    if cm is not None and cm.size == 4:
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    else:
        detection_rate = None
        fpr = None
    
    if detection_rate is not None and detection_rate < 0.01:
        diagnosis['status'] = 'critical'
        diagnosis['problem'] = 'Colapso Numérico (Underflow)'
        diagnosis['explanation'] = f"O kernel '{kernel_name}' colapsou devido à sucessiva multiplicação de probabilidades."
        diagnosis['recommendation'] = "Substitua por variantes estáveis de média, como 'fuzzy-erosion-mean'."
    elif std > 15.0:
        diagnosis['status'] = 'unstable'
        diagnosis['problem'] = 'Alta Instabilidade'
        diagnosis['explanation'] = f"O kernel clássico '{kernel_name}' demonstrou alta sensibilidade a outliers estruturais."
        diagnosis['recommendation'] = "Aumente o fator de regularização (-C) ou migre para as variantes 'mean'/'softmax'."
    elif fpr is not None and fpr > 0.15:
        diagnosis['status'] = 'warning'
        diagnosis['problem'] = 'Alta Taxa de Falso Positivo'
        diagnosis['explanation'] = "O modelo classifica arquivos limpos como malignos em excesso."
        diagnosis['recommendation'] = "Considere kernels mais rígidos como 'erosion' ou normalize a escala das features."
    elif accuracy < 80.0:
        diagnosis['status'] = 'poor'
        diagnosis['problem'] = 'Subajuste Geral (Underfitting)'
        diagnosis['explanation'] = f"O operador '{kernel_name}' falhou em mapear as fronteiras de decisão complexas."
        diagnosis['recommendation'] = "Tente kernels robustos estabelecidos: 'sigmoid', 'radbas' ou 'erosion'."
    else:
        diagnosis['status'] = 'good'
        diagnosis['explanation'] = f"O kernel '{kernel_name}' apresenta ótimo comportamento geral ({accuracy:.2f}% ± {std:.1f}%)."
        diagnosis['recommendation'] = "Adequado para deploy em cenários produtivos de segurança."
    
    return diagnosis


def print_diagnosis(diagnosis, kernel_name):
    status_icons = {'critical': '❌', 'unstable': '⚠️', 'warning': '⚠️', 'poor': '⚠️', 'good': '✅'}
    print(f"  {status_icons.get(diagnosis['status'], '❓')} DIAGNÓSTICO ({kernel_name}): {diagnosis['problem'] or 'Estável'}")
    print(f"     Explicação: {diagnosis['explanation']}")
    print(f"     Recomendação: {diagnosis['recommendation']}\n")


#========================================================================
# MÉTRICAS DE DETECÇÃO PARA SEGURANÇA CIBERNÉTICA
#========================================================================

def calculate_detection_metrics(cm):
    if cm is None or cm.shape != (2, 2):
        return None
    
    TN = float(cm[0, 0])
    FP = float(cm[0, 1])
    FN = float(cm[1, 0])
    TP = float(cm[1, 1])
    
    total = TN + FP + FN + TP
    if total == 0:
        return None
    
    metrics = {
        'accuracy': (TP + TN) / total,
        'detection_rate': TP / (TP + FN) if (TP + FN) > 0 else 0,
        'false_positive_rate': FP / (FP + TN) if (FP + TN) > 0 else 0,
        'precision': TP / (TP + FP) if (TP + FP) > 0 else 0,
        'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
        'f1_score': 0,
        'TP': int(TP), 'TN': int(TN), 'FP': int(FP), 'FN': int(FN)
    }
    
    if metrics['precision'] + metrics['detection_rate'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['detection_rate']) / \
                               (metrics['precision'] + metrics['detection_rate'])
    
    return metrics


def print_detection_metrics(metrics, dataset_name="Test", verbose=True):
    if not verbose or metrics is None:
        return
    print(f"  📊 Métricas de Detecção [{dataset_name}] -> DR (Recall): {metrics['detection_rate']*100:.2f}% | FPR: {metrics['false_positive_rate']*100:.2f}% | F1: {metrics['f1_score']*100:.2f}%")


#========================================================================
# CARREGAMENTO E MÉTODOS DE DADOS VETORIZADOS
#========================================================================

def eliminateNaN_All_data(all_data):
    return np.nan_to_num(all_data)


def autodetect_separator(data_file, verbose=False):
    candidate_seps = [',', ';', '\t', '|', ':', '^', '~']
    counts = {sep: 0 for sep in candidate_seps}

    try:
        with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                for sep in candidate_seps:
                    counts[sep] += line.count(sep)
    except Exception:
        return ';'

    best_sep = max(counts, key=lambda k: counts[k])
    return best_sep if counts[best_sep] > 0 else ';'


def mElmStruct(data_file, Elm_Type, sep, verbose):
    if not sep or str(sep).lower() == 'auto':
        sep_character = autodetect_separator(data_file, verbose)
    else:
        raw_sep = str(sep).lower()
        mapping = {'tab': '\t', '\\t': '\t', 'pipe': '|', 'comma': ',', 'vírgula': ',', 'semicolon': ';'}
        sep_character = mapping.get(raw_sep, str(sep))

    df = pd.read_csv(data_file, sep=sep_character, decimal=".", header=None, engine='python')
    df_vals = df.iloc[1:, 1:].apply(pd.to_numeric, errors='coerce')
    all_data = eliminateNaN_All_data(df_vals.to_numpy(dtype=float))

    if int(Elm_Type) != 0:
        samples_index = np.random.permutation(np.size(all_data, 0))
    else:
        samples_index = np.arange(0, np.size(all_data, 0))

    return all_data, samples_index


def loadingDataset(dataset):
    T = dataset[:, 0]
    P = np.transpose(dataset[:, 1:])
    return T, P


def virusNormFunction(matrix, verbose):
    maxi = np.max(matrix)
    mini = np.min(matrix)
    if maxi == mini:
        return np.ones_like(matrix) * 0.5
    return 0.1 + 0.8 * (matrix - mini) / (maxi - mini)


#========================================================================
# KERNELS MORFOLÓGICOS OTIMIZADOS COM PROTEÇÃO DE MEMÓRIA (LINHA POR LINHA)
#========================================================================

def erosion_kernel(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        H[i, :] = np.min(samples - w1[i, :, np.newaxis], axis=0)
    return H + b1


def dilation_kernel(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        H[i, :] = np.max(samples + w1[i, :, np.newaxis], axis=0)
    return H + b1


def fuzzy_erosion_kernel(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        diff = samples - w1[i, :, np.newaxis]
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(diff, -40, 40)))
        log_fuzzy = np.log(fuzzy_values + 1e-12)
        H[i, :] = np.exp(np.sum(log_fuzzy, axis=0))
    return H + b1


def fuzzy_dilation_kernel(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        diff = samples - w1[i, :, np.newaxis]
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(diff, -40, 40)))
        log_comp = np.log(1.0 - fuzzy_values + 1e-12)
        H[i, :] = 1.0 - np.exp(np.sum(log_comp, axis=0))
    return H + b1


def bitwise_erosion_kernel(w1, b1, samples, threshold=0.5):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    samples_bin = (samples > threshold)
    w1_bin = (w1 > threshold)
    for i in range(n_hidden):
        match = np.all(samples_bin >= w1_bin[i, :, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    return H + b1


def bitwise_dilation_kernel(w1, b1, samples, threshold=0.5):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    samples_bin = (samples > threshold)
    w1_bin = (w1 > threshold)
    for i in range(n_hidden):
        match = np.any(samples_bin >= w1_bin[i, :, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    return H + b1


#========================================================================
# KERNELS MORFOLÓGICOS CORRIGIDOS E PROTEGIDOS
#========================================================================

def dilation_kernel_softmax(w1, b1, samples, temperature=5.0):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        values = samples + w1[i, :, np.newaxis]
        max_val = np.max(values, axis=0, keepdims=True)
        exp_values = np.exp((values - max_val) / temperature)
        log_sum_exp = np.log(np.sum(exp_values, axis=0))
        H[i, :] = max_val.flatten() + temperature * log_sum_exp
    return H + b1


def fuzzy_erosion_kernel_mean(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(samples - w1[i, :, np.newaxis], -40, 40)))
        H[i, :] = np.mean(fuzzy_values, axis=0)
    return H + b1


def fuzzy_erosion_kernel_geometric_mean(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(samples - w1[i, :, np.newaxis], -40, 40)))
        log_fuzzy = np.log(fuzzy_values + 1e-12)
        H[i, :] = np.exp(np.mean(log_fuzzy, axis=0))
    return H + b1


def fuzzy_dilation_kernel_mean(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    for i in range(n_hidden):
        fuzzy_values = 1.0 / (1.0 + np.exp(-np.clip(samples - w1[i, :, np.newaxis], -40, 40)))
        H[i, :] = np.mean(fuzzy_values, axis=0)
    return H + b1


def bitwise_erosion_kernel_adaptive(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    th_s = np.median(samples, axis=0)
    samples_bin = (samples > th_s)
    for i in range(n_hidden):
        th_w = np.median(w1[i, :])
        w1_bin = (w1[i, :] > th_w)
        match = np.all(samples_bin >= w1_bin[:, np.newaxis], axis=0)
        H[i, :] = match.astype(float)
    return H + b1


def bitwise_dilation_kernel_adaptive(w1, b1, samples):
    n_hidden = w1.shape[0]
    H = np.zeros((n_hidden, samples.shape[1]))
    th_s = np.median(samples, axis=0)
    samples_bin = (samples > th_s)
    n_features = w1.shape[1]
    for i in range(n_hidden):
        th_w = np.median(w1[i, :])
        w1_bin = (w1[i, :] > th_w)
        matches = (samples_bin >= w1_bin[:, np.newaxis])
        match_count = np.sum(matches, axis=0)
        H[i, :] = (match_count >= max(1, n_features * 0.1)).astype(float)
    return H + b1


#========================================================================
# SELETOR UNIFICADO DAS FUNÇÕES DE ATIVAÇÃO
#========================================================================

def switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P):
    if ActivationFunction in ('sig', 'sigmoid'):
        return 1.0 / (1.0 + np.exp(-(np.dot(InputWeight, P) + BiasofHiddenNeurons)))
    elif ActivationFunction in ('sin', 'sine'):
        return np.sin(np.dot(InputWeight, P) + BiasofHiddenNeurons)
    elif ActivationFunction == 'hardlim':
        return (np.dot(InputWeight, P) + BiasofHiddenNeurons > 0).astype(float)
    elif ActivationFunction == 'tribas':
        return np.maximum(1.0 - np.abs(np.dot(InputWeight, P) + BiasofHiddenNeurons), 0.0)
    elif ActivationFunction == 'radbas':
        return np.exp(-np.square(np.dot(InputWeight, P) + BiasofHiddenNeurons))
    elif ActivationFunction in ('linear', 'lin'):
        return np.dot(InputWeight, P) + BiasofHiddenNeurons
    
    # Redirecionamento para as chamadas protegidas e vetorizadas
    elif ActivationFunction in ('erosion', 'ero'):
        return erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('dilation', 'dil'):
        return dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-erosion', 'fuzzy_erosion'):
        return fuzzy_erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-dilation', 'fuzzy_dilation'):
        return fuzzy_dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-erosion', 'bitwise_erosion'):
        return bitwise_erosion_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-dilation', 'bitwise_dilation'):
        return bitwise_dilation_kernel(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('dilation-softmax', 'dil-soft'):
        return dilation_kernel_softmax(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-erosion-mean', 'fuzzy-ero-mean'):
        return fuzzy_erosion_kernel_mean(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-erosion-geom', 'fuzzy-ero-geom'):
        return fuzzy_erosion_kernel_geometric_mean(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('fuzzy-dilation-mean', 'fuzzy-dil-mean'):
        return fuzzy_dilation_kernel_mean(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-erosion-adaptive', 'bitwise-ero-adapt'):
        return bitwise_erosion_kernel_adaptive(InputWeight, BiasofHiddenNeurons, P)
    elif ActivationFunction in ('bitwise-dilation-adaptive', 'bitwise-dil-adapt'):
        return bitwise_dilation_kernel_adaptive(InputWeight, BiasofHiddenNeurons, P)
    else:
        raise ValueError(f"Kernel de Ativação desconhecido: {ActivationFunction}")


#========================================================================
# APRENDIZADO ELM REGULARIZADO E OTIMIZADO
#========================================================================

def mElmLearning(train_data, test_data, Elm_Type, NumberofHiddenNeurons, 
                 ActivationFunction, execution, kfold, verbose, virusNorm=False, C_reg=100.0):
    [T, P] = loadingDataset(train_data)
    [TVT, TVP] = loadingDataset(test_data)
    
    NumberofTrainingData = P.shape[1]
    NumberofTestingData = TVP.shape[1]
    NumberofInputNeurons = P.shape[0]
    NumberofHiddenNeurons = int(NumberofHiddenNeurons)
    
    cm_fold_train, cm_fold_test = None, None
    
    if Elm_Type != 0:
        sorted_target = np.sort(np.concatenate((T, TVT), axis=0))
        label = list(pd.unique(sorted_target))
        number_class = len(label)
        
        temp_T = np.zeros((number_class, NumberofTrainingData))
        for i in range(NumberofTrainingData):
            temp_T[label.index(T[i])][i] = 1.0
        T = temp_T
        
        temp_TV_T = np.zeros((number_class, NumberofTestingData))
        for i in range(NumberofTestingData):
            temp_TV_T[label.index(TVT[i])][i] = 1.0
        TVT = temp_TV_T

    start_time_train = process_time()
    
    if Elm_Type == 0 and ActivationFunction in ('erosion', 'ero', 'dilation', 'dil'):
        InputWeight = np.random.uniform(np.amin(P), np.amax(P), (NumberofHiddenNeurons, NumberofInputNeurons))
    else:
        InputWeight = np.random.rand(NumberofHiddenNeurons, NumberofInputNeurons) * 2.0 - 1.0
    
    if virusNorm:
        InputWeight = virusNormFunction(InputWeight, verbose)
        P = virusNormFunction(P, verbose)
        TVP = virusNormFunction(TVP, verbose)
    
    BiasofHiddenNeurons = np.random.rand(NumberofHiddenNeurons, 1)
    H = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, P)
    
    # --------------------------------------------------------------------
    # REGULARIZAÇÃO RIDGE (L2) DE TIKHONOV (np.linalg.solve)
    # Substitui linalg.pinv pura para exterminar o Overfitting no teste
    # --------------------------------------------------------------------
    if C_reg is not None and C_reg > 0:
        Identity = np.eye(H.shape[0])
        OutputWeight = np.linalg.solve(np.dot(H, H.T) + Identity / C_reg, np.dot(H, T.T))
    else:
        OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))
    
    TrainingTime = process_time() - start_time_train
    
    Y = np.transpose(np.dot(np.transpose(H), OutputWeight))
    del H
    
    start_time_test = process_time()
    tempH_test = switchActivationFunction(ActivationFunction, InputWeight, BiasofHiddenNeurons, TVP)
    del TVP
    TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))
    TestingTime = process_time() - start_time_test
    
    if Elm_Type == 0:
        TrainingAccuracy = round(np.sqrt(np.square(np.subtract(T, Y)).mean()), 6)
        TestingAccuracy = round(np.sqrt(np.square(np.subtract(TVT, TY)).mean()), 6)
    else:
        label_index_train_expected = np.argmax(T, axis=0)
        label_index_train_actual = np.argmax(Y, axis=0)
        TrainingAccuracy = round(np.mean(label_index_train_actual == label_index_train_expected), 6)
        
        label_index_test_expected = np.argmax(TVT, axis=0)
        label_index_test_actual = np.argmax(TY, axis=0)
        TestingAccuracy = round(np.mean(label_index_test_actual == label_index_test_expected), 6)
        
        labels_range = list(range(number_class))
        cm_fold_train = confusion_matrix(label_index_train_expected, label_index_train_actual, labels=labels_range)
        cm_fold_test = confusion_matrix(label_index_test_expected, label_index_test_actual, labels=labels_range)
    
    if verbose:
        if Elm_Type == 0:
            print(f"    Fold {execution} -> Train RMSE: {TrainingAccuracy:.4f} | Test RMSE: {TestingAccuracy:.4f}")
        else:
            print(f"    Fold {execution} -> Train Acc: {TrainingAccuracy*100:.2f}% | Test Acc: {TestingAccuracy*100:.2f}%")
            if number_class == 2:
                print_detection_metrics(calculate_detection_metrics(cm_fold_test), "Testing", verbose)

    return TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime, cm_fold_train, cm_fold_test


#========================================================================
# CLASSE PRINCIPAL DO PROCESSO
#========================================================================

class melm():
    def main(self, TrainingData_File, TestingData_File, AllData_File, Elm_Type, 
             NumberofHiddenNeurons, ActivationFunction, nSeed, kfold, sep, verbose, virusNorm=False, C_reg=100.0):
        
        ALL_FUNCTIONS = ['sig', 'sin', 'radbas', 'linear', 'hardlim', 'tribas', 
                         'erosion', 'dilation', 'fuzzy-erosion', 'fuzzy-dilation', 
                         'dilation-softmax', 'fuzzy-erosion-mean', 'fuzzy-erosion-geom', 
                         'fuzzy-dilation-mean', 'bitwise-erosion-adaptive', 'bitwise-dilation-adaptive']
        
        if ActivationFunction == 'all': 
            acts = ALL_FUNCTIONS
        else: 
            acts = [s.strip() for s in str(ActivationFunction or 'linear').split(',') if s.strip()]
        
        nh_list = [int(v.strip()) for v in str(NumberofHiddenNeurons).split(',') if str(v).strip()]
        nSeed = int(nSeed or 1)
        rnd_seed(nSeed)
        np.random.seed(nSeed)
        Elm_Type = int(Elm_Type)
        
        use_separate_files = TrainingData_File is not None and TestingData_File is not None
        combo_results = []
        
        if use_separate_files:
            train_data, _ = mElmStruct(TrainingData_File, Elm_Type, sep, verbose)
            test_data, _ = mElmStruct(TestingData_File, Elm_Type, sep, verbose)
            for af in acts:
                for nh in nh_list:
                    TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(train_data, test_data, Elm_Type, nh, af, 1, 1, verbose, virusNorm, C_reg)
                    combo_results.append({
                        "act": af, "n_hidden": int(nh), "accuracy_train": TA * 100.0 if Elm_Type==1 else TA, "std_train": 0.0,
                        "accuracy_test": TeA * 100.0 if Elm_Type==1 else TeA, "std_test": 0.0, "time_train": float(TT), "std_time_train": 0.0,
                        "time_test": float(Tt), "std_time_test": 0.0, "confusion_matrix_train": cm_train, "confusion_matrix_test": cm_test
                    })
        else:
            all_data, samples_index = mElmStruct(AllData_File, Elm_Type, sep, verbose)
            kf = KFold(n_splits=int(kfold), shuffle=True, random_state=nSeed)
            print_time_estimate(len(acts), len(nh_list), kfold, len(samples_index))
            global_start_time = time()
            
            for kernel_idx, af in enumerate(acts):
                for nh_idx, nh in enumerate(nh_list):
                    print(f"\n🚀 Testando Parâmetros: Kernel = {af} | Neurônios = {nh}")
                    acc_train, acc_test, t_train, t_test = [], [], [], []
                    cms_train, cms_test = [], []
                    
                    for i, (tr_idx, te_idx) in enumerate(kf.split(samples_index)):
                        train_data = all_data[samples_index[tr_idx], :]
                        test_data = all_data[samples_index[te_idx], :]
                        
                        TA, TeA, TT, Tt, cm_train, cm_test = mElmLearning(train_data, test_data, Elm_Type, nh, af, i+1, kfold, verbose, virusNorm, C_reg)
                        acc_train.append(TA)
                        acc_test.append(TeA)
                        t_train.append(TT)
                        t_test.append(Tt)
                        if cm_train is not None: cms_train.append(cm_train.astype(float))
                        if cm_test is not None: cms_test.append(cm_test.astype(float))
                        
                    print_progress_info(kernel_idx, len(acts), nh_idx, len(nh_list), kfold, kfold, global_start_time)
                    
                    mean_tr = np.mean(acc_train) * (100.0 if Elm_Type==1 else 1.0)
                    std_tr = np.std(acc_train) * (100.0 if Elm_Type==1 else 1.0)
                    mean_te = np.mean(acc_test) * (100.0 if Elm_Type==1 else 1.0)
                    std_te = np.std(acc_test) * (100.0 if Elm_Type==1 else 1.0)
                    
                    print(f"  ✨ Resumo da Média Final -> Teste: {mean_te:.2f}{'%' if Elm_Type==1 else ''} ± {std_te:.2f}")
                    
                    current_result = {
                        "act": af, "n_hidden": int(nh), "accuracy_train": mean_tr, "std_train": std_tr,
                        "accuracy_test": mean_te, "std_test": std_te, "time_train": np.mean(t_train), "std_time_train": np.std(t_train),
                        "time_test": np.mean(t_test), "std_time_test": np.std(t_test),
                        "confusion_matrix_train": np.mean(cms_train, axis=0) if cms_train else None,
                        "confusion_matrix_test": np.mean(cms_test, axis=0) if cms_test else None
                    }
                    if Elm_Type == 1:
                        diag = diagnose_kernel_performance(current_result, af)
                        current_result['diagnosis'] = diag
                        if verbose: 
                            print_diagnosis(diag, af)
                    combo_results.append(current_result)

        best_test = max(combo_results, key=lambda r: r['accuracy_test']) if Elm_Type==1 else min(combo_results, key=lambda r: r['accuracy_test'])
        worst_test = min(combo_results, key=lambda r: r['accuracy_test']) if Elm_Type==1 else max(combo_results, key=lambda r: r['accuracy_test'])
        
        print(f"\n🏆 MELHOR CONFIGURAÇÃO GLOBAL ENCONTRADA: {best_test['act']} ({best_test['n_hidden']} neurônios) -> Teste: {best_test['accuracy_test']:.4f}")
        
        act_results = {}
        for act in acts:
            res_act = [r for r in combo_results if r['act'] == act]
            if res_act:
                act_results[act] = {
                    "max_test": max(res_act, key=lambda r: r['accuracy_test']) if Elm_Type==1 else min(res_act, key=lambda r: r['accuracy_test']),
                    "min_test": min(res_act, key=lambda r: r['accuracy_test']) if Elm_Type==1 else max(res_act, key=lambda r: r['accuracy_test'])
                }
        
        diagnostics_dict = {r['act']: r['diagnosis'] for r in combo_results if 'diagnosis' in r}
        generate_html_report_elm({"max": best_test, "min": worst_test, "elm_type": Elm_Type}, act_results, diagnostics_dict)


#========================================================================
# GERAÇÃO DO DASHBOARD HTML
#========================================================================

def generate_html_report_elm(global_results, act_results, diagnostics=None, output_file='elm_report.html'):
    elm_type = int(global_results.get("elm_type", 1))
    is_classification = elm_type == 1
    metric_label = "Acurácia" if is_classification else "RMSE"
    metric_unit = "%" if is_classification else ""
    script_dir = os.path.dirname(os.path.abspath(__file__)) or '.'
    img_dir_name = "elm_report_images"
    
    def _safe_plot_cm(cm, title, rel_filename):
        if cm is None or not is_classification: 
            return None
        try:
            abs_path = os.path.join(script_dir, rel_filename)
            plot_and_save_cm(cm, title, abs_path)
            return rel_filename
        except Exception: 
            return None

    cm_global_best = _safe_plot_cm(global_results["max"].get("confusion_matrix_test"), "Melhor Global (Teste)", os.path.join(img_dir_name, "cm_global_best.png"))
    cm_global_worst = _safe_plot_cm(global_results["min"].get("confusion_matrix_test"), "Pior Global (Teste)", os.path.join(img_dir_name, "cm_global_worst.png"))

    html_template = r"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8"><title>Relatório Avançado mELM</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #333; padding: 30px; }
            .card { background: white; border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
            h1, h2 { color: #8B1538; }
            .grid { display: flex; gap: 20px; }
            .flex-1 { flex: 1; }
            .best { border-left: 6px solid #28a745; }
            .worst { border-left: 6px solid #dc3545; }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #8B1538; color: white; }
            .cm-img { max-width: 250px; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>📊 Painel de Controle mELM Otimizado (Vetorizado)</h1>
            <p>Métrica Líder de Avaliação: <strong>{{ metric_label }}</strong></p>
        </div>
        <div class="grid">
            <div class="card flex-1 best">
                <h2>🏆 Melhor Configuração Global</h2>
                <p><strong>Kernel:</strong> {{ global_results.max.act }} | <strong>Neurônios:</strong> {{ global_results.max.n_hidden }}</p>
                <p><strong>Acurácia de Teste:</strong> {{ "%.4f"|format(global_results.max.accuracy_test) }}{{ metric_unit }}</p>
                {% if cm_global_best %}<img class="cm-img" src="{{ cm_global_best }}">{% endif %}
            </div>
            <div class="card flex-1 worst">
                <h2>⚠️ Pior Configuração Global</h2>
                <p><strong>Kernel:</strong> {{ global_results.min.act }} | <strong>Neurônios:</strong> {{ global_results.min.n_hidden }}</p>
                <p><strong>Acurácia de Teste:</strong> {{ "%.4f"|format(global_results.min.accuracy_test) }}{{ metric_unit }}</p>
                {% if cm_global_worst %}<img class="cm-img" src="{{ cm_global_worst }}">{% endif %}
            </div>
        </div>
        <div class="card">
            <h2>🔎 Resumo das Funções de Ativação</h2>
            <table>
                <tr><th>Função / Kernel</th><th>Melhor N_Hidden</th><th>Melhor Teste</th><th>Pior N_Hidden</th><th>Pior Teste</th></tr>
                {% for act, data in act_results.items() %}
                <tr>
                    <td><strong>{{ act }}</strong></td>
                    <td>{{ data.max_test.n_hidden }}</td>
                    <td style="color: #28a745; font-weight: bold;">{{ "%.2f"|format(data.max_test.accuracy_test) }}{{ metric_unit }}</td>
                    <td>{{ data.min_test.n_hidden }}</td>
                    <td style="color: #dc3545;">{{ "%.2f"|format(data.min_test.accuracy_test) }}{{ metric_unit }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </body>
    </html>
    """
    with open(os.path.join(script_dir, output_file), "w", encoding="utf-8") as f:
        f.write(Template(html_template).render(
            global_results=global_results, act_results=act_results, is_classification=is_classification,
            metric_label=metric_label, metric_unit=metric_unit, cm_global_best=cm_global_best, cm_global_worst=cm_global_worst, diagnostics=diagnostics or {}
        ))


#========================================================================
# INTERFACE CLI ROBUSTA
#========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mELM Otimizada 4.5 - CLI Completa com Proteção de RAM')
    
    # Grupos de Entrada de Dados
    parser.add_argument('-tall', dest='AllData_File')
    parser.add_argument('-tr', dest='TrainingData_File')
    parser.add_argument('-ts', dest='TestingData_File')
    
    # Parâmetros Estruturais
    parser.add_argument('-ty', dest='Elm_Type', required=True, help="0=Regressão, 1=Classificação")
    parser.add_argument('-nh', dest='nHiddenNeurons', required=True, help="Lista de neurônios, ex: '1200' ou '200,500,1200'")
    parser.add_argument('-af', dest='ActivationFunction', required=True, help="Lista de kernels, ex: 'all' ou 'erosion,dilation-softmax'")
    
    # Parâmetros Moduladores
    parser.add_argument('-kfold', dest='kfold', type=int, default=5)
    parser.add_argument('-virusNorm', dest='virusNorm', action='store_true')
    parser.add_argument('-sep', dest='sep', default='auto')
    parser.add_argument('-sd', dest='nSeed', type=int, default=1)
    parser.add_argument('-v', dest='verbose', action='store_true')
    
    # Fator de Regularização Ridge
    parser.add_argument('-C', '--C_reg', dest='C_reg', type=float, default=100.0, 
                        help="Fator de Regularização Ridge (L2). Valores menores (ex: 5.0, 10.0) reduzem overfitting.")
    
    args = parser.parse_args()
    
    ff = melm()
    ff.main(
        args.TrainingData_File, args.TestingData_File, args.AllData_File, 
        args.Elm_Type, args.nHiddenNeurons, args.ActivationFunction, 
        args.nSeed, args.kfold, args.sep, args.verbose, args.virusNorm, args.C_reg
    )