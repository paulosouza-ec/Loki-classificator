# 🛡️ Detecção de Malware LokiBot via Aprendizado de Máquina

> **Trabalho de Conclusão de Curso (TCC)** — Universidade Federal de Pernambuco (UFPE)  
> **Autor:** Paulo Souza  
> **Licença:** MIT

Reprodução e extensão do experimento do Capítulo 5 (PE Scanner) utilizando a família de malware **LokiBot**, com classificação binária (benigno vs. malware) por meio de três algoritmos de aprendizado de máquina supervisionado: **SVM**, **KNN** e **Random Forest**.

---

## Sumário

- [Visão Geral do Projeto](#visão-geral-do-projeto)
- [Arquitetura do Repositório](#arquitetura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Instalação e Configuração do Ambiente](#instalação-e-configuração-do-ambiente)
- [Dataset](#dataset)
- [Pipeline Experimental](#pipeline-experimental)
  - [Etapa 1 — Coleta e Submissão ao VirusTotal](#etapa-1--coleta-e-submissão-ao-virustotal)
  - [Etapa 2 — Classificação com SVM](#etapa-2--classificação-com-svm)
  - [Etapa 3 — Classificação com KNN](#etapa-3--classificação-com-knn)
  - [Etapa 4 — Classificação com Random Forest](#etapa-4--classificação-com-random-forest)
- [Referência de Parâmetros CLI](#referência-de-parâmetros-cli)
- [Saídas e Relatórios](#saídas-e-relatórios)
- [Estrutura do CSV de Entrada](#estrutura-do-csv-de-entrada)
- [Notas de Reprodutibilidade](#notas-de-reprodutibilidade)
- [Referências Bibliográficas](#referências-bibliográficas)
- [Licença](#licença)

---

## Visão Geral do Projeto

Este trabalho investiga a eficácia de três algoritmos clássicos de aprendizado de máquina supervisionado na detecção do malware **LokiBot** (família de *infostealers* para Windows, formato PE32) em contraste com executáveis benignos. O pipeline experimental contempla:

1. **Coleta de amostras** benignas e maliciosas (LokiBot)
2. **Submissão automatizada ao VirusTotal** para obtenção de laudos de detecção por múltiplos antivírus comerciais
3. **Extração de features** a partir dos relatórios de detecção (vetor binário de respostas de antivírus)
4. **Classificação binária** com:
   - **SVM** (Support Vector Machine) — com otimização de parâmetros (C, γ) e múltiplos kernels
   - **KNN** (K-Nearest Neighbors) — com Grid Search para K, pesos e métrica de distância
   - **Random Forest** — com Grid Search para `n_estimators`, `max_depth` e `min_samples_split`
5. **Validação cruzada K-Fold** (padrão: 10 folds) com análise de generalização (média ± desvio padrão)
6. **Geração automática de relatórios HTML** com dashboards interativos contendo matrizes de confusão, métricas detalhadas e análise de overfitting

---

## Arquitetura do Repositório

```
TCC/
├── README.md                          ← Este arquivo
├── Loki-classificator/                ← Repositório principal (com controle Git)
│   ├── LICENSE                        ← Licença MIT
│   ├── Loki.csv                       ← Dataset principal (~16 MB, 6269 amostras)
│   │
│   ├── ML/                            ← Módulo de Machine Learning
│   │   ├── knn/
│   │   │   ├── knn_classifier.py      ← Classificador KNN (K-Fold + Grid Search)
│   │   │   ├── Loki.csv               ← Cópia local do dataset para o KNN
│   │   │   ├── nothreshold/           ← Resultados sem threshold personalizado
│   │   │   ├── threshold-01/          ← Resultados com threshold 0.1
│   │   │   ├── threshold-02/          ← Resultados com threshold 0.2
│   │   │   └── threshold-03/          ← Resultados com threshold 0.3
│   │   │
│   │   ├── rf/
│   │   │   ├── rf_classifier.py       ← Classificador Random Forest (K-Fold + Grid Search)
│   │   │   ├── Loki.csv               ← Cópia local do dataset para o RF
│   │   │   ├── nothreshold/           ← Resultados sem threshold
│   │   │   └── threshold-01..03/      ← Resultados com thresholds variados
│   │   │
│   │   └── svm/
│   │       ├── requirements.txt       ← Dependências Python do projeto
│   │       ├── converter_libsvm.py    ← Conversor CSV → formato LIBSVM
│   │       ├── loki/
│   │       │   ├── Loki.libsvm        ← Dataset convertido para formato LIBSVM
│   │       │   └── DATA-org.libsvm    ← Dataset original em LIBSVM
│   │       ├── EN-US/classification/
│   │       │   ├── svmParameters.py   ← SVM com otimização de parâmetros (C, γ)
│   │       │   ├── svmKfold.py        ← SVM com validação cruzada K-Fold
│   │       │   └── svm.py             ← SVM básico
│   │       ├── PT-BR/classificacao/   ← Versão em português dos scripts SVM
│   │       ├── Antiviruses/           ← Datasets de outros antivírus (referência)
│   │       ├── globalPruned.libsvm    ← Dataset com poda de features
│   │       └── selected_features.csv  ← Features selecionadas após poda
│   │
│   ├── submissao-VirusTotal/          ← Módulo de submissão ao VirusTotal
│   │   ├── scr/
│   │   │   ├── virustotal.py          ← Classe de integração com API v2 do VirusTotal
│   │   │   ├── utils.py               ← Utilitários (conversão JSON, rate limiting)
│   │   │   └── ajuste_tempo.py        ← Controle de throttling da API
│   │   ├── benign/                    ← Amostras benignas e scripts de curadoria
│   │   │   ├── Submissao_virustotal/  ← Scripts de submissão para benignos
│   │   │   ├── excluir_denunciados.py ← Remove falsos positivos denunciados
│   │   │   ├── excluir_excedentes.py  ← Balanceamento do dataset
│   │   │   └── excluir_repetidos.py   ← Remoção de duplicatas por hash
│   │   ├── malware/                   ← Amostras de LokiBot e scripts de curadoria
│   │   │   ├── Submissao_virustotal/  ← Scripts de submissão para malware
│   │   │   ├── Ranking_Antivirus/     ← Ranking de detecção por antivírus
│   │   │   └── virustotal/            ← Respostas JSON do VirusTotal
│   │   ├── subir_github_init.py       ← Inicialização do repositório remoto
│   │   └── subir_github_use.py        ← Upload automatizado de amostras
│   │
│   └── imgs/TCC-imgs/                 ← Imagens dos relatórios gerados
│       ├── knn/                       ← Capturas de tela dos resultados KNN
│       ├── rf/                        ← Capturas de tela dos resultados RF
│       └── svm/                       ← Capturas de tela dos resultados SVM
│
├── Code/                              ← Versões anteriores dos classificadores
│   ├── Loki/Loki.csv                  ← Versão anterior do dataset
│   ├── knn_classifier.py              ← Versão anterior do KNN
│   └── rf_classifier.py               ← Versão anterior do RF
│
├── CLASSIFICADORES/                   ← Documentação auxiliar e comandos rápidos
│   ├── how-to-run-knn.txt             ← Comando de execução rápida do KNN
│   ├── how-to-run-rf.txt              ← Comando de execução rápida do RF
│   ├── knn_classifier.py.txt          ← Backup do código-fonte KNN
│   └── rf_classifier.py.txt           ← Backup do código-fonte RF
│
└── CAP 5 MALWARE [23-03-2026].txt     ← Nota de contexto do experimento
```

---

## Pré-requisitos

| Dependência       | Versão Requerida  | Verificação                    |
| :---------------- | :---------------- | :----------------------------- |
| **Python**        | ≥ 3.8             | `python --version`             |
| **pip**           | ≥ 21.0            | `pip --version`                |
| **Git**           | ≥ 2.30 (opcional) | `git --version`                |
| **Sistema Operacional** | Windows 10+, Linux ou macOS | — |

### Dependências Python

As dependências são gerenciadas pelo arquivo `requirements.txt` localizado em `Loki-classificator/ML/svm/`:

| Pacote            | Versão   | Finalidade                                           |
| :---------------- | :------- | :--------------------------------------------------- |
| `scikit-learn`    | 1.6.1    | Algoritmos de ML (SVM, KNN, RF), métricas, K-Fold    |
| `libsvm`          | 3.23.0.4 | Backend nativo para classificação SVM                |
| `matplotlib`      | latest   | Geração de gráficos e matrizes de confusão           |
| `seaborn`         | latest   | Heatmaps estilizados para matrizes de confusão (RF)  |
| `jinja2`          | 3.1.6    | Templates HTML para relatórios do SVM                |
| `brisque`         | 0.0.16   | Métricas de qualidade de imagem (utilidade auxiliar)  |
| `pandas`          | latest   | Leitura e manipulação do dataset CSV                 |
| `numpy`           | latest   | Operações numéricas vetorizadas                      |

> **Nota:** `pandas` e `numpy` são dependências transitivas do `scikit-learn`, mas são importadas diretamente nos scripts dos classificadores KNN e RF.

---

## Instalação e Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/Loki-classificator.git
cd Loki-classificator
```

Ou, se trabalhando a partir do diretório `TCC/` já existente:

```bash
cd TCC/Loki-classificator
```

### 2. Criar e ativar o ambiente virtual

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r ML/svm/requirements.txt
```

Para instalar dependências adicionais necessárias para os classificadores KNN e RF (caso não cobertas pelo `requirements.txt`):

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4. Verificar a instalação

```bash
python -c "import sklearn; import pandas; import numpy; import matplotlib; print('Todas as dependências instaladas com sucesso.')"
```

---

## Dataset

O dataset principal é o arquivo **`Loki.csv`** (~16 MB), que contém:

| Propriedade                | Valor                              |
| :------------------------- | :--------------------------------- |
| **Total de amostras**      | 6.268 (excluindo cabeçalho)        |
| **Formato**                | CSV (separador `;`)                |
| **Features**               | ~1.400+ colunas binárias           |
| **Coluna alvo (label)**    | Última coluna (`class (0:benign, 1:malware)`) |
| **Classes**                | `0` = Benigno, `1` = LokiBot (malware) |
| **Representação**          | Vetor binário de detecção por antivírus comerciais |

Cada linha corresponde a uma amostra PE32 (executável Windows). A primeira coluna contém o hash SHA-256 do executável, seguido de colunas binárias indicando se cada antivírus comercial (via VirusTotal) classificou a amostra como maliciosa (`1`) ou benigna (`0`). A última coluna é o rótulo de verdade absoluta (*ground truth*).

O arquivo `Loki.csv` está presente em múltiplas localizações para conveniência de cada classificador:

- `Loki-classificator/Loki.csv` (raiz)
- `Loki-classificator/ML/knn/Loki.csv`
- `Loki-classificator/ML/rf/Loki.csv`

Para o classificador **SVM**, o dataset precisa ser convertido para o formato **LIBSVM** (veja [Etapa 2](#etapa-2--classificação-com-svm)).

---

## Pipeline Experimental

### Etapa 1 — Coleta e Submissão ao VirusTotal

> ⚠️ **Esta etapa já foi executada e os dados resultantes estão no dataset `Loki.csv`.** Ela é documentada aqui para fins de reprodutibilidade completa.

A submissão das amostras PE32 ao VirusTotal é feita via API v2, usando os scripts em `submissao-VirusTotal/`:

```bash
cd Loki-classificator/submissao-VirusTotal
```

O módulo `scr/virustotal.py` encapsula as chamadas à API:

- **`scan(file)`** — Submete um arquivo para análise
- **`report(file)`** — Obtém o relatório de detecção (calcula o SHA-256 automaticamente)
- **`status(response)`** — Verifica o status HTTP (rate limiting, erros de autenticação)

> **Limitação da API:** A API gratuita do VirusTotal permite 4 requisições por minuto. O módulo `ajuste_tempo.py` gerencia automaticamente o *throttling*.

**Scripts de curadoria do dataset:**

| Script                       | Função                                              |
| :--------------------------- | :-------------------------------------------------- |
| `benign/excluir_denunciados.py` | Remove amostras benignas denunciadas como maliciosas |
| `benign/excluir_excedentes.py`  | Balanceia a quantidade de benignos e malware        |
| `benign/excluir_repetidos.py`   | Remove duplicatas por hash SHA-256                  |
| `malware/excluir_excedentes.py` | Balanceia a quantidade de malware e benignos        |
| `malware/excluir_repetidos.py`  | Remove duplicatas por hash SHA-256                  |

---

### Etapa 2 — Classificação com SVM

#### 2.1 Conversão do Dataset para formato LIBSVM

O SVM requer o dataset no formato LIBSVM. Use o conversor:

```bash
cd Loki-classificator/ML/svm

python converter_libsvm.py ../../Loki.csv Loki_SVM_format.libsvm
```

**Parâmetros do conversor:**

| Parâmetro      | Obrigatório | Descrição                                                      |
| :------------- | :---------: | :------------------------------------------------------------- |
| `input_file`   | ✅          | Caminho do arquivo CSV de entrada                              |
| `output_file`  | ✅          | Nome do arquivo de saída (salvo em `Antiviruses/`)             |
| `--label`      | ❌          | Nome exato da coluna alvo (padrão: `class (0:benign, 1:malware)`) |

> O arquivo convertido será salvo automaticamente na pasta `Antiviruses/`.

#### 2.2 Execução do SVM com Otimização de Parâmetros

**Execução padrão (recomendada):**

```bash
python EN-US/classification/svmParameters.py -dataset loki/Loki.libsvm
```

**Execução com poda de features (Feature Selection):**

```bash
python EN-US/classification/svmParameters.py -dataset loki/Loki.libsvm -threshold 0.15
```

| Parâmetro     | Obrigatório | Descrição                                                     |
| :------------ | :---------: | :------------------------------------------------------------ |
| `-dataset`    | ✅          | Caminho do arquivo no formato LIBSVM                         |
| `-threshold`  | ❌          | Correlação mínima para manter uma feature (ex: `0.15` = 15%) |

**Scripts SVM disponíveis:**

| Script             | Descrição                                                 |
| :----------------- | :-------------------------------------------------------- |
| `svm.py`           | Classificação SVM básica                                  |
| `svmKfold.py`      | SVM com validação cruzada K-Fold                          |
| `svmParameters.py` | SVM com estudo exaustivo de kernels e parâmetros (C, γ)   |

---

### Etapa 3 — Classificação com KNN

```bash
cd Loki-classificator/ML/knn
```

#### Execução Completa (Grid Search + K-Fold + Normalização + Verbose)

```bash
python knn_classifier.py -tall Loki.csv -kfold 10 -virusNorm -tune -v
```

#### Execução Manual (sem Grid Search)

```bash
python knn_classifier.py -tall Loki.csv -kfold 10 -k 5 -v
```

#### Execução com Threshold Personalizado

```bash
python knn_classifier.py -tall Loki.csv -kfold 10 -virusNorm -tune -threshold 0.3 -v
```

**Referência de parâmetros:**

| Parâmetro     | Tipo    | Padrão | Descrição                                                          |
| :------------ | :------ | :----- | :----------------------------------------------------------------- |
| `-tall`       | string  | —      | **(Obrigatório)** Caminho para o arquivo CSV do dataset            |
| `-kfold`      | int     | `10`   | Número de folds na validação cruzada                               |
| `-k`          | int     | `5`    | Número de vizinhos (ignorado se `-tune` estiver ativo)             |
| `-threshold`  | float   | `None` | Limiar de decisão personalizado (`0.0` a `1.0`). Se omitido, usa o padrão do algoritmo |
| `-virusNorm`  | flag    | `False`| Aplica normalização MinMaxScaler no intervalo `[0.1, 0.9]`        |
| `-tune`       | flag    | `False`| Ativa Grid Search automático para K, pesos e métrica de distância  |
| `-v`          | flag    | `False`| Modo verbose (exibe progresso detalhado no terminal)               |

**Espaço de busca do Grid Search (KNN):**

```python
{
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan']
}
```

---

### Etapa 4 — Classificação com Random Forest

```bash
cd Loki-classificator/ML/rf
```

#### Execução Completa (Grid Search + K-Fold + Verbose)

```bash
python rf_classifier.py -tall Loki.csv -kfold 10 -tune -v
```

#### Execução Manual (sem Grid Search)

```bash
python rf_classifier.py -tall Loki.csv -kfold 10 -ntree 100 -v
```

#### Execução com Threshold Personalizado

```bash
python rf_classifier.py -tall Loki.csv -kfold 10 -tune -threshold 0.3 -v
```

**Referência de parâmetros:**

| Parâmetro     | Tipo    | Padrão | Descrição                                                              |
| :------------ | :------ | :----- | :--------------------------------------------------------------------- |
| `-tall`       | string  | —      | **(Obrigatório)** Caminho para o arquivo CSV do dataset                |
| `-kfold`      | int     | `10`   | Número de folds na validação cruzada                                   |
| `-ntree`      | int     | `100`  | Número de árvores (ignorado se `-tune` estiver ativo)                  |
| `-threshold`  | float   | `None` | Limiar de decisão personalizado (`0.0` a `1.0`)                        |
| `-virusNorm`  | flag    | `False`| Aplica normalização MinMaxScaler no intervalo `[0.1, 0.9]`             |
| `-tune`       | flag    | `False`| Ativa Grid Search automático para `n_estimators`, `max_depth`, `min_samples_split` |
| `-v`          | flag    | `False`| Modo verbose                                                           |

**Espaço de busca do Grid Search (Random Forest):**

```python
{
    'n_estimators':     [50, 100, 200],
    'max_depth':        [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
```

---

## Referência de Parâmetros CLI

### Resumo Comparativo dos Três Classificadores

| Funcionalidade           | SVM (`svmParameters.py`)    | KNN (`knn_classifier.py`)    | RF (`rf_classifier.py`)      |
| :----------------------- | :-------------------------- | :--------------------------- | :--------------------------- |
| **Formato de entrada**   | `.libsvm`                   | `.csv`                       | `.csv`                       |
| **Validação cruzada**    | K-Fold (interno)            | K-Fold (`-kfold`)            | K-Fold (`-kfold`)            |
| **Grid Search**          | Parâmetros (C, γ)           | K, pesos, distância (`-tune`)| n_estimators, depth (`-tune`)|
| **Threshold ajustável**  | Via poda (`-threshold`)     | Decisão (`-threshold`)       | Decisão (`-threshold`)       |
| **Normalização**         | —                           | MinMaxScaler (`-virusNorm`)  | MinMaxScaler (`-virusNorm`)  |
| **Relatório HTML**       | ✅ Dashboard com heatmaps   | ✅ Dashboard interativo       | ✅ Dashboard + Feature Importance |
| **`random_state`**       | —                           | `42` (K-Fold)                | `42` (RF + K-Fold)           |

---

## Saídas e Relatórios

Cada execução gera automaticamente um **relatório HTML** no diretório de trabalho atual, com nomenclatura padronizada:

```
<ALGORITMO>_TCC_Lokibot_Report_<YYYYMMDD>_<HHMMSS>.html
```

**Exemplos:**
- `KNN_TCC_Lokibot_Report_20260528_163029.html`
- `RF_TCC_Lokibot_Report_20260505_181514.html`

### Conteúdo do Relatório HTML

| Seção                                      | Descrição                                                    |
| :----------------------------------------- | :----------------------------------------------------------- |
| **Melhor Fold**                             | Métricas + Matriz de Confusão do fold com maior F1-Score     |
| **Pior Fold**                               | Métricas + Matriz de Confusão do fold com menor F1-Score     |
| **Configurações do Experimento**            | Base utilizada, distribuição de classes, normalização, hiperparâmetros |
| **Análise de Generalização**                | Acurácia e F1-Score (média ± desvio padrão) para treino e teste |
| **Relatório de Classificação**              | `classification_report` do Scikit-Learn (precisão, recall, F1 por classe) |
| **Resumo Detalhado por Fold**               | Tabela com acurácia treino/teste, F1-Score e tempo por fold  |
| **Feature Importance** *(apenas RF)*        | Gráfico de barras com as 15 features mais relevantes         |

---

## Estrutura do CSV de Entrada

O dataset `Loki.csv` utiliza o seguinte esquema:

```
<hash_SHA256>.exe_repor ; AV_1 ; AV_2 ; ... ; AV_N ; class
```

| Coluna                           | Tipo      | Descrição                                       |
| :------------------------------- | :-------- | :---------------------------------------------- |
| Coluna 0 (identificador)        | `string`  | Hash SHA-256 do executável + sufixo `_repor`    |
| Colunas 1 a N (features)        | `int` (0/1) | Detecção binária por cada antivírus do VirusTotal |
| Última coluna (label)           | `int`     | `0` = benigno, `1` = LokiBot (malware)         |

**Detecção automática da coluna alvo:** Os scripts KNN e RF detectam automaticamente a coluna de rótulos procurando colunas binárias com proporção balanceada (≥ 40% da classe minoritária).

**Binarização de labels:** Os valores `0`, `0.0`, `-1`, `-1.0`, `benign`, `benigno`, `normal` são mapeados para `0` (benigno). Qualquer outro valor é mapeado para `1` (malware).

---

## Notas de Reprodutibilidade

### Determinismo dos Experimentos

| Componente              | Mecanismo de Controle                                |
| :---------------------- | :--------------------------------------------------- |
| **K-Fold split**        | `random_state=42`, `shuffle=True`                    |
| **Random Forest**       | `random_state=42`                                    |
| **Grid Search (CV)**    | `cv=5` folds internos, `scoring='f1'`                |
| **Paralelismo**         | `n_jobs=-1` (todos os cores disponíveis)             |

### Reprodução Completa Passo a Passo

```bash
# 1. Configurar o ambiente
cd TCC/Loki-classificator
python -m venv venv
source venv/bin/activate          # Linux/macOS
# .\venv\Scripts\Activate.ps1    # Windows PowerShell

# 2. Instalar dependências
pip install -r ML/svm/requirements.txt
pip install pandas numpy

# 3. Executar KNN (com Grid Search + Normalização + 10-Fold)
cd ML/knn
python knn_classifier.py -tall Loki.csv -kfold 10 -virusNorm -tune -v

# 4. Executar Random Forest (com Grid Search + 10-Fold)
cd ../rf
python rf_classifier.py -tall Loki.csv -kfold 10 -tune -v

# 5. Executar SVM
cd ../svm
python converter_libsvm.py ../../Loki.csv Loki_converted.libsvm
python EN-US/classification/svmParameters.py -dataset loki/Loki.libsvm

# 6. Os relatórios HTML serão gerados no diretório de cada classificador
```

### Versões do Ambiente de Desenvolvimento Original

```
Python 3.x
scikit-learn==1.6.1
libsvm==3.23.0.4
jinja2==3.1.6
matplotlib (latest)
seaborn (latest)
```

> **Dica:** Para garantir reprodutibilidade exata, congele as versões com `pip freeze > requirements_full.txt` após instalar as dependências.

---

## Referências Bibliográficas

Este trabalho se baseia na metodologia e nas ferramentas desenvolvidas pelo grupo de pesquisa **DejavuForensics** (UFPE). Os seguintes artigos fundamentam o pipeline experimental:

1. **Tavares-Silva, S.H.M., Lopes-Lima, S.M., et al.** Antivirus solution to IoT malware detection with authorial next-generation sandbox. *The Journal of Supercomputing*, 81, 151 (2025). [DOI: 10.1007/s11227-024-06506-x](https://doi.org/10.1007/s11227-024-06506-x)

2. **Santos, C.H.M., Lima, S.M.L.** XAI-driven antivirus in pattern identification of citadel malware. *Journal of Computational Science*, 82, 102389 (2024). [DOI: 10.1016/j.jocs.2024.102389](https://doi.org/10.1016/j.jocs.2024.102389)

3. **Pinheiro, R.P., Lima, S.M.L., et al.** Antivirus applied to JAR malware detection based on runtime behaviors. *Scientific Reports - Nature Research*, 12, 1945 (2022). [DOI: 10.1038/s41598-022-05921-5](https://doi.org/10.1038/s41598-022-05921-5)

4. **Lima, S.M.L., et al.** Next-generation antivirus for JavaScript malware detection based on dynamic features. *Knowledge and Information Systems*, 66, 1337–1370 (2024). [DOI: 10.1007/s10115-023-01978-4](https://doi.org/10.1007/s10115-023-01978-4)

5. **Pereira, G.L., Brito, L.S., Lima, S.M.L.** Antivirus applied to Google Chrome's extension malware. *Computers & Security*, 156, 104465 (2025). [DOI: 10.1016/j.cose.2025.104465](https://doi.org/10.1016/j.cose.2025.104465)

---

## Licença

Este projeto está licenciado sob a **MIT License**. Consulte o arquivo [LICENSE](Loki-classificator/LICENSE) para detalhes.

```
MIT License — Copyright (c) 2026 Paulo Souza
```
