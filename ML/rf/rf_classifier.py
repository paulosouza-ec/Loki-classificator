import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, ConfusionMatrixDisplay, 
                             classification_report)
import sys
import os
import time
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import io
import base64

# Ignorar os avisos do Scikit-Learn no terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def plot_cm_to_base64(cm, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    display_labels = ['Benigno (0)', 'Lokibot (1)']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='g', colorbar=True)
    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Classe Verdadeira (Real)', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def plot_feature_importance_to_base64(importances, title="Top 15 Características mais Importantes"):
    indices = np.argsort(importances)[::-1][:15]
    top_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(top_importances)), top_importances, align="center", color="#8e44ad")
    ax.set_xticks(range(len(top_importances)))
    ax.set_xticklabels([f"Feature {i}" for i in indices], rotation=45, ha='right')
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Nível de Importância", fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def gerar_relatorio_html(args, fold_metrics, best_fold, worst_fold, total_time, qtd_benigno, qtd_lokibot, rf_params):
    # Calculando Médias e Desvios Padrão para Teste
    acc_test_mean = np.mean([f['acc_test'] for f in fold_metrics])
    acc_test_std = np.std([f['acc_test'] for f in fold_metrics])
    f1_test_mean = np.mean([f['f1_test'] for f in fold_metrics])
    f1_test_std = np.std([f['f1_test'] for f in fold_metrics])
    prec_mean = np.mean([f['prec_test'] for f in fold_metrics])
    rec_mean = np.mean([f['rec_test'] for f in fold_metrics])

    # Calculando Médias e Desvios Padrão para Treino (Análise de Overfitting)
    acc_train_mean = np.mean([f['acc_train'] for f in fold_metrics])
    acc_train_std = np.std([f['acc_train'] for f in fold_metrics])
    f1_train_mean = np.mean([f['f1_train'] for f in fold_metrics])
    f1_train_std = np.std([f['f1_train'] for f in fold_metrics])

    best_fm = next(f for f in fold_metrics if f['idx'] == best_fold['idx'])
    worst_fm = next(f for f in fold_metrics if f['idx'] == worst_fold['idx'])

    img_best = plot_cm_to_base64(best_fold['cm'], f"Matriz - Melhor Fold ({best_fold['idx']}) - F1 Teste: {best_fold['score']:.4f}")
    img_worst = plot_cm_to_base64(worst_fold['cm'], f"Matriz - Pior Fold ({worst_fold['idx']}) - F1 Teste: {worst_fold['score']:.4f}")
    img_feat = plot_feature_importance_to_base64(best_fold['feature_importances'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"RF_TCC_Lokibot_Report_{timestamp}.html"

    # Preparar variáveis para o HTML
    threshold_val = f"{args.threshold}" if args.threshold is not None else "Padrão do Algoritmo"
    is_tuned = "Sim (Grid Search)" if args.tune else "Não (Valores Manuais)"

    linhas_folds = ""
    for f in fold_metrics:
        linhas_folds += f"<tr style='border-bottom: 1px solid rgba(0,0,0,0.05);'><td style='padding:10px; font-weight:600;'>Fold {f['idx']}</td><td>{f['acc_train']*100:.2f}%</td><td>{f['acc_test']*100:.2f}%</td><td>{f['f1_test']:.4f}</td><td>{f['time']:.2f}s</td></tr>\n"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Dashboard Random Forest - Relatório de Resultados</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif, Arial; background: linear-gradient(135deg, #8B1538 0%, #A91E4A 50%, #6B1429 100%); min-height: 100vh; color: #333; }}
            .dashboard-container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
            .header {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; margin-bottom: 30px; text-align: center; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
            .header h1 {{ font-size: 2.5em; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 10px; }}
            .subtitle {{ font-size: 1.2em; color: #666; font-weight: 300; }}
            .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 15px; padding: 25px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; }}
            .stat-card:hover {{ transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15); }}
            .stat-card.best {{ border-left: 10px solid #A5D7A7; }} .stat-card.worst {{ border-left: 10px solid #f9a19a; }}
            .card-header {{ display: flex; align-items: center; margin-bottom: 20px; }}
            .card-icon {{ width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 1.5em; font-weight: bold; color: white; }}
            .card-icon.best {{ background: linear-gradient(45deg, #4CAF50, #66BB6A); }}
            .card-icon.worst {{ background: linear-gradient(45deg, #f44336, #EF5350); }}
            .card-icon.info {{ background: linear-gradient(45deg, #8B1538, #A91E4A); }}
            .card-title {{ font-size: 1.3em; font-weight: 600; }}
            .metric-row {{ display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(0, 0, 0, 0.05); }}
            .metric-row:last-child {{ border-bottom: none; }}
            .metric-label {{ font-weight: 500; color: #555; }}
            .metric-value {{ font-weight: 600; padding: 4px 12px; border-radius: 20px; text-align: right; }}
            .metric-value.best {{ background: rgba(76, 175, 80, 0.1); }}
            .metric-value.worst {{ background: rgba(244, 67, 54, 0.1); }}
            .metric-value.neutral {{ background: rgba(0, 0, 0, 0.05); }}
            .cm-container {{ text-align: center; margin-top: 20px; }}
            .cm-image {{ max-width: 85%; height: auto; border-radius: 5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }}
            .kernels-section {{ background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 30px; margin-bottom:30px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); }}
            .section-title {{ font-size: 1.8em; margin-bottom: 20px; font-weight: 600; background: linear-gradient(45deg, #8B1538, #A91E4A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
            .logo-ufpe {{ height: 150px; width: auto; margin-bottom: 15px; }}
            .custom-table {{ width: 100%; border-collapse: collapse; text-align: center; margin-top: 15px; }}
            .custom-table th {{ background: rgba(0,0,0,0.05); padding: 12px; font-weight: 600; color: #333; border-radius: 5px; }}
            .custom-table td {{ padding: 10px; border-bottom: 1px solid rgba(0,0,0,0.05); }}
            .report-box {{ background: #f8f9fa; color: #333; padding: 20px; border-radius: 8px; font-family: monospace; white-space: pre; font-size: 14px; border: 1px solid rgba(0,0,0,0.1); overflow-x:auto; }}
            @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            .stat-card, .kernels-section {{ animation: fadeInUp 0.6s ease forwards; }}
        </style>
    </head>
    <body>
        <div class="dashboard-container">
            <div class="header">
                <img src="../../src/ufpe_logo.png" alt="University Logo" class="logo-ufpe" onerror="this.style.display='none'">
                <h1>Random Forest - Avaliação de Modelo</h1>
                <p class="subtitle">Detecção de Lokibot vs Benigno - Validação Cruzada K-Fold</p>
            </div>

            <div class="stats-grid">
                <div class="stat-card best">
                    <div class="card-header"><div class="card-icon best">🏆</div><div class="card-title">Melhor Fold (Fold {best_fold['idx']})</div></div>
                    <div class="metric-row"><span class="metric-label">Acurácia Treino</span><span class="metric-value best">{best_fm['acc_train']*100:.2f}%</span></div>
                    <div class="metric-row"><span class="metric-label">Acurácia Teste</span><span class="metric-value best">{best_fm['acc_test']*100:.2f}%</span></div>
                    <div class="metric-row"><span class="metric-label">F1-Score (Lokibot)</span><span class="metric-value best">{best_fm['f1_test']:.4f}</span></div>
                    <div class="metric-row"><span class="metric-label">Tempo Execução</span><span class="metric-value best">{best_fm['time']:.4f}s</span></div>
                    <div class="cm-container"><img class="cm-image" src="{img_best}" alt="Matriz Melhor Fold"></div>
                </div>

                <div class="stat-card worst">
                    <div class="card-header"><div class="card-icon worst">💎</div><div class="card-title">Pior Fold (Fold {worst_fold['idx']})</div></div>
                    <div class="metric-row"><span class="metric-label">Acurácia Treino</span><span class="metric-value worst">{worst_fm['acc_train']*100:.2f}%</span></div>
                    <div class="metric-row"><span class="metric-label">Acurácia Teste</span><span class="metric-value worst">{worst_fm['acc_test']*100:.2f}%</span></div>
                    <div class="metric-row"><span class="metric-label">F1-Score (Lokibot)</span><span class="metric-value worst">{worst_fm['f1_test']:.4f}</span></div>
                    <div class="metric-row"><span class="metric-label">Tempo Execução</span><span class="metric-value worst">{worst_fm['time']:.4f}s</span></div>
                    <div class="cm-container"><img class="cm-image" src="{img_worst}" alt="Matriz Pior Fold"></div>
                </div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">⚙️ Configurações do Experimento</h2>
                <div class="stats-grid" style="margin-bottom: 0;">
                    <div class="stat-card" style="border-left: 5px solid #8B1538; padding: 20px;">
                        <div class="card-header" style="margin-bottom: 10px;"><div class="card-icon info" style="width: 40px; height: 40px; font-size:1.1em;">📊</div><div class="card-title" style="font-size:1.1em;">Dados da Base</div></div>
                        <div class="metric-row"><span class="metric-label">Base Utilizada</span><span class="metric-value neutral">{os.path.basename(args.AllData_File)}</span></div>
                        <div class="metric-row"><span class="metric-label">Benignos</span><span class="metric-value neutral">{qtd_benigno}</span></div>
                        <div class="metric-row"><span class="metric-label">Lokibot</span><span class="metric-value neutral">{qtd_lokibot}</span></div>
                        <div class="metric-row"><span class="metric-label">Validação (K-Folds)</span><span class="metric-value neutral">{args.kfold}</span></div>
                        <div class="metric-row"><span class="metric-label">Threshold</span><span class="metric-value neutral">{threshold_val}</span></div>
                    </div>
                    <div class="stat-card" style="border-left: 5px solid #A91E4A; padding: 20px;">
                        <div class="card-header" style="margin-bottom: 10px;"><div class="card-icon info" style="width: 40px; height: 40px; font-size:1.1em;">🧠</div><div class="card-title" style="font-size:1.1em;">Hiperparâmetros</div></div>
                        <div class="metric-row"><span class="metric-label">Método</span><span class="metric-value neutral">{is_tuned}</span></div>
                        <div class="metric-row"><span class="metric-label">n_estimators</span><span class="metric-value neutral">{rf_params['n_estimators']}</span></div>
                        <div class="metric-row"><span class="metric-label">max_depth</span><span class="metric-value neutral">{rf_params['max_depth']}</span></div>
                        <div class="metric-row"><span class="metric-label">min_samples_split</span><span class="metric-value neutral">{rf_params['min_samples_split']}</span></div>
                        <div class="metric-row"><span class="metric-label">Tempo Total Execução</span><span class="metric-value neutral">{total_time:.2f}s</span></div>
                    </div>
                </div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">📈 Análise de Generalização (Média ± Desvio Padrão)</h2>
                <table class="custom-table">
                    <thead>
                        <tr><th>Métrica</th><th>Fase de Treinamento (Conhecido)</th><th>Fase de Teste (Inédito)</th></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Acurácia Global</strong></td>
                            <td>{acc_train_mean*100:.2f}% &plusmn; {acc_train_std*100:.2f}%</td>
                            <td><strong style="color: #28a745;">{acc_test_mean*100:.2f}% &plusmn; {acc_test_std*100:.2f}%</strong></td>
                        </tr>
                        <tr>
                            <td><strong>F1-Score (Lokibot)</strong></td>
                            <td>{f1_train_mean:.4f} &plusmn; {f1_train_std:.4f}</td>
                            <td><strong style="color: #28a745;">{f1_test_mean:.4f} &plusmn; {f1_test_std:.4f}</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Precisão Média (Teste)</strong></td>
                            <td colspan="2">{prec_mean:.4f}</td>
                        </tr>
                        <tr>
                            <td><strong>Taxa de Detecção / Revocação Média (Teste)</strong></td>
                            <td colspan="2">{rec_mean:.4f}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">🌟 Importância das Características (Feature Importance)</h2>
                <div class="cm-container" style="margin-top:0;">
                    <img class="cm-image" src="{img_feat}" alt="Feature Importance" style="max-width: 100%;">
                </div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">📝 Relatório de Classificação (Referente ao Melhor Fold)</h2>
                <div class="report-box">{best_fold['class_report']}</div>
            </div>

            <div class="kernels-section">
                <h2 class="section-title">📋 Resumo Detalhado por Fold</h2>
                <table class="custom-table">
                    <thead>
                        <tr><th>Fold</th><th>Acurácia Treino</th><th>Acurácia Teste</th><th>F1-Score Teste</th><th>Tempo Execução</th></tr>
                    </thead>
                    <tbody>{linhas_folds}</tbody>
                </table>
            </div>

        </div>
    </body>
    </html>
    """

    with open(html_filename, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    return html_filename

def run_rf():
    parser = argparse.ArgumentParser(description='Random Forest Classifier TCC with Grid Search')
    parser.add_argument('-tall', dest='AllData_File', required=True, help="Caminho para o ficheiro CSV")
    parser.add_argument('-kfold', dest='kfold', type=int, default=10, help="Número de folds")
    parser.add_argument('-ntree', dest='n_estimators', type=int, default=100, help="N de árvores manual (ignorado se usar -tune)")
    parser.add_argument('-threshold', dest='threshold', type=float, default=None, help="Limiar de decisão personalizado (opcional)")
    parser.add_argument('-virusNorm', dest='virusNorm', action='store_true', help="Normalização [0.1, 0.9]")
    parser.add_argument('-tune', dest='tune', action='store_true', help="Ativa o Grid Search Automático para os hiperparâmetros")
    parser.add_argument('-v', dest='verbose', action='store_true', help="Modo detalhado")

    args = parser.parse_args()
    tempo_inicio_global = time.time()

    # Validação do Threshold (agora só valida se foi passado)
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        print("Erro: O limiar (-threshold) deve ser um valor entre 0.0 e 1.0.")
        sys.exit(1)

    if args.verbose: print(f"[*] Carregando dados de: {args.AllData_File}")
    try:
        data = pd.read_csv(args.AllData_File, sep=None, engine='python')
    except Exception as e:
        print(f"Erro ao carregar o ficheiro: {e}")
        sys.exit(1)
    
    # Auto-detectar coluna alvo
    if args.verbose: print("[*] Procurando automaticamente a coluna de rótulos (balanceada com 2 classes)...")
    coluna_alvo = None
    for col in data.columns:
        valores_unicos = data[col].dropna().unique()
        if len(valores_unicos) == 2:
            contagem = data[col].value_counts()
            proporcao_menor_classe = contagem.min() / contagem.sum()
            if proporcao_menor_classe >= 0.40:
                coluna_alvo = col
                break

    if coluna_alvo is None:
        print("[-] ERRO: Não consegui encontrar a coluna das classes automaticamente.")
        sys.exit(1)

    if args.verbose: print(f"[+] SUCESSO: Coluna alvo detectada -> '{coluna_alvo}'")

    y_raw = data[coluna_alvo].values
    X_df = data.drop(columns=[coluna_alvo]).select_dtypes(include=[np.number])
    X = X_df.values

    if X.shape[1] == 0:
        print("Erro: Nenhuma coluna numérica encontrada nas features.")
        sys.exit(1)

    # Binarização
    y_bin = []
    for val in y_raw:
        val_str = str(val).lower().strip()
        if val_str in ['0', '0.0', '-1', '-1.0', 'benign', 'benigno', 'normal']:
            y_bin.append(0)
        else:
            y_bin.append(1)
            
    y = np.array(y_bin)
    qtd_benigno = np.sum(y == 0)
    qtd_lokibot = np.sum(y == 1)

    if args.verbose: 
        print(f"[*] Features numéricas válidas utilizadas: {X.shape[1]}")
        print(f"[*] Distribuição Final das Classes: Benignos ({qtd_benigno}) | Lokibot ({qtd_lokibot})")
        if args.threshold is not None:
            print(f"[*] Usando Threshold Personalizado: {args.threshold}")
        else:
            print(f"[*] Usando predição padrão do Random Forest (Sem threshold forçado)")

    if args.virusNorm:
        if args.verbose: print("[*] Aplicando normalização virusNorm [0.1, 0.9]...")
        scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        X = scaler.fit_transform(X)

    # =========================================================================
    # GRID SEARCH (OTIMIZAÇÃO DE HIPERPARÂMETROS DO RANDOM FOREST)
    # =========================================================================
    if args.tune:
        if args.verbose:
            print("\n" + "="*50)
            print("[*] INICIANDO GRID SEARCH PARA RANDOM FOREST...")
            print("[*] Isso pode demorar vários minutos.")
            print("="*50)
            
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        # Verbose 3 para você acompanhar o progresso como fez no KNN
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=3 if args.verbose else 0)
        grid_search.fit(X, y)
        
        melhor_n = grid_search.best_params_['n_estimators']
        melhor_depth = grid_search.best_params_['max_depth']
        melhor_split = grid_search.best_params_['min_samples_split']
        
        print("\n" + "="*50)
        print("🏆 MELHORES HIPERPARÂMETROS ENCONTRADOS 🏆")
        print("="*50)
        print(f" -> Árvores (n_estimators): {melhor_n}")
        print(f" -> Profundidade (max_depth): {melhor_depth}")
        print(f" -> Mín. Amostras Divisão:  {melhor_split}")
        print(f" -> F1-Score Est.: {grid_search.best_score_:.4f}")
        print("="*50)
        
        rf = RandomForestClassifier(n_estimators=melhor_n, max_depth=melhor_depth, min_samples_split=melhor_split, random_state=42, n_jobs=-1)
        rf_params = {'n_estimators': melhor_n, 'max_depth': str(melhor_depth), 'min_samples_split': melhor_split}
    else:
        # Modo manual padrão
        rf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
        rf_params = {'n_estimators': args.n_estimators, 'max_depth': 'None (Ilimitado)', 'min_samples_split': 2}

    # =========================================================================
    # K-FOLD CROSS VALIDATION
    # =========================================================================
    if args.verbose: 
        msg_thresh = f"Threshold de {args.threshold}" if args.threshold is not None else "Predição Padrão"
        print(f"\n[*] Iniciando K-Fold ({args.kfold} folds) com {msg_thresh}...")
    
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    
    fold_metrics = []
    best_fold = {'score': -1.0, 'cm': None, 'idx': 0, 'class_report': "", 'feature_importances': None}
    worst_fold = {'score': 2.0, 'cm': None, 'idx': 0}

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        tempo_inicio_fold = time.time()
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Treinamento
        rf.fit(X_train, y_train)
        
        # ==========================================================
        # LÓGICA DE PREDIÇÃO: PADRÃO VS THRESHOLD PERSONALIZADO
        # ==========================================================
        if args.threshold is not None:
            # Previsão TREINO com Threshold
            y_train_proba = rf.predict_proba(X_train)[:, 1]
            y_train_pred = (y_train_proba >= args.threshold).astype(int)

            # Previsão TESTE com Threshold
            y_test_proba = rf.predict_proba(X_test)[:, 1]
            y_test_pred = (y_test_proba >= args.threshold).astype(int)
        else:
            # Usa o preditor puro do Scikit-Learn
            y_train_pred = rf.predict(X_train)
            y_test_pred = rf.predict(X_test)
        
        # ==========================================================
        # CÁLCULO DE MÉTRICAS
        # ==========================================================
        acc_train = accuracy_score(y_train, y_train_pred)
        f1_train = f1_score(y_train, y_train_pred, average='binary', pos_label=1, zero_division=0)

        acc_test = accuracy_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred, average='binary', pos_label=1, zero_division=0)
        prec_test = precision_score(y_test, y_test_pred, average='binary', pos_label=1, zero_division=0)
        rec_test = recall_score(y_test, y_test_pred, average='binary', pos_label=1, zero_division=0)
        cm_test = confusion_matrix(y_test, y_test_pred, labels=[0, 1])

        tempo_fold = time.time() - tempo_inicio_fold

        fold_metrics.append({
            'idx': fold_idx + 1, 
            'acc_train': acc_train, 'f1_train': f1_train, 
            'acc_test': acc_test, 'f1_test': f1_test, 
            'prec_test': prec_test, 'rec_test': rec_test, 
            'time': tempo_fold
        })

        if args.verbose:
            print(f"    -> Fold {fold_idx + 1:02d}: Treino Acc={acc_train:.4f} | Teste Acc={acc_test:.4f} | F1-Loki(Teste)={f1_test:.4f} | Tempo: {tempo_fold:.2f}s")

        if f1_test > best_fold['score']:
            report = classification_report(y_test, y_test_pred, labels=[0, 1], target_names=['Benigno', 'Lokibot'], zero_division=0)
            best_fold = {'score': f1_test, 'cm': cm_test, 'idx': fold_idx + 1, 'class_report': report, 'feature_importances': rf.feature_importances_}
        
        if f1_test < worst_fold['score']:
            worst_fold = {'score': f1_test, 'cm': cm_test, 'idx': fold_idx + 1}

    tempo_total = time.time() - tempo_inicio_global

    acc_train_mean = np.mean([f['acc_train'] for f in fold_metrics])
    acc_train_std = np.std([f['acc_train'] for f in fold_metrics])
    acc_test_mean = np.mean([f['acc_test'] for f in fold_metrics])
    acc_test_std = np.std([f['acc_test'] for f in fold_metrics])
    f1_test_mean = np.mean([f['f1_test'] for f in fold_metrics])
    f1_test_std = np.std([f['f1_test'] for f in fold_metrics])

    threshold_str = f"Threshold: {args.threshold}" if args.threshold is not None else "Predição Padrão"
    
    print("\n" + "="*50)
    print(f"      RESULTADOS FINAIS - LOKIBOT (Random Forest - {threshold_str})")
    print("="*50)
    print(f"Acurácia Média TREINO: {acc_train_mean:.4f} (± {acc_train_std:.4f})")
    print(f"Acurácia Média TESTE:  {acc_test_mean:.4f} (± {acc_test_std:.4f})")
    print(f"F1-Score (Lokibot) Teste: {f1_test_mean:.4f} (± {f1_test_std:.4f})")
    print("="*50)

    html_file = gerar_relatorio_html(args, fold_metrics, best_fold, worst_fold, tempo_total, qtd_benigno, qtd_lokibot, rf_params)
    print(f"\n[+] Relatório HTML gerado com sucesso: {html_file}")

if __name__ == "__main__":
    run_rf()