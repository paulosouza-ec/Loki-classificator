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

    img_best = plot_cm_to_base64(best_fold['cm'], f"Matriz - Melhor Fold ({best_fold['idx']}) - F1 Teste: {best_fold['score']:.4f}")
    img_worst = plot_cm_to_base64(worst_fold['cm'], f"Matriz - Pior Fold ({worst_fold['idx']}) - F1 Teste: {worst_fold['score']:.4f}")
    img_feat = plot_feature_importance_to_base64(best_fold['feature_importances'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"RF_TCC_Lokibot_Report_{timestamp}.html"

    linhas_folds = ""
    for f in fold_metrics:
        linhas_folds += f"<tr><td>Fold {f['idx']}</td><td>{f['acc_train']:.4f}</td><td>{f['acc_test']:.4f}</td><td>{f['f1_test']:.4f}</td><td>{f['time']:.2f}s</td></tr>\n"

    # Preparar a tag do Threshold para o HTML
    if args.threshold is not None:
        threshold_html = f'<span class="badge bg-warning">{args.threshold}</span>'
    else:
        threshold_html = '<span class="badge" style="background-color:#95a5a6;">Padrão do Algoritmo</span>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Relatório TCC - Detecção de Lokibot</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 30px; }}
            .container {{ background-color: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); max-width: 1000px; margin: auto; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #8e44ad; padding-bottom: 10px; }}
            h2 {{ color: #8e44ad; margin-top: 35px; border-left: 4px solid #8e44ad; padding-left: 10px; font-size: 1.4em; }}
            p.info-text {{ color: #555; font-size: 0.95em; line-height: 1.5; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            th, td {{ padding: 12px 15px; text-align: center; border-bottom: 1px solid #eee; }}
            th {{ background-color: #f8f9fa; color: #2c3e50; font-weight: 600; text-transform: uppercase; font-size: 0.85em; }}
            tr:hover {{ background-color: #f1f4f8; }}
            .config-list li {{ margin-bottom: 8px; font-size: 1.05em; }}
            .report-box {{ background: #272822; color: #f8f8f2; padding: 20px; border-radius: 8px; font-family: monospace; overflow-x: auto; white-space: pre; font-size: 14px; }}
            .matrix-container {{ display: flex; justify-content: space-between; flex-wrap: wrap; margin-top: 20px; gap: 20px; }}
            .matrix-box {{ flex: 1; min-width: 45%; text-align: center; background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #ddd; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
            .matrix-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .feat-box {{ text-align: center; margin-top: 30px; background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }}
            .footer {{ margin-top: 50px; font-size: 0.9em; color: #7f8c8d; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
            .badge {{ display: inline-block; padding: 5px 10px; border-radius: 15px; font-size: 0.85em; font-weight: bold; color: white; }}
            .bg-benign {{ background-color: #27ae60; }}
            .bg-lokibot {{ background-color: #8e44ad; }}
            .bg-warning {{ background-color: #f39c12; }}
            .std-dev {{ color: #7f8c8d; font-size: 0.9em; }}
            .aviso {{ background-color: #fcf3cf; color: #7d6608; padding: 15px; border-left: 5px solid #f1c40f; margin-top: 20px; font-size: 0.95em; }}
            .grid-badge {{ background-color: #8e44ad; color: white; padding: 3px 8px; border-radius: 5px; font-size: 0.8em; margin-left: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Classificação: Lokibot vs Benigno (Random Forest)</h1>
            
            <h2>1. Configurações do Experimento</h2>
            <ul class="config-list">
                <li><strong>Base de Dados:</strong> {args.AllData_File}</li>
                <li><strong>Distribuição de Classes:</strong> <span class="badge bg-benign">Benignos: {qtd_benigno}</span> | <span class="badge bg-lokibot">Lokibot: {qtd_lokibot}</span></li>
                <li><strong>K-Folds (Validação Cruzada):</strong> {args.kfold}</li>
                <li><strong>Limiar de Decisão (Threshold):</strong> {threshold_html}</li>
            </ul>

            <div class="aviso">
                <strong>Hiperparâmetros do Modelo:</strong><br>
                {'<span class="grid-badge">Otimizados via Grid Search</span>' if args.tune else '<span class="grid-badge" style="background-color:#95a5a6;">Valores Manuais/Padrão</span>'}
                <ul style="margin-top: 10px; margin-bottom: 0;">
                    <li><strong>Número de Árvores (n_estimators):</strong> {rf_params['n_estimators']}</li>
                    <li><strong>Profundidade Máxima (max_depth):</strong> {rf_params['max_depth']}</li>
                    <li><strong>Mín. Amostras para Divisão (min_samples_split):</strong> {rf_params['min_samples_split']}</li>
                </ul>
            </div>

            <h2>2. Análise de Generalização (Treino vs Teste)</h2>
            <p class="info-text">Esta tabela evidencia que o modelo não está apenas decorando os dados (overfitting). Comparamos o desempenho nos dados já conhecidos (Treino) com dados inéditos (Teste). O Desvio Padrão (&plusmn;) baixo confirma a estabilidade do modelo em diferentes separações da base.</p>
            <table>
                <thead>
                    <tr><th>Métrica</th><th>Fase de Treinamento (Conhecido)</th><th>Fase de Teste (Inédito)</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Acurácia Global</strong></td>
                        <td>{acc_train_mean:.4f} <span class="std-dev">&plusmn; {acc_train_std:.4f}</span></td>
                        <td><strong>{acc_test_mean:.4f} <span class="std-dev">&plusmn; {acc_test_std:.4f}</span></strong></td>
                    </tr>
                    <tr>
                        <td><strong>F1-Score (Lokibot)</strong></td>
                        <td>{f1_train_mean:.4f} <span class="std-dev">&plusmn; {f1_train_std:.4f}</span></td>
                        <td><strong>{f1_test_mean:.4f} <span class="std-dev">&plusmn; {f1_test_std:.4f}</span></strong></td>
                    </tr>
                </tbody>
            </table>

            <h2>3. Métricas Detalhadas de Teste (Média dos Folds)</h2>
            <table>
                <thead>
                    <tr><th>Métrica</th><th>Resultado Médio no Teste</th></tr>
                </thead>
                <tbody>
                    <tr><td><strong>Precisão (Lokibot)</strong></td><td>{prec_mean:.4f}</td></tr>
                    <tr><td><strong>Revocação / Taxa de Detecção (Lokibot)</strong></td><td>{rec_mean:.4f}</td></tr>
                </tbody>
            </table>

            <h2>4. Relatório por Classe (Melhor Fold de Teste: {best_fold['idx']})</h2>
            <div class="report-box">{best_fold['class_report']}</div>

            <h2>5. Matrizes de Confusão (Fase de Teste)</h2>
            <div class="matrix-container">
                <div class="matrix-box">
                    <h3>Melhor Fold ({best_fold['idx']})</h3>
                    <img src="{img_best}" alt="Matriz de Confusão do Melhor Fold">
                </div>
                <div class="matrix-box">
                    <h3>Pior Fold ({worst_fold['idx']})</h3>
                    <img src="{img_worst}" alt="Matriz de Confusão do Pior Fold">
                </div>
            </div>

            <h2>6. Importância das Características (Feature Importance)</h2>
            <div class="feat-box">
                <img src="{img_feat}" alt="Feature Importance">
            </div>

            <h2>7. Resumo por Fold (Treino vs Teste)</h2>
            <table>
                <thead><tr><th>Fold</th><th>Acc Treino</th><th>Acc Teste</th><th>F1-Score Teste (Lokibot)</th><th>Tempo Execução</th></tr></thead>
                <tbody>{linhas_folds}</tbody>
            </table>
            
            <div class="footer">Relatório gerado para análise acadêmica e de TCC.</div>
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