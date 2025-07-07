import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math
from scipy import stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
import io
import warnings

# Ignorar avisos que podem poluir a saída
warnings.filterwarnings('ignore')

# --- Configuração Geral da Página ---
st.set_page_config(
    page_title="Análise de Ressalto Hidráulico",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

#======================================================================
# SEÇÃO DE FUNÇÕES (ETAPA 1 E ETAPA 2)
#======================================================================

# --- Funções Comuns e da Etapa 1 (Análise Exploratória) ---

def carregar_dados(arquivo):
    """
    Carrega os dados do arquivo CSV com delimitador ';'
    """
    try:
        df = pd.read_csv(arquivo, delimiter=';')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

def limpar_dados(df):
    """
    Realiza a limpeza dos dados para a análise exploratória (Etapa 1):
    - Remove espaços em branco dos nomes das colunas
    - Converte colunas numéricas para o tipo correto
    - Trata valores ausentes
    """
    if df is None:
        return None
    
    # Renomeando colunas para remover espaços e caracteres especiais
    df.columns = [col.strip().replace('[', '').replace(']', '') for col in df.columns]
    
    # Converte colunas numéricas para float
    colunas_numericas = ['tempo s', 'Z m', 'w m/s']
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Tratamento de valores nulos
    if df.isnull().sum().sum() > 0:
        st.warning(f"Foram encontrados {df.isnull().sum().sum()} valores nulos no conjunto de dados.")
        df = df.dropna()
        st.info(f"Depois da limpeza, o conjunto de dados tem {df.shape[0]} registros.")
    
    return df

def resumo_estatistico(df):
    """
    Calcula as estatísticas descritivas do conjunto de dados
    """
    if df is None or df.empty:
        return None
    
    # Selecionando apenas colunas numéricas
    df_num = df.select_dtypes(include=[np.number])
    
    # Estatísticas básicas
    estatisticas = df_num.describe().T
    
    # Calculando assimetria e curtose
    estatisticas['assimetria'] = df_num.skew()
    estatisticas['curtose'] = df_num.kurtosis()
    
    return estatisticas

def detectar_outliers(df, coluna, metodo='iqr'):
    """
    Detecta outliers na coluna especificada usando o método IQR ou Z-Score
    """
    if df is None or df.empty or coluna not in df.columns:
        return df, []
    
    serie = df[coluna].dropna()
    
    if metodo == 'iqr':
        Q1 = serie.quantile(0.25)
        Q3 = serie.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]
    else:  # z-score
        z_scores = np.abs(stats.zscore(serie))
        outliers = df[z_scores > 3]
    
    return outliers, len(outliers)

def calcular_correlacoes(df):
    """
    Calcula a matriz de correlação para as variáveis numéricas
    """
    if df is None or df.empty:
        return None
    
    # Seleciona apenas colunas numéricas
    df_num = df.select_dtypes(include=[np.number])
    
    # Calcula a matriz de correlação
    corr = df_num.corr()
    
    return corr

def criar_histograma(df, coluna, bins=20):
    """
    Cria um histograma para a coluna especificada
    """
    if df is None or df.empty or coluna not in df.columns:
        return None
    
    fig = px.histogram(
        df, x=coluna, 
        nbins=bins,
        title=f'Distribuição de {coluna}',
        labels={coluna: coluna},
        color_discrete_sequence=['#3366CC']
    )
    
    # Adiciona linha para média
    media = df[coluna].mean()
    fig.add_vline(x=media, line_dash="dash", line_color="red", annotation_text=f"Média: {media:.3f}")
    
    # Adiciona linha para mediana
    mediana = df[coluna].median()
    fig.add_vline(x=mediana, line_dash="dash", line_color="green", annotation_text=f"Mediana: {mediana:.3f}")
    
    fig.update_layout(
        xaxis_title=coluna,
        yaxis_title='Frequência',
        bargap=0.1
    )
    
    return fig

def criar_boxplot(df, coluna):
    """
    Cria um boxplot para a coluna especificada
    """
    if df is None or df.empty or coluna not in df.columns:
        return None
    
    fig = px.box(
        df, y=coluna,
        title=f'Boxplot de {coluna}',
        labels={coluna: coluna},
        color_discrete_sequence=['#3366CC']
    )
    
    fig.update_layout(
        yaxis_title=coluna
    )
    
    return fig

def criar_scatter(df, coluna_x, coluna_y):
    """
    Cria um gráfico de dispersão (scatter plot) entre duas variáveis
    """
    if df is None or df.empty or coluna_x not in df.columns or coluna_y not in df.columns:
        return None
    
    fig = px.scatter(
        df, x=coluna_x, y=coluna_y,
        title=f'Relação entre {coluna_x} e {coluna_y}',
        labels={coluna_x: coluna_x, coluna_y: coluna_y},
        color_discrete_sequence=['#3366CC']
    )
    
    # Calculando a linha de tendência
    try:
        x = df[coluna_x]
        y = df[coluna_y]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(mask) > 1:  # Precisa de pelo menos 2 pontos para calcular a regressão
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
            y_pred = intercept + slope * x
            fig.add_traces(
                go.Scatter(
                    x=x, y=y_pred,
                    mode='lines',
                    name=f'Linha de tendência (R²={r_value**2:.3f})',
                    line=dict(color='red', dash='dash')
                )
            )
    except Exception as e:
        st.warning(f"Não foi possível calcular a linha de tendência: {e}")
    
    fig.update_layout(
        xaxis_title=coluna_x,
        yaxis_title=coluna_y
    )
    
    return fig

def criar_heatmap_correlacao(df):
    """
    Cria um mapa de calor (heatmap) da matriz de correlação
    """
    if df is None or df.empty:
        return None
    
    corr = calcular_correlacoes(df)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Correlação'),
        text=np.round(corr.values, 2),
        texttemplate='%{text:.2f}'
    ))
    
    fig.update_layout(
        title='Matriz de Correlação',
        height=500,
        width=500
    )
    
    return fig

def criar_grafico_linha_tempo(df, coluna_y):
    """
    Cria um gráfico de linha para a evolução da variável ao longo do tempo
    """
    if df is None or df.empty or 'tempo s' not in df.columns or coluna_y not in df.columns:
        return None
    
    fig = px.line(
        df, x='tempo s', y=coluna_y,
        title=f'Evolução de {coluna_y} ao longo do tempo',
        labels={'tempo s': 'Tempo (s)', coluna_y: coluna_y},
        line_shape='linear'
    )
    
    fig.update_layout(
        xaxis_title='Tempo (s)',
        yaxis_title=coluna_y
    )
    
    return fig

def analise_estatistica_robusta(df, coluna):
    """
    Realiza análise estatística robusta para a coluna especificada
    """
    if df is None or df.empty or coluna not in df.columns:
        return {}
    
    dados = df[coluna].dropna()
    
    # Testes de normalidade
    # Shapiro-Wilk teste (para amostras pequenas)
    if len(dados) < 5000:
        shapiro_test = stats.shapiro(dados)
        shapiro_p_valor = shapiro_test.pvalue
    else:
        shapiro_p_valor = None
    
    # Teste Kolmogorov-Smirnov (para amostras maiores)
    ks_test = stats.kstest(dados, 'norm', args=(dados.mean(), dados.std()))
    ks_p_valor = ks_test.pvalue
    
    # Estatísticas descritivas
    descritivas = {
        'média': dados.mean(),
        'mediana': dados.median(),
        'desvio_padrão': dados.std(),
        'min': dados.min(),
        'max': dados.max(),
        'assimetria': stats.skew(dados),
        'curtose': stats.kurtosis(dados),
        'shapiro_p_valor': shapiro_p_valor,
        'ks_p_valor': ks_p_valor,
        'é_normal': (shapiro_p_valor > 0.05 if shapiro_p_valor is not None else None) or ks_p_valor > 0.05
    }
    
    return descritivas

# --- Funções da Etapa 2 (Modelagem) ---

def preparar_dados_modelagem(df):
    """
    Prepara os dados para modelagem, criando variáveis derivadas (Etapa 2)
    """
    if df is None:
        return None
    
    # Limpeza básica
    df.columns = [col.strip().replace('[', '').replace(']', '') for col in df.columns]
    
    # Converte colunas numéricas
    colunas_numericas = ['tempo s', 'Z m', 'w m/s']
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove valores nulos
    df = df.dropna()
    
    # Criar variáveis derivadas para modelagem
    df['tempo_quadrado'] = df['tempo s'] ** 2
    df['Z_quadrado'] = df['Z m'] ** 2
    df['w_quadrado'] = df['w m/s'] ** 2
    df['tempo_Z'] = df['tempo s'] * df['Z m']
    df['tempo_w'] = df['tempo s'] * df['w m/s']
    df['Z_w'] = df['Z m'] * df['w m/s']
    
    # Extrair posição do arquivo de origem
    if 'arquivo_origem' in df.columns:
        df['posicao'] = df['arquivo_origem'].str.extract(r'x(\d+\.?\d*)').astype(float)
    
    return df

def modelo_linear_multipla(df, variavel_dependente, variaveis_independentes):
    """
    Ajusta um modelo de regressão linear múltipla
    """
    if df is None or df.empty:
        return None, None
    
    # Preparar dados
    X = df[variaveis_independentes]
    y = df[variavel_dependente]
    
    # Adicionar constante (intercepto)
    X = sm.add_constant(X)
    
    # Ajustar modelo
    modelo = sm.OLS(y, X).fit()
    
    return modelo, X

def modelo_nao_linear(df, variavel_dependente, variaveis_independentes, tipo='polinomial'):
    """
    Ajusta modelos não lineares
    """
    if df is None or df.empty:
        return None, None
    
    X = df[variaveis_independentes].values
    y = df[variavel_dependente].values
    
    if tipo == 'polinomial':
        # Modelo polinomial: y = a + b*x + c*x²
        def modelo_polinomial(x, a, b, c):
            return a + b * x[:, 0] + c * x[:, 0]**2
        
        try:
            params, _ = curve_fit(modelo_polinomial, X, y)
            return params, modelo_polinomial
        except:
            return None, None
    
    elif tipo == 'exponencial':
        # Modelo exponencial: y = a * exp(b*x)
        def modelo_exponencial(x, a, b):
            return a * np.exp(b * x[:, 0])
        
        try:
            params, _ = curve_fit(modelo_exponencial, X, y)
            return params, modelo_exponencial
        except:
            return None, None
    
    return None, None

def modelo_ressalto_hidraulico(df):
    """
    Modelo físico baseado na equação de Bélanger para ressalto hidráulico
    """
    if df is None or df.empty:
        return None
    
    # Criar variáveis para o modelo de ressalto
    # Baseado na equação de Bélanger: h2/h1 = 0.5 * (sqrt(1 + 8*Fr1²) - 1)
    # Onde Fr1 é o número de Froude inicial
    
    # Assumindo que Z representa a altura da água
    df['altura_ressalto'] = df['Z m'].max() - df['Z m']
    df['velocidade_media'] = df['w m/s'].abs()
    
    # Calcular número de Froude aproximado (assumindo profundidade média)
    profundidade_media = df['Z m'].mean()
    df['numero_froude'] = df['velocidade_media'] / np.sqrt(9.81 * profundidade_media)
    
    # Modelo de Bélanger simplificado
    df['altura_teorica'] = 0.5 * (np.sqrt(1 + 8 * df['numero_froude']**2) - 1) * profundidade_media
    
    return df

def validar_modelo(modelo, X, y):
    """
    Realiza validação completa do modelo
    """
    if modelo is None:
        return {}
    
    # Predições
    y_pred = modelo.predict(X)
    residuos = y - y_pred
    
    # Métricas de qualidade
    r_squared = modelo.rsquared
    r_squared_adj = modelo.rsquared_adj
    f_statistic = modelo.fvalue
    f_pvalue = modelo.f_pvalue
    
    # Teste de normalidade dos resíduos (Shapiro-Wilk)
    if len(residuos) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuos)
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Teste de homocedasticidade (Breusch-Pagan)
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuos, X)
    except:
        bp_stat, bp_p = None, None
    
    # Teste de autocorrelação (Ljung-Box)
    try:
        # Usar return_df=False para versões mais antigas do statsmodels
        lb_results = acorr_ljungbox(residuos, lags=[10])
        lb_stat = lb_results['lb_stat'].iloc[0] if isinstance(lb_results, pd.DataFrame) else lb_results[0][0]
        lb_p = lb_results['lb_pvalue'].iloc[0] if isinstance(lb_results, pd.DataFrame) else lb_results[1][0]
    except:
        lb_stat, lb_p = None, None
    
    # RMSE e MAE
    rmse = np.sqrt(np.mean(residuos**2))
    mae = np.mean(np.abs(residuos))
    
    return {
        'r_squared': r_squared,
        'r_squared_adj': r_squared_adj,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        'lb_stat': lb_stat,
        'lb_p': lb_p,
        'rmse': rmse,
        'mae': mae,
        'residuos': residuos,
        'y_pred': y_pred
    }

def criar_graficos_validacao(df, validacao, variavel_dependente):
    """
    Cria gráficos de validação do modelo
    """
    if not validacao:
        return None, None, None, None
    
    residuos = validacao['residuos']
    y_pred = validacao['y_pred']
    y_real = df[variavel_dependente]
    
    # 1. Gráfico de resíduos vs valores preditos
    fig_residuos = px.scatter(
        x=y_pred, y=residuos,
        title='Resíduos vs Valores Preditos',
        labels={'x': 'Valores Preditos', 'y': 'Resíduos'},
        color_discrete_sequence=['#3366CC']
    )
    fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")
    fig_residuos.update_layout(
        xaxis_title='Valores Preditos',
        yaxis_title='Resíduos'
    )
    
    # 2. Histograma dos resíduos
    fig_hist_residuos = px.histogram(
        x=residuos,
        title='Distribuição dos Resíduos',
        labels={'x': 'Resíduos', 'y': 'Frequência'},
        nbins=30,
        color_discrete_sequence=['#3366CC']
    )
    fig_hist_residuos.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist_residuos.update_layout(
        xaxis_title='Resíduos',
        yaxis_title='Frequência'
    )
    
    # 3. Q-Q Plot
    qq_data = stats.probplot(residuos, dist="norm")
    fig_qq = px.scatter(
        x=qq_data[0][0], y=qq_data[0][1],
        title='Q-Q Plot dos Resíduos',
        labels={'x': 'Quantis Teóricos', 'y': 'Quantis dos Resíduos'},
        color_discrete_sequence=['#3366CC']
    )
    
    min_val = min(qq_data[0][0])
    max_val = max(qq_data[0][0])
    fig_qq.add_traces(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha de Referência',
            line=dict(color='red', dash='dash')
        )
    )
    
    # 4. Valores observados vs preditos
    fig_obs_pred = px.scatter(
        x=y_real, y=y_pred,
        title='Valores Observados vs Preditos',
        labels={'x': 'Valores Observados', 'y': 'Valores Preditos'},
        color_discrete_sequence=['#3366CC']
    )
    
    min_val_obs_pred = min(y_real.min(), y_pred.min())
    max_val_obs_pred = max(y_real.max(), y_pred.max())
    fig_obs_pred.add_traces(
        go.Scatter(
            x=[min_val_obs_pred, max_val_obs_pred],
            y=[min_val_obs_pred, max_val_obs_pred],
            mode='lines',
            name='Linha de Identidade',
            line=dict(color='red', dash='dash')
        )
    )
    
    return fig_residuos, fig_hist_residuos, fig_qq, fig_obs_pred

def comparar_modelos(df, variavel_dependente, variaveis_independentes):
    """
    Compara diferentes modelos e retorna métricas de comparação
    """
    modelos = {}
    
    # Modelo linear múltipla
    modelo_linear, X_linear = modelo_linear_multipla(df, variavel_dependente, variaveis_independentes)
    if modelo_linear:
        validacao_linear = validar_modelo(modelo_linear, X_linear, df[variavel_dependente])
        modelos['Linear Múltipla'] = {
            'modelo': modelo_linear,
            'validacao': validacao_linear,
            'X': X_linear,
            'y_pred': validacao_linear.get('y_pred')
        }
    
    # Modelos não lineares
    for tipo in ['polinomial', 'exponencial']:
        params, func = modelo_nao_linear(df, variavel_dependente, variaveis_independentes[:1], tipo)
        if params is not None and func is not None:
            # Calcular predições para modelo não linear
            X_nl = df[variaveis_independentes[:1]].values
            y_pred_nl = func(X_nl, *params)
            residuos_nl = df[variavel_dependente] - y_pred_nl
            
            # Métricas para modelo não linear
            ss_res = np.sum(residuos_nl**2)
            ss_tot = np.sum((df[variavel_dependente] - df[variavel_dependente].mean())**2)
            r_squared = 1 - (ss_res / ss_tot)
            rmse = np.sqrt(np.mean(residuos_nl**2))
            mae = np.mean(np.abs(residuos_nl))
            
            modelos[f'Modelo {tipo.title()}'] = {
                'params': params,
                'func': func,
                'r_squared': r_squared,
                'rmse': rmse,
                'mae': mae,
                'residuos': residuos_nl,
                'y_pred': y_pred_nl
            }
    
    return modelos

#======================================================================
# INTERFACE STREAMLIT - ROTINAS DE EXECUÇÃO
#======================================================================

def run_etapa1(df):
    """
    Executa a interface e a lógica da Etapa 1: Análise Exploratória.
    """
    st.title("📊 Etapa 1: Análise de Dados de Ressalto Hidráulico")
    
    st.markdown("""
    Esta aplicação realiza uma análise exploratória completa de dados de ressalto hidráulico.
    Os dados incluem medições de tempo, posição (Z) e velocidade (w).
    """)
    
    df_limpo = limpar_dados(df.copy())
    if df_limpo is not None:
        st.success(f"Dados carregados e limpos com sucesso: {df_limpo.shape[0]} registros e {df_limpo.shape[1]} colunas")
        
        # Exibir as primeiras linhas dos dados
        st.subheader("Visualização dos Dados")
        
        with st.expander("Ver primeiras linhas dos dados", expanded=True):
            st.dataframe(df_limpo.head(10))
            
            # Download dos dados limpos
            csv = df_limpo.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download dos dados limpos (CSV)",
                data=csv,
                file_name="dados_ressalto_limpos.csv",
                mime="text/csv"
            )
        
        # Resumo estatístico
        st.subheader("1. Resumo Estatístico")
        
        with st.expander("Ver resumo estatístico", expanded=True):
            resumo = resumo_estatistico(df_limpo)
            if resumo is not None:
                st.dataframe(resumo)
                
                # Download do resumo estatístico
                csv_resumo = resumo.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download do resumo estatístico (CSV)",
                    data=csv_resumo,
                    file_name="resumo_estatistico.csv",
                    mime="text/csv"
                )
        
        # Análise de cada variável
        st.subheader("2. Análise por Variável")
        
        # Seleção das variáveis numéricas para análise
        variaveis_numericas = df_limpo.select_dtypes(include=[np.number]).columns.tolist()
        
        variavel_selecionada = st.selectbox(
            "Selecione uma variável para análise detalhada:",
            variaveis_numericas,
            key='select_var_etapa1'
        )
        
        if variavel_selecionada:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma
                st.subheader(f"Histograma de {variavel_selecionada}")
                hist_fig = criar_histograma(df_limpo, variavel_selecionada)
                if hist_fig:
                    st.plotly_chart(hist_fig, use_container_width=True)
                
                # Estatísticas robustas
                st.subheader(f"Estatísticas de {variavel_selecionada}")
                estat = analise_estatistica_robusta(df_limpo, variavel_selecionada)
                
                if estat:
                    normalidade = "Normal" if estat.get('é_normal') else "Não Normal"
                    if estat.get('shapiro_p_valor') is not None:
                        normalidade_texto = f"Shapiro-Wilk p-valor: {estat['shapiro_p_valor']:.4f} - Distribuição: {normalidade}"
                    else:
                        normalidade_texto = f"Kolmogorov-Smirnov p-valor: {estat['ks_p_valor']:.4f} - Distribuição: {normalidade}"
                    
                    estatisticas_df = pd.DataFrame({
                        'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Assimetria', 'Curtose'],
                        'Valor': [f"{v:.4f}" for v in [estat['média'], estat['mediana'], estat['desvio_padrão'], estat['min'], estat['max'], estat['assimetria'], estat['curtose']]]
                    })
                    
                    st.dataframe(estatisticas_df)
                    st.info(normalidade_texto)
            
            with col2:
                # Boxplot
                st.subheader(f"Boxplot de {variavel_selecionada}")
                box_fig = criar_boxplot(df_limpo, variavel_selecionada)
                if box_fig:
                    st.plotly_chart(box_fig, use_container_width=True)
                
                # Detecção de outliers
                st.subheader("Detecção de Outliers")
                metodo_outlier = st.radio(
                    "Método para detecção de outliers:",
                    ["IQR (Intervalo Interquartil)", "Z-Score"],
                    horizontal=True,
                    key='radio_outlier_etapa1'
                )
                
                metodo = 'iqr' if "IQR" in metodo_outlier else 'z-score'
                outliers, num_outliers = detectar_outliers(df_limpo, variavel_selecionada, metodo)
                
                st.info(f"Foram detectados {num_outliers} outliers usando o método {metodo_outlier}.")
                
                if num_outliers > 0 and not outliers.empty:
                    st.dataframe(outliers.head(10))
            
            # Gráfico de linha do tempo
            st.subheader(f"Evolução de {variavel_selecionada} ao longo do tempo")
            linha_fig = criar_grafico_linha_tempo(df_limpo, variavel_selecionada)
            if linha_fig:
                st.plotly_chart(linha_fig, use_container_width=True)
        
        # Análise de correlação
        st.subheader("3. Análise de Correlação")
        
        with st.expander("Ver matriz de correlação", expanded=True):
            heatmap_fig = criar_heatmap_correlacao(df_limpo)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Análise de relações entre variáveis
        st.subheader("4. Relações entre Variáveis")
        
        col1_scatter, col2_scatter = st.columns(2)
        
        with col1_scatter:
            var_x = st.selectbox("Selecione a variável para o eixo X:", variaveis_numericas, key='select_x_etapa1')
        
        with col2_scatter:
            var_y = st.selectbox("Selecione a variável para o eixo Y:", 
                                [v for v in variaveis_numericas if v != var_x], key='select_y_etapa1')
        
        if var_x and var_y:
            scatter_fig = criar_scatter(df_limpo, var_x, var_y)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                corr = df_limpo[[var_x, var_y]].corr().iloc[0, 1]
                st.info(f"Correlação de Pearson entre {var_x} e {var_y}: {corr:.4f}")
                
                if abs(corr) < 0.3:
                    st.write("Interpretação: Correlação fraca.")
                elif abs(corr) < 0.7:
                    st.write("Interpretação: Correlação moderada.")
                else:
                    st.write("Interpretação: Correlação forte.")
        
        # Conclusões e Recomendações
        st.subheader("5. Conclusões e Recomendações")
        
        with st.expander("Ver conclusões da análise", expanded=True):
            st.write("""
            ### Conclusões da Análise
            1. **Estatísticas Descritivas:** Os dados apresentam padrões interessantes nas medidas de tendência central e dispersão.
            2. **Distribuição dos Dados:** A análise de normalidade e os histogramas mostram como as variáveis se distribuem.
            3. **Outliers:** A detecção de outliers permite identificar valores extremos.
            4. **Correlações:** A análise revela relações entre as variáveis tempo, posição e velocidade.
            5. **Evolução Temporal:** Os gráficos de linha mostram como as variáveis evoluem ao longo do tempo.
            """)
            st.write("""
            ### Recomendações
            1. **Modelagem Matemática:** Desenvolver modelos matemáticos baseados nas relações observadas.
            2. **Comparação com Teoria:** Confrontar os resultados com as equações teóricas do ressalto hidráulico.
            3. **Análise de Sensibilidade:** Investigar como variações nas condições afetam o ressalto.
            4. **Experimentos Adicionais:** Realizar novos experimentos para validar os resultados.
            5. **Visualização 3D:** Considerar visualizações tridimensionais que integrem tempo, espaço e velocidade.
            """)

def run_etapa2(df):
    """
    Executa a interface e a lógica da Etapa 2: Modelagem e Validação.
    """
    st.title("📈 Etapa 2: Modelagem de Ressalto Hidráulico")
    
    st.markdown("## Etapa II: Escolha e Teste de Modelo Científico")
    
    df_modelagem = preparar_dados_modelagem(df.copy())
    if df_modelagem is None:
        st.error("Erro ao preparar dados para modelagem.")
        return
    
    st.success(f"Dados preparados para modelagem: {df_modelagem.shape[0]} registros e {df_modelagem.shape[1]} colunas")
    
    # --- Controles da Sidebar para Etapa 2 ---
    st.sidebar.header("Configurações do Modelo")
    
    variaveis_disponiveis = df_modelagem.select_dtypes(include=[np.number]).columns.tolist()
    variaveis_excluir = ['tempo_quadrado', 'Z_quadrado', 'w_quadrado', 'tempo_Z', 'tempo_w', 'Z_w', 'altura_ressalto', 'velocidade_media', 'numero_froude', 'altura_teorica']
    variaveis_selecionaveis = [v for v in variaveis_disponiveis if v not in variaveis_excluir]

    variavel_dependente = st.sidebar.selectbox(
        "Variável Dependente (Y):",
        variaveis_selecionaveis,
        index=variaveis_selecionaveis.index('Z m') if 'Z m' in variaveis_selecionaveis else 0,
        key='select_dep_etapa2'
    )
    
    variaveis_independentes = st.sidebar.multiselect(
        "Variáveis Independentes (X):",
        [v for v in variaveis_selecionaveis if v != variavel_dependente],
        default=[v for v in ['tempo s', 'w m/s'] if v in variaveis_selecionaveis and v != variavel_dependente],
        key='multi_indep_etapa2'
    )
    
    tipo_modelo = st.sidebar.selectbox(
        "Tipo de Modelo Estatístico:",
        ["Regressão Linear Múltipla", "Comparação de Modelos"],
        key='select_model_etapa2'
    )
    
    # --- Seção 1: Modelagem Estatística ---
    st.header("1. Modelagem Estatística")
    
    if not variaveis_independentes:
        st.warning("Selecione pelo menos uma variável independente na barra lateral.")
        return
    
    if tipo_modelo == "Regressão Linear Múltipla":
        st.subheader("Regressão Linear Múltipla")
        
        modelo, X = modelo_linear_multipla(df_modelagem, variavel_dependente, variaveis_independentes)
        
        if modelo:
            st.subheader("Resumo do Modelo")
            st.text(str(modelo.summary()))
            
            st.subheader("Validação do Modelo")
            validacao = validar_modelo(modelo, X, df_modelagem[variavel_dependente])
            
            if validacao:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R²", f"{validacao['r_squared']:.4f}")
                    st.metric("R² Ajustado", f"{validacao['r_squared_adj']:.4f}")
                    st.metric("RMSE", f"{validacao['rmse']:.4f}")
                
                with col2:
                    st.metric("Estatística F", f"{validacao['f_statistic']:.4f}")
                    st.metric("p-valor F", f"{validacao['f_pvalue']:.4f}")
                    st.metric("MAE", f"{validacao['mae']:.4f}")
                
                st.subheader("Interpretação dos Resultados")
                if validacao['f_pvalue'] < 0.05:
                    st.success("✅ Teste F: O modelo é estatisticamente significativo (p < 0.05)")
                else:
                    st.error("❌ Teste F: O modelo não é estatisticamente significativo (p ≥ 0.05)")
                
                if validacao['r_squared'] > 0.7:
                    st.success(f"✅ R² = {validacao['r_squared']:.4f}: O modelo explica bem a variabilidade")
                elif validacao['r_squared'] > 0.5:
                    st.warning(f"⚠️ R² = {validacao['r_squared']:.4f}: O modelo tem explicação moderada")
                else:
                    st.error(f"❌ R² = {validacao['r_squared']:.4f}: O modelo tem baixa explicação")
                
                st.subheader("Gráficos de Validação")
                fig_res, fig_hist, fig_qq, fig_obs = criar_graficos_validacao(df_modelagem, validacao, variavel_dependente)
                
                if fig_res:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(fig_res, use_container_width=True)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    with c2:
                        st.plotly_chart(fig_qq, use_container_width=True)
                        st.plotly_chart(fig_obs, use_container_width=True)

    elif tipo_modelo == "Comparação de Modelos":
        st.subheader("Comparação de Modelos (Linear, Polinomial, Exponencial)")
        
        modelos_comp = comparar_modelos(df_modelagem, variavel_dependente, variaveis_independentes)
        
        if modelos_comp:
            st.subheader("Métricas de Comparação")
            dados_comp = []
            for nome, info in modelos_comp.items():
                r2 = info.get('validacao', {}).get('r_squared') if 'validacao' in info else info.get('r_squared')
                rmse = info.get('validacao', {}).get('rmse') if 'validacao' in info else info.get('rmse')
                mae = info.get('validacao', {}).get('mae') if 'validacao' in info else info.get('mae')
                dados_comp.append({'Modelo': nome, 'R²': r2, 'RMSE': rmse, 'MAE': mae})

            df_comp = pd.DataFrame(dados_comp).dropna().sort_values(by='R²', ascending=False)
            st.dataframe(df_comp)
            
            st.subheader("Gráfico de Comparação: Observado vs. Predito")
            fig_comp = go.Figure()
            for nome, info in modelos_comp.items():
                if info.get('y_pred') is not None:
                    fig_comp.add_trace(go.Scatter(x=df_modelagem[variavel_dependente], y=info['y_pred'], mode='markers', name=nome, opacity=0.7))
            
            min_val = df_modelagem[variavel_dependente].min()
            max_val = df_modelagem[variavel_dependente].max()
            fig_comp.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Linha de Identidade', line=dict(color='red', dash='dash')))
            fig_comp.update_layout(title='Comparação de Modelos', xaxis_title='Valores Observados', yaxis_title='Valores Preditos')
            st.plotly_chart(fig_comp, use_container_width=True)

    # --- Seção 2: Modelo Físico ---
    st.header("2. Modelo Físico do Ressalto Hidráulico")
    st.markdown("""
    ### Equação de Bélanger
    A equação de Bélanger é fundamental para o ressalto hidráulico: **$h_2/h_1 = 0.5 \\times (\\sqrt{1 + 8 \\cdot Fr_1^2} - 1)$**
    - $h_1$: profundidade inicial
    - $h_2$: profundidade após o ressalto
    - $Fr_1$: número de Froude inicial
    """)
    
    df_fisico = modelo_ressalto_hidraulico(df_modelagem.copy())
    
    if df_fisico is not None:
        st.subheader("Análise do Modelo Físico")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Profundidade Média", f"{df_fisico['Z m'].mean():.4f} m")
            st.metric("Velocidade Média", f"{df_fisico['velocidade_media'].mean():.4f} m/s")
            st.metric("Número de Froude Médio", f"{df_fisico['numero_froude'].mean():.4f}")
        with c2:
            st.metric("Altura Máxima do Ressalto", f"{df_fisico['altura_ressalto'].max():.4f} m")
            st.metric("Altura Teórica Média (Bélanger)", f"{df_fisico['altura_teorica'].mean():.4f} m")
        
        fig_fisico = make_subplots(rows=2, cols=1, subplot_titles=('Altura do Ressalto vs Tempo', 'Comparação: Observado vs Teórico'), vertical_spacing=0.15)
        fig_fisico.add_trace(go.Scatter(x=df_fisico['tempo s'], y=df_fisico['altura_ressalto'], mode='lines', name='Altura Observada'), row=1, col=1)
        fig_fisico.add_trace(go.Scatter(x=df_fisico['altura_ressalto'], y=df_fisico['altura_teorica'], mode='markers', name='Observado vs Teórico'), row=2, col=1)
        
        min_alt = min(df_fisico['altura_ressalto'].min(), df_fisico['altura_teorica'].min())
        max_alt = max(df_fisico['altura_ressalto'].max(), df_fisico['altura_teorica'].max())
        fig_fisico.add_trace(go.Scatter(x=[min_alt, max_alt], y=[min_alt, max_alt], mode='lines', name='Linha de Identidade', line=dict(color='red', dash='dash')), row=2, col=1)
        
        fig_fisico.update_layout(height=700, title_text="Modelo Físico do Ressalto Hidráulico")
        st.plotly_chart(fig_fisico, use_container_width=True)
    
    # --- Seção 3: Conclusões e Recomendações ---
    st.header("3. Conclusões e Recomendações da Modelagem")
    with st.expander("Ver Conclusões e Recomendações", expanded=False):
        st.markdown("""
        ### Principais Conclusões
        1. **Validação Estatística**: O modelo foi validado através de Teste F, análise de resíduos, e outros.
        2. **Qualidade do Ajuste**: R², RMSE e MAE fornecem medidas da qualidade do modelo.
        3. **Modelo Físico**: A equação de Bélanger foi aplicada para comparar valores observados e teóricos.
        
        ### Recomendações
        1. **Validação Cruzada**: Implementar para verificar a robustez do modelo.
        2. **Modelos Avançados**: Considerar modelos mais complexos (ex: Machine Learning).
        3. **Experimentos Adicionais**: Realizar novos experimentos para validar o modelo em diferentes condições.
        4. **Análise de Incerteza**: Quantificar incertezas nas medições e suas propagações.
        """)
        
    # --- Seção 4: Exportação ---
    st.header("4. Exportação de Resultados do Modelo")
    
    if 'modelo' in locals() and modelo and 'validacao' in locals() and validacao:
        resumo_modelo_df = pd.DataFrame({
            'Métrica': ['R²', 'R² Ajustado', 'RMSE', 'MAE', 'Estatística F', 'p-valor F'],
            'Valor': [f"{v:.4f}" for v in [validacao['r_squared'], validacao['r_squared_adj'], validacao['rmse'], validacao['mae'], validacao['f_statistic'], validacao['f_pvalue']]]
        })
        
        csv_resumo_modelo = resumo_modelo_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download do Resumo do Modelo (CSV)",
            data=csv_resumo_modelo,
            file_name="resumo_modelo_ressalto.csv",
            mime="text/csv"
        )

#======================================================================
# FUNÇÃO PRINCIPAL (MAIN)
#======================================================================

def main():
    """
    Função principal que controla a navegação entre as etapas.
    """
    # --- Menu da Barra Lateral ---
    st.sidebar.title("Navegação do Projeto")
    st.sidebar.markdown("---")
    
    etapa_selecionada = st.sidebar.radio(
        "Selecione a Etapa do Projeto:",
        ("Etapa 1: Análise Exploratória", "Etapa 2: Modelagem e Validação")
    )
    
    # --- Carregamento dos Dados ---
    arquivo_dados = "dados_unidos.csv"
    if not os.path.exists(arquivo_dados):
        st.error(f"Arquivo de dados não encontrado: '{arquivo_dados}'")
        st.info("Por favor, certifique-se de que o arquivo está na mesma pasta que o script.")
        return
        
    df_bruto = carregar_dados(arquivo_dados)
    if df_bruto is None:
        return

    # --- Execução da Etapa Selecionada ---
    if etapa_selecionada == "Etapa 1: Análise Exploratória":
        run_etapa1(df_bruto)
    
    elif etapa_selecionada == "Etapa 2: Modelagem e Validação":
        run_etapa2(df_bruto)


if __name__ == "__main__":
    main()