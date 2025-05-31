import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math
from scipy import stats
import io

# Configuração da página Streamlit
st.set_page_config(
    page_title="Análise de Ressalto Hidráulico",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funções de carregamento e limpeza de dados
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
    Realiza a limpeza dos dados:
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

# Funções de Análise Exploratória de Dados
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

# Funções de Visualização
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

# Interface Streamlit
def main():
    st.title("📊 Análise de Dados de Ressalto Hidráulico")
    
    st.markdown("""
    Esta aplicação realiza uma análise exploratória completa de dados de ressalto hidráulico.
    Os dados incluem medições de tempo, posição (Z) e velocidade (w).
    """)
    
    # Sidebar para controles
    st.sidebar.header("Controles")
    
    arquivo = "dados_unidos.csv"
    
    # Carregar dados
    if os.path.exists(arquivo):
        df = carregar_dados(arquivo)
        if df is not None:
            df = limpar_dados(df)
            st.sidebar.success(f"Dados carregados com sucesso: {df.shape[0]} registros e {df.shape[1]} colunas")
            
            # Exibir as primeiras linhas dos dados
            st.subheader("Visualização dos Dados")
            
            with st.expander("Ver primeiras linhas dos dados", expanded=True):
                st.dataframe(df.head(10))
                
                # Download dos dados limpos
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download dos dados limpos (CSV)",
                    data=csv,
                    file_name="dados_ressalto_limpos.csv",
                    mime="text/csv"
                )
            
            # Resumo estatístico
            st.subheader("1. Resumo Estatístico")
            
            with st.expander("Ver resumo estatístico", expanded=True):
                resumo = resumo_estatistico(df)
                if resumo is not None:
                    st.dataframe(resumo)
                    
                    # Download do resumo estatístico
                    csv_resumo = resumo.reset_index().to_csv(index=False)
                    st.download_button(
                        label="Download do resumo estatístico (CSV)",
                        data=csv_resumo,
                        file_name="resumo_estatistico.csv",
                        mime="text/csv"
                    )
            
            # Análise de cada variável
            st.subheader("2. Análise por Variável")
            
            # Seleção das variáveis numéricas para análise
            variaveis_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            
            variavel_selecionada = st.selectbox(
                "Selecione uma variável para análise detalhada:",
                variaveis_numericas
            )
            
            if variavel_selecionada:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma
                    st.subheader(f"Histograma de {variavel_selecionada}")
                    hist_fig = criar_histograma(df, variavel_selecionada)
                    if hist_fig:
                        st.plotly_chart(hist_fig, use_container_width=True)
                    
                    # Estatísticas robustas
                    st.subheader(f"Estatísticas de {variavel_selecionada}")
                    estat = analise_estatistica_robusta(df, variavel_selecionada)
                    
                    if estat:
                        # Verificação de normalidade
                        normalidade = "Normal" if estat['é_normal'] else "Não Normal"
                        if estat['shapiro_p_valor'] is not None:
                            normalidade_texto = f"Shapiro-Wilk p-valor: {estat['shapiro_p_valor']:.4f} - Distribuição: {normalidade}"
                        else:
                            normalidade_texto = f"Kolmogorov-Smirnov p-valor: {estat['ks_p_valor']:.4f} - Distribuição: {normalidade}"
                        
                        # Exibir estatísticas em formato de tabela
                        estatisticas_df = pd.DataFrame({
                            'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Assimetria', 'Curtose'],
                            'Valor': [
                                f"{estat['média']:.4f}",
                                f"{estat['mediana']:.4f}",
                                f"{estat['desvio_padrão']:.4f}",
                                f"{estat['min']:.4f}",
                                f"{estat['max']:.4f}",
                                f"{estat['assimetria']:.4f}",
                                f"{estat['curtose']:.4f}"
                            ]
                        })
                        
                        st.dataframe(estatisticas_df)
                        st.info(normalidade_texto)
                
                with col2:
                    # Boxplot
                    st.subheader(f"Boxplot de {variavel_selecionada}")
                    box_fig = criar_boxplot(df, variavel_selecionada)
                    if box_fig:
                        st.plotly_chart(box_fig, use_container_width=True)
                    
                    # Detecção de outliers
                    st.subheader("Detecção de Outliers")
                    metodo_outlier = st.radio(
                        "Método para detecção de outliers:",
                        ["IQR (Intervalo Interquartil)", "Z-Score"],
                        horizontal=True
                    )
                    
                    metodo = 'iqr' if metodo_outlier == "IQR (Intervalo Interquartil)" else 'z-score'
                    outliers, num_outliers = detectar_outliers(df, variavel_selecionada, metodo)
                    
                    st.info(f"Foram detectados {num_outliers} outliers usando o método {metodo_outlier}.")
                    
                    if num_outliers > 0 and not outliers.empty:
                        st.dataframe(outliers.head(10))
                
                # Gráfico de linha do tempo
                st.subheader(f"Evolução de {variavel_selecionada} ao longo do tempo")
                linha_fig = criar_grafico_linha_tempo(df, variavel_selecionada)
                if linha_fig:
                    st.plotly_chart(linha_fig, use_container_width=True)
            
            # Análise de correlação
            st.subheader("3. Análise de Correlação")
            
            with st.expander("Ver matriz de correlação", expanded=True):
                heatmap_fig = criar_heatmap_correlacao(df)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Análise de relações entre variáveis
            st.subheader("4. Relações entre Variáveis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                var_x = st.selectbox("Selecione a variável para o eixo X:", variaveis_numericas)
            
            with col2:
                var_y = st.selectbox("Selecione a variável para o eixo Y:", 
                                    [v for v in variaveis_numericas if v != var_x])
            
            if var_x and var_y:
                scatter_fig = criar_scatter(df, var_x, var_y)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    # Calculando e exibindo a correlação
                    corr = df[[var_x, var_y]].corr().iloc[0, 1]
                    st.info(f"Correlação de Pearson entre {var_x} e {var_y}: {corr:.4f}")
                    
                    # Interpretação da correlação
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
                
                Com base na análise exploratória dos dados de ressalto hidráulico, podemos concluir:
                
                1. **Estatísticas Descritivas:** Os dados apresentam padrões interessantes nas medidas de tendência central e dispersão, que podem ser úteis para entender o comportamento do ressalto hidráulico.
                
                2. **Distribuição dos Dados:** A análise de normalidade e os gráficos de histograma mostram como as variáveis se distribuem, o que é essencial para escolher métodos estatísticos apropriados.
                
                3. **Outliers:** A detecção de outliers permite identificar valores extremos que podem representar erros de medição ou fenômenos interessantes no experimento.
                
                4. **Correlações:** A análise de correlação revela relações entre as variáveis tempo, posição e velocidade, fundamentais para compreender a dinâmica do ressalto hidráulico.
                
                5. **Evolução Temporal:** Os gráficos de linha mostram como as variáveis evoluem ao longo do tempo, permitindo identificar padrões e tendências.
                """)
                
                st.write("""
                ### Recomendações
                
                Para aprofundar a análise e obter mais insights:
                
                1. **Modelagem Matemática:** Desenvolver modelos matemáticos baseados nas relações observadas entre tempo, posição e velocidade.
                
                2. **Comparação com Teoria:** Confrontar os resultados experimentais com as equações teóricas do ressalto hidráulico, como a equação de Bélanger.
                
                3. **Análise de Sensibilidade:** Investigar como pequenas variações nas condições iniciais afetam o comportamento do ressalto.
                
                4. **Experimentos Adicionais:** Realizar novos experimentos com diferentes condições para validar os resultados e expandir o conhecimento sobre o fenômeno.
                
                5. **Visualização 3D:** Considerar visualizações tridimensionais que integrem tempo, espaço e velocidade para uma compreensão mais completa do ressalto hidráulico.
                """)
        else:
            st.error(f"Não foi possível carregar os dados do arquivo: {arquivo}")
    else:
        st.error(f"Arquivo não encontrado: {arquivo}")
        st.info("Por favor, verifique se o arquivo está na pasta raiz do projeto.")

if __name__ == "__main__":
    main()