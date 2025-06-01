# 🌊 Análise de Dados de Ressalto Hidráulico

## 🎓 Contexto do Projeto

Este projeto foi desenvolvido a partir de um trabalho acadêmico da minha irmã, estudante de pós-graduação. Inicialmente, os dados experimentais estavam distribuídos em 18 arquivos CSV distintos. Para possibilitar uma análise consolidada, foi utilizado o script `juntar.py` (disponível neste repositório) para unificar todos esses arquivos em um único conjunto de dados, o `dados_unidos.csv`. Este arquivo consolidado é a base para todas as análises apresentadas neste dashboard.

O dashboard interativo, construído com Streamlit, visa realizar uma análise exploratória completa dos dados experimentais de um ressalto hidráulico. O objetivo é fornecer uma ferramenta visual e analítica para entender as características e dinâmicas deste fenômeno hidráulico a partir de medições de tempo, posição (Z) e velocidade (w).

## 🚀 Funcionalidades Principais

* **Pré-processamento com `juntar.py`:** Unificação de 18 arquivos CSV em um único dataset (`dados_unidos.csv`) para análise.
* **Carregamento e Limpeza de Dados:** Carrega dados do arquivo CSV unificado (`dados_unidos.csv`), remove espaços e caracteres especiais dos nomes das colunas, converte colunas para tipos numéricos apropriados e trata valores ausentes.
* **Visualização Interativa dos Dados:** Permite a visualização das primeiras linhas do conjunto de dados e o download dos dados limpos.
* **Resumo Estatístico Detalhado:** Calcula e exibe estatísticas descritivas (média, mediana, desvio padrão, min, max), além de assimetria e curtose para as variáveis numéricas.
* **Análise Univariada Detalhada:** Para cada variável numérica selecionada:
    * **Histogramas:** Com indicação de média e mediana para visualizar a distribuição.
    * **Boxplots:** Para identificar a dispersão dos dados e possíveis outliers.
    * **Estatísticas Robustas:** Incluindo testes de normalidade (Shapiro-Wilk e Kolmogorov-Smirnov).
    * **Detecção de Outliers:** Utilizando métodos IQR (Intervalo Interquartil) ou Z-Score.
    * **Gráficos de Linha Temporal:** Para observar a evolução da variável ao longo do tempo.
* **Análise de Correlação:**
    * **Matriz de Correlação:** Apresentada como um heatmap para visualizar a força e direção das relações lineares entre as variáveis numéricas.
* **Análise Bivariada:**
    * **Gráficos de Dispersão (Scatter Plots):** Com linha de tendência (regressão linear) e cálculo do coeficiente de correlação de Pearson para investigar a relação entre duas variáveis selecionadas.
* **Conclusões e Recomendações:** Seção com interpretações gerais da análise e sugestões para estudos futuros.

## 🛠️ Tecnologias Utilizadas

* **Python:** Linguagem principal para desenvolvimento e para o script de unificação de dados.
* **Streamlit:** Para a criação do dashboard interativo e da interface web.
* **Pandas:** Para manipulação, análise de dados tabulares e para a leitura/escrita dos arquivos CSV no script `juntar.py`.
* **NumPy:** Para operações numéricas eficientes.
* **Plotly (Express & Graph Objects):** Para a criação de gráficos interativos e visualizações de dados.
* **SciPy:** Para cálculos estatísticos, como detecção de outliers (Z-Score), testes de normalidade e regressão linear.
* **OS & Glob:** Bibliotecas Python utilizadas no script `juntar.py` para manipulação de caminhos de arquivos e listagem de arquivos CSV.
* **Math:** Bibliotecas padrão do Python para interações com o sistema operacional e funções matemáticas (usadas implicitamente ou minimamente).

## 📊 Processo de Análise de Dados (Passo a Passo no Dashboard)

0.  **Unificação dos Dados (Pré-etapa):**
    * O script `juntar.py` é executado para combinar 18 arquivos CSV do diretório de entrada em um único arquivo `dados_unidos.csv`. O script adiciona uma coluna `arquivo_origem` para identificar a fonte de cada registro.

1.  **Carregamento e Visualização Inicial dos Dados:**
    * O arquivo `dados_unidos.csv` é carregado automaticamente pelo dashboard.
    * Os dados são limpos (nomes de colunas padronizados, tipos convertidos, NaNs tratados).
    * As primeiras linhas do DataFrame limpo são exibidas, com opção de download.

2.  **Resumo Estatístico Geral:**
    * Uma tabela com estatísticas descritivas (média, desvio padrão, quartis, assimetria, curtose) para todas as variáveis numéricas é apresentada, com opção de download.

3.  **Análise por Variável:**
    * O usuário seleciona uma variável numérica (`tempo s`, `Z m`, `w m/s`) para análise detalhada.
    * **Visualizações:** Histograma e Boxplot da variável selecionada são exibidos.
    * **Estatísticas:** Métricas como média, mediana, desvio padrão, e testes de normalidade (Shapiro-Wilk, Kolmogorov-Smirnov) são calculados e mostrados.
    * **Outliers:** Detecção e exibição de outliers (usuário escolhe método IQR ou Z-Score).
    * **Evolução Temporal:** Um gráfico de linha mostra a variável selecionada em função do tempo.

4.  **Análise de Correlação:**
    * Um heatmap da matriz de correlação entre todas as variáveis numéricas é exibido, mostrando a intensidade e direção das relações.

5.  **Relações entre Variáveis (Análise Bivariada):**
    * O usuário seleciona duas variáveis numéricas (X e Y).
    * Um gráfico de dispersão é gerado, incluindo uma linha de tendência e o valor da correlação de Pearson entre as duas variáveis.
    * Uma interpretação básica da força da correlação é fornecida.

6.  **Conclusões e Recomendações:**
    * Uma seção de texto apresenta conclusões gerais derivadas da análise exploratória e sugere caminhos para investigações futuras.

## 💡 Principais Insights Gerados (Exemplos)

* **Distribuição das Variáveis:** Compreensão se os dados de `Z m` (posição) ou `w m/s` (velocidade) seguem uma distribuição normal ou apresentam assimetria/curtose significativas.
* **Presença de Outliers:** Identificação de medições atípicas que podem indicar erros experimentais ou fenômenos específicos no ressalto.
* **Relações Lineares:** Quantificação da correlação entre, por exemplo, a posição (Z) e a velocidade (w), ou como estas variam com o tempo.
* **Evolução Temporal:** Observação de como a posição e a velocidade do ressalto hidráulico mudam ao longo da duração do experimento.
* **Tendências Centrais e Dispersão:** Determinação dos valores médios, medianos e da variabilidade para cada parâmetro medido.
* **Consistência entre Arquivos:** A coluna `arquivo_origem` (adicionada pelo `juntar.py`) pode ser utilizada para verificar se há variações significativas entre os diferentes conjuntos de dados originais.

## 🚀 Como Executar o Projeto Localmente

### Pré-requisito: Unificar os Arquivos CSV

1.  Clone o repositório (ou certifique-se de ter o script `juntar.py` e seus arquivos CSV de entrada).
2.  Modifique os caminhos `diretorio_entrada` e `arquivo_saida` no final do script `juntar.py` para apontar para a pasta contendo seus 18 arquivos CSV e o local onde o `dados_unidos.csv` será salvo.
3.  Execute o script `juntar.py`:
    ```bash
    python juntar.py
    ```
    Isso criará o arquivo `dados_unidos.csv` necessário para o dashboard.


## 🤝 Contribuições

Contribuições são bem-vindas! Se você tiver sugestões para melhorar este projeto, sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

## 👨‍💻 Autor

* [Seu Nome Aqui]
* **LinkedIn:** [Seu LinkedIn Aqui (opcional)]
* **GitHub:** [Seu GitHub Aqui (opcional)]
* **Agradecimento:** À minha irmã, por fornecer os dados e o contexto acadêmico para este projeto de análise.

## 📄 Licença

Este projeto é distribuído sob a licença MIT (ou outra de sua escolha). Veja o arquivo `LICENSE` para mais detalhes (você pode adicionar um arquivo LICENSE.md com o texto da licença, se desejar).
