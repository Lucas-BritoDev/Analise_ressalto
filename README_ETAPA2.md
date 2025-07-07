# 📈 Etapa II: Modelagem de Ressalto Hidráulico

## Visão Geral

Esta é a **segunda etapa** do projeto de análise de ressalto hidráulico, focada na **modelagem estatística** e **validação de modelos científicos**.

## 🎯 Objetivos da Etapa II

1. **Escolha de Modelo Científico**
   - Regressão linear múltipla
   - Modelos não lineares (polinomial, exponencial)
   - Modelo físico baseado na equação de Bélanger

2. **Validação Estatística**
   - Teste F para significância
   - Análise de resíduos
   - Verificação de pressupostos
   - Métricas de qualidade (R², RMSE, MAE)

3. **Comparação de Modelos**
   - Métricas comparativas
   - Gráficos de validação
   - Seleção do melhor modelo

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install -r requirements_etapa2.txt
```

### 2. Executar a Aplicação

```bash
streamlit run app4_modelagem.py
```

## 📊 Funcionalidades Principais

### 1. Modelagem Estatística

- **Regressão Linear Múltipla**: Ajuste de modelos lineares com múltiplas variáveis
- **Modelos Não Lineares**: Polinomial e exponencial
- **Seleção de Variáveis**: Interface para escolher variáveis dependentes e independentes

### 2. Validação Completa

- **Teste F**: Verificação de significância estatística
- **Análise de Resíduos**: 
  - Normalidade (Shapiro-Wilk)
  - Homocedasticidade (Breusch-Pagan)
  - Autocorrelação (Ljung-Box)
- **Gráficos de Validação**:
  - Resíduos vs Preditos
  - Histograma dos resíduos
  - Q-Q Plot
  - Observados vs Preditos

### 3. Modelo Físico

- **Equação de Bélanger**: Implementação do modelo físico do ressalto hidráulico
- **Número de Froude**: Cálculo automático
- **Comparação Teórico vs Observado**

### 4. Comparação de Modelos

- **Métricas Comparativas**: R², RMSE, MAE
- **Gráficos de Comparação**: Visualização da performance de diferentes modelos
- **Seleção Automática**: Identificação do melhor modelo

## 📈 Métricas de Qualidade

### R² (Coeficiente de Determinação)
- **> 0.7**: Excelente ajuste
- **0.5 - 0.7**: Bom ajuste
- **< 0.5**: Ajuste insuficiente

### RMSE (Raiz do Erro Quadrático Médio)
- Quanto menor, melhor o modelo
- Medida absoluta de erro

### Teste F
- **p < 0.05**: Modelo estatisticamente significativo
- **p ≥ 0.05**: Modelo não significativo

## 🔧 Configurações

### Variáveis Disponíveis
- `tempo s`: Tempo em segundos
- `Z m`: Posição vertical em metros
- `w m/s`: Velocidade em m/s
- `posicao`: Posição extraída do nome do arquivo

### Variáveis Derivadas
- `tempo_quadrado`: Tempo ao quadrado
- `Z_quadrado`: Posição ao quadrado
- `w_quadrado`: Velocidade ao quadrado
- `tempo_Z`: Interação tempo × posição
- `tempo_w`: Interação tempo × velocidade
- `Z_w`: Interação posição × velocidade

## 📋 Exemplo de Uso

1. **Carregar dados**: A aplicação carrega automaticamente `dados_unidos.csv`

2. **Selecionar modelo**:
   - Escolher "Regressão Linear Múltipla" para análise completa
   - Ou "Comparação de Modelos" para ver diferentes opções

3. **Configurar variáveis**:
   - **Variável Dependente**: `Z m` (posição)
   - **Variáveis Independentes**: `tempo s`, `w m/s`

4. **Analisar resultados**:
   - Verificar R² e significância estatística
   - Examinar gráficos de validação
   - Comparar com modelo físico

## 📊 Saídas da Aplicação

### 1. Resumo Estatístico
- Coeficientes do modelo
- Erros padrão
- Valores t e p-valores
- R² e R² ajustado

### 2. Validação
- Métricas de qualidade
- Testes estatísticos
- Gráficos de diagnóstico

### 3. Modelo Físico
- Comparação com equação de Bélanger
- Análise do número de Froude
- Gráficos comparativos

### 4. Exportação
- Download dos resultados em CSV
- Gráficos interativos
- Relatórios completos

## 🎓 Conceitos Teóricos

### Equação de Bélanger
```
h₂/h₁ = 0.5 × (√(1 + 8×Fr₁²) - 1)
```

Onde:
- h₁ = profundidade inicial
- h₂ = profundidade após o ressalto
- Fr₁ = número de Froude inicial

### Número de Froude
```
Fr = v/√(g×h)
```

Onde:
- v = velocidade
- g = aceleração da gravidade
- h = profundidade

## 🔍 Interpretação dos Resultados

### Teste F
- **H₀**: O modelo não explica significativamente os dados
- **H₁**: O modelo explica significativamente os dados
- **Decisão**: Rejeitar H₀ se p < 0.05

### Análise de Resíduos
- **Normalidade**: Resíduos devem seguir distribuição normal
- **Homocedasticidade**: Variância constante
- **Independência**: Sem autocorrelação

## 📚 Próximos Passos

1. **Validação Cruzada**: Implementar k-fold cross-validation
2. **Modelos Avançados**: Machine Learning (Random Forest, SVM)
3. **Análise de Sensibilidade**: Estudo de variações nos parâmetros
4. **Experimentos Adicionais**: Novas condições experimentais

## 🛠️ Solução de Problemas

### Erro de Carregamento de Dados
- Verificar se `dados_unidos.csv` existe
- Confirmar formato do arquivo (delimitador ';')

### Erro de Modelagem
- Verificar se variáveis selecionadas são numéricas
- Confirmar que não há valores nulos

### Problemas de Validação
- Verificar tamanho da amostra para testes estatísticos
- Confirmar que dados seguem pressupostos dos testes

## 📞 Suporte

Para dúvidas ou problemas:
1. Verificar logs do Streamlit
2. Confirmar instalação das dependências
3. Validar formato dos dados de entrada 