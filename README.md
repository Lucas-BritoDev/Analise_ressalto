# Análise de Ressalto Hidráulico

## 📊 Sobre o Projeto

Este projeto realiza uma análise completa de dados de ressalto hidráulico, incluindo:

- **Etapa 1**: Análise exploratória dos dados
- **Etapa 2**: Modelagem estatística e física do ressalto hidráulico

## 🚀 Deploy no Streamlit Cloud

### Pré-requisitos

1. Tenha uma conta no [Streamlit Cloud](https://streamlit.io/cloud)
2. Seu código deve estar em um repositório Git (GitHub, GitLab, etc.)

### Como fazer o Deploy

1. **Faça push do código para o repositório**
   ```bash
   git add .
   git commit -m "Preparando para deploy no Streamlit Cloud"
   git push origin main
   ```

2. **No Streamlit Cloud:**
   - Acesse [share.streamlit.io](https://share.streamlit.io)
   - Faça login com sua conta
   - Clique em "New app"
   - Selecione seu repositório
   - Configure:
     - **Main file path**: `app.py`
     - **Python version**: 3.8 ou superior

3. **Arquivos necessários já estão configurados:**
   - `requirements.txt` - Dependências Python
   - `pyproject.toml` - Configuração do projeto
   - `.streamlit/config.toml` - Configurações do Streamlit
   - `packages.txt` - Dependências do sistema (vazio, mas presente)

## 📁 Estrutura do Projeto

```
analise_ressalto/
├── app.py                    # Aplicação principal
├── requirements.txt          # Dependências Python
├── pyproject.toml           # Configuração do projeto
├── packages.txt             # Dependências do sistema
├── .streamlit/
│   └── config.toml         # Configurações do Streamlit
├── dados_unidos.csv         # Dados unificados
└── README.md               # Este arquivo
```

## 🔧 Dependências

### Python
- `streamlit>=1.28.0` - Framework web
- `pandas>=1.5.0` - Manipulação de dados
- `numpy>=1.21.0` - Computação numérica
- `plotly>=5.15.0` - Gráficos interativos
- `scipy>=1.9.0` - Funções científicas
- `statsmodels>=0.13.0` - Modelagem estatística

## 📈 Funcionalidades

### Etapa 1: Análise Exploratória
- Carregamento e limpeza de dados
- Estatísticas descritivas
- Detecção de outliers
- Análise de correlação
- Visualizações (histogramas, boxplots, scatter plots)

### Etapa 2: Modelagem
- Regressão linear múltipla
- Modelos não lineares (polinomial, exponencial)
- Modelo físico baseado na equação de Bélanger
- Validação estatística completa
- Comparação de modelos

## 🎯 Como Usar

1. **Localmente:**
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **No Streamlit Cloud:**
   - Acesse o link fornecido após o deploy
   - Navegue entre as etapas usando o menu lateral
   - Interaja com os gráficos e análises

## 📊 Dados

O projeto utiliza dados de ressalto hidráulico contendo:
- **Tempo (s)**: Medições temporais
- **Z (m)**: Posição/altura da água
- **w (m/s)**: Velocidade da água

## 🔍 Análises Disponíveis

- **Estatísticas robustas** com testes de normalidade
- **Detecção de outliers** usando IQR e Z-score
- **Análise de correlação** entre variáveis
- **Modelagem estatística** com validação
- **Modelo físico** do ressalto hidráulico

## 📝 Licença

Este projeto está sob a licença MIT.

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.
