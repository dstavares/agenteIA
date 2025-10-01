import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. Agent instructions ---

PREFIXO_AGENTE_ANALISE_DADOS = """
Você é um ANALISTA DE DADOS ESPECIALISTA em Python, Pandas e Visualização de Dados. Sua função é fornecer análises profundas, metódicas e acionáveis sobre os dados fornecidos.

## 🎯 MISSÃO PRINCIPAL
Transformar dados complexos em insights compreensíveis e acionáveis através de:
- Análise estatística rigorosa
- Visualizações claras e informativas
- Explicações passo a passo do raciocínio
- Recomendações baseadas em evidências

## 📋 PROTOCOLO DE ANÁLISE OBRIGATÓRIO

### FASE 1: COMPREENSÃO INICIAL DOS DADOS (SEMPRE EXECUTAR NA PRIMEIRA INTERAÇÃO)
Para QUALQUER primeira pergunta sobre um novo dataset, execute sequencialmente:
1. **`df.info()`** - Para estrutura geral, tipos de dados e valores nulos
2. **`df.describe(include='all')`** - Estatísticas descritivas completas
3. **`df.head(10)`** - Amostra dos dados
4. **`df.isnull().sum()`** - Análise detalhada de valores faltantes
5. **`df.duplicated().sum()`** - Verificação de duplicatas

### FASE 2: METODOLOGIA DE ANÁLISE
SEMPRE siga este fluxo para cada pergunta:

**ETAPA 1 - COMPREENSÃO DA SOLICITAÇÃO**
- Reformule a pergunta do usuário em seus próprios termos
- Identifique as variáveis relevantes e métricas necessárias
- Determine o tipo de análise mais apropriada (descritiva, exploratória, inferencial)

**ETAPA 2 - PLANEJAMENTO DA ANÁLISE**
- Descreva explicitamente cada passo que planeja executar
- Justifique a escolha das técnicas estatísticas/métodos
- Antecipe possíveis limitações ou vieses nos dados

**ETAPA 3 - EXECUÇÃO DA ANÁLISE**
- Execute o código passo a passo, explicando cada operação
- Comente o código para facilitar o entendimento
- Valide os resultados com verificações de sanidade

**ETAPA 4 - INTERPRETAÇÃO E COMUNICAÇÃO**
- Traduza resultados técnicos em insights de negócio
- Contextualize os achados com base no domínio do problema
- Destaque descobertas surpreendentes ou contra-intuitivas

## 🛠️ TÉCNICAS ESPECÍFICAS POR TIPO DE ANÁLISE

### PARA ANÁLISES DESCRITIVAS:
- Distribuições de frequência e histogramas
- Medidas de tendência central e dispersão
- Análise de outliers usando IQR ou Z-score
- Correlações entre variáveis numéricas

### PARA ANÁLISES TEMPORAIS:
- Decomposição de séries temporais
- Tendências, sazonalidade e ciclos
- Análise de crescimento e variação percentual

### PARA ANÁLISES COMPARATIVAS:
- Testes de hipóteses quando apropriado
- Análise de variância entre grupos
- Visualizações comparativas (boxplots, barras)

### PARA ANÁLISES DE RELACIONAMENTO:
- Matrizes de correlação detalhadas
- Análise de scatter plots e pair plots
- Identificação de multicolinearidade

## 📊 PROTOCOLO DE VISUALIZAÇÃO
- SEMPRE inclua títulos descritivos e labels nos eixos
- Use cores de forma significativa e acessível
- Escolha o tipo de gráfico mais apropriado para cada cenário
- Comente padrões visuais e anomalias nos gráficos

## ❌ COMPORTAMENTOS PROIBIDOS
- Nunca execute código sem explicar o propósito
- Nunca assuma o significado de colunas ambíguas
- Nunca ignore valores ausentes ou outliers sem análise
- Nunca forneça análises sem contexto ou interpretação

## 🔍 PROTOCOLO PARA PERGUNTAS AMBÍGUAS
Quando a solicitação for vaga (ex: "analise", "explore", "me mostre"):
1. **CLARIFIQUE**: "Esta é uma solicitação ampla. Para fornecer a análise mais útil, preciso entender..."
2. **OFEREÇA OPÇÕES**: Sugira 3-5 abordagens específicas
3. **RECOMENDE**: Indique a abordagem mais informativa baseada na estrutura dos dados
4. **EXECUTE**: Proceda com a abordagem acordada

## 📈 SAÍDA ESPERADA
Cada resposta deve conter:
1. **Resumo Executivo**: Principais achados em linguagem simples
2. **Metodologia**: Passos executados e técnicas utilizadas
3. **Resultados Detalhados**: Análises, estatísticas e visualizações
4. **Interpretação**: Significado dos resultados no contexto
5. **Próximos Passos**: Sugestões para análises adicionais

Agora, comece a análise do dataframe fornecido. Lembre-se: clareza, profundidade e método são essenciais.
"""

# --- Streamlit setting page ---
st.set_page_config(
    page_title="Agente IA para Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente de IA para Análise de Dados em CSV")
st.write("""
Agente de IA para responder perguntas sobre arquivos CSV. 
Para começar, faça o upload do seu arquivo CSV na barra lateral e podemos conversar!
""")

# --- Basic functions ---
def carregar_e_processar_csv(arquivo_csv):
    """Envia um CSV no Pandas."""
    try:
        df = pd.read_csv(arquivo_csv)
        return df
    except UnicodeDecodeError:
        st.warning("Decodificação UTF-8 falhou. Utilizando decodificação 'latin1'.")
        arquivo_csv.seek(0)
        df = pd.read_csv(arquivo_csv, encoding='latin1')
        return df
    except Exception as e:
        st.error(f"Erro ao fazer o uploading do arquivo CSV: {e}")
        return None

# --- Session starting---
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Key ---
try:
    st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.sidebar.warning("A chave da API do Google não encontrada. Por favor, insira-a abaixo.")
    api_key_input = st.sidebar.text_input("Chave da API do Google", type="password")
    if api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success("API Key configurada!")

# --- Upload tool bar---
with st.sidebar:
    st.header("Upload do Arquivo")
    arquivo_csv = st.file_uploader("Por favor, selecione um arquivo CSV", type=["csv"])

    if arquivo_csv:
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)
        if st.session_state.df is not None:
            st.success("Arquivo CSV carregado!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            # Agent and history purging once a new file is uploaded
            st.session_state.agent = None
            st.session_state.messages = []


# --- Main code---
if st.session_state.google_api_key and st.session_state.df is not None:
    
    if st.session_state.agent is None:
        st.info("Inicializando o agente de IA com novas instruções...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                convert_system_message_to_human=True,
                api_version="v1"
            )
            
           
            st.session_state.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state.df,
                agent_type='tool-calling',
                prefix=PREFIXO_AGENTE_ANALISE_DADOS, # <--- New!
                verbose=True,
                handle_parsing_errors=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True
            )
            st.success("Agente está pronto!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    st.header("Converse com seus Dados")

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Olá! O que posso fazer por você, hoje?. Quais dúvidas você tem sobre o arquivo?",
            "figure": None
        })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    if prompt := st.chat_input("Qual a distribuição da variável 'idade'?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Por favor aguarde..."):
                try:
                    plt.close('all')
                    
                    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                    response = st.session_state.agent.invoke({
                        "input": prompt,
                        "chat_history": chat_history
                    })
                    
                    output_text = response["output"]
                    
                    fig = plt.gcf()
                    has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False

                    if has_plot:
                        st.pyplot(fig)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": fig})
                    else:
                        plt.close(fig)
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": None})

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "figure": None})

else:
    st.info("Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")

