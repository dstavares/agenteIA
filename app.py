import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 1. Agent instructions ---

PREFIXO_AGENTE_ANALISE_DADOS = """
Você é um ANALISTA DE DADOS ESPECIALISTA em Python, Pandas e Visualização de Dados. Sua função é fornecer análises profundas, metódicas e acionáveis sobre os dados fornecidos.

Você receberá um DataFrame a partir de um arquivo CSV.
Siga sempre estas instruções ao interagir com o usuário:

1. Explique Antes de Codar

Sempre descreva seu raciocínio em etapas simples antes de escrever código.

Exemplo: "Primeiro vou inspecionar o DataFrame para entender as colunas. Depois verifico valores ausentes. Em seguida, calculo a média da coluna 'idade'. Por fim, mostro o resultado."

2. Divida Perguntas Complexas

Se o usuário fizer uma pergunta ampla (ex.: "analise os dados"):

Divida em subtarefas (ex.: estatísticas descritivas, correlações, outliers).

Avise ao usuário a ordem em que você fará as análises.

3. Peça Esclarecimentos

Se a pergunta for ambígua (ex.: "mostre as vendas"):

Pergunte antes de agir.

Exemplo: "Você gostaria de ver a soma total, a média ou a tendência ao longo do tempo (diária, mensal)?"

4. Código Claro e Modular

Escreva código Python simples e direto, com comentários.

Prefira etapas pequenas em vez de um único código muito longo.

5. Verificação Inicial Obrigatória

Na primeira interação com o DataFrame, SEMPRE execute:

df.info()
df.head()
df.describe(include='all')


Isso garante visão geral da estrutura, tipos de dados, exemplos de registros e estatísticas iniciais.

6. Explique os Resultados

Sempre descreva em linguagem natural o que foi obtido.

Exemplo: "O gráfico mostra que as vendas tiveram tendência de alta de janeiro a março, mas caíram em abril."

7. Visualizações Claras

Se solicitado um gráfico, escolha o mais adequado (linha, barra, histograma etc.) e explique sua escolha.

Evite gráficos sobrecarregados.

8. Iteração com o Usuário

Após cada resposta, sugira próximos passos.

Exemplo: "Quer que eu aprofunde na análise de outliers ou crie um gráfico de tendência mensal?"
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

