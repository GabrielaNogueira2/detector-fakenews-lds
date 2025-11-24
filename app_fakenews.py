import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Detector de Fake News - LDS",
    page_icon="🕵️‍♀️",
    layout="wide"
)

# --- ESTILOS CUSTOMIZADOS (CSS) ---
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #FF4B4B; text-align: center; margin-bottom: 1rem;}
    .sub-text {text-align: center; color: #555;}
    .result-box {padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;}
    .safe {background-color: #D4EDDA; color: #155724; border: 2px solid #C3E6CB;}
    .fake {background-color: #F8D7DA; color: #721C24; border: 2px solid #F5C6CB;}
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÕES DE LIMPEZA (Baseadas no seu Notebook) ---
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# --- CACHE E TREINAMENTO DO MODELO ---
# Usamos @st.cache_resource para não treinar toda vez que a página recarregar
@st.cache_resource
def train_model(uploaded_true, uploaded_fake):
    """
    Treina o modelo se os arquivos forem fornecidos.
    Retorna o pipeline treinado (Vetorizador + Modelo).
    """
    try:
        # Carregando dados
        df_true = pd.read_csv(uploaded_true)
        df_fake = pd.read_csv(uploaded_fake)
        
        # Criando coluna alvo (Target)
        df_true["class"] = 1 # Notícia Real
        df_fake["class"] = 0 # Fake News
        
        # Juntando os dataframes (Conforme seu notebook)
        # O notebook remove as últimas 10 linhas para teste manual, 
        # mas aqui usaremos tudo para treinar o "motor" do site.
        df_merge = pd.concat([df_fake, df_true], axis=0)
        
        # Pré-processamento essencial (Criação da coluna Full Text)
        df_merge["full_text"] = df_merge["title"] + " " + df_merge["text"]
        df_merge["full_text"] = df_merge["full_text"].apply(wordopt)
        
        # Definindo X e Y
        X = df_merge["full_text"]
        y = df_merge["class"]
        
        # Divisão Treino/Teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        # Pipeline: Vetorização -> Classificação
        # Pipeline garante que o texto novo sofra as mesmas transformações do treino
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression())
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Validação rápida
        pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, pred)
        
        return pipeline, acc

    except Exception as e:
        return None, str(e)

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2910/2910768.png", width=100)
    st.title("Sobre o Projeto")
    st.markdown("**Liga de Data Science**")
    
    st.markdown("### Equipe Técnica:")
    st.markdown("""
    - **João Pacolla:** Estrutura/Dados
    - **Matheus Schartner:** Limpeza/NLP
    - **Victor Godoy:** Modelagem
    - **Renan Ribeiro:** Avaliação
    - **Gabriela Nogueira:** Documentação
    """)
    
    st.info("""
    **Contexto:**
    Este projeto utiliza Processamento de Linguagem Natural (NLP) para classificar notícias 
    baseando-se no estilo de escrita e formatação (Nível 1).
    """)

    st.markdown("---")
    st.subheader("⚙️ Configuração do Modelo")
    
    # Upload dos datasets caso o modelo não exista na memória
    st.markdown("Para o site funcionar, precisamos treinar o modelo. Faça upload dos CSVs originais (True.csv e Fake.csv).")
    upl_true = st.file_uploader("Carregar True.csv", type="csv")
    upl_fake = st.file_uploader("Carregar Fake.csv", type="csv")

# --- ÁREA PRINCIPAL ---

st.markdown('<div class="main-header">Detector de Fake News 🇺🇸</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Insira o título e o texto da notícia para verificar sua veracidade.</div>', unsafe_allow_html=True)

st.write("") # Espaçamento

# Lógica de Inicialização
model = None
accuracy = 0

if upl_true and upl_fake:
    with st.spinner('Treinando o modelo com seus dados... aguarde um momento...'):
        model, accuracy = train_model(upl_true, upl_fake)
    
    if isinstance(model, str): # Se retornou string, é erro
        st.error(f"Erro ao treinar: {model}")
    else:
        st.success(f"Modelo treinado com sucesso! Acurácia estimada: {accuracy:.2%}")

else:
    st.warning("⚠️ Por favor, faça o upload dos arquivos `True.csv` e `Fake.csv` na barra lateral para ativar o sistema.")

# Formulário de Entrada
with st.form("prediction_form"):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("### 1. Título")
        title_input = st.text_area("Cole o título aqui", height=150, placeholder="Ex: Trump Says...")
    
    with col2:
        st.write("### 2. Corpo da Notícia")
        text_input = st.text_area("Cole o texto completo aqui", height=150, placeholder="Ex: Washington (Reuters) - ...")
    
    submit_btn = st.form_submit_button("🔍 Verificar Veracidade", type="primary")

# Lógica de Predição
if submit_btn:
    if model is None:
        st.error("O modelo ainda não foi treinado. Use a barra lateral para carregar os dados.")
    elif not title_input and not text_input:
        st.warning("Por favor, preencha pelo menos um dos campos.")
    else:
        # Prepara o texto (Mesma lógica do notebook: Title + Text)
        full_text_input = str(title_input) + " " + str(text_input)
        processed_text = wordopt(full_text_input)
        
        # Predição
        prediction = model.predict([processed_text])[0]
        probabilidade = model.predict_proba([processed_text]).max()
        
        st.markdown("---")
        st.subheader("Resultado da Análise:")
        
        if prediction == 1:
            st.markdown(
                f'<div class="result-box safe">✅ NOTÍCIA VERDADEIRA (REAL)<br><span style="font-size:16px">Confiança do modelo: {probabilidade:.2%}</span></div>', 
                unsafe_allow_html=True
            )
            st.balloons()
        else:
            st.markdown(
                f'<div class="result-box fake">🚨 FAKE NEWS DETECTADA<br><span style="font-size:16px">Confiança do modelo: {probabilidade:.2%}</span></div>', 
                unsafe_allow_html=True
            )
            
        # Explicabilidade Simples (Insights do Relatório)
        with st.expander("ℹ️ Entenda como o modelo decidiu"):
            st.write("""
            O modelo analisa padrões linguísticos. Segundo o relatório do projeto (Sprint 4):
            - **Notícias Reais:** Tendem a ter linguagem formal, citar agências (ex: Reuters) e ter estrutura padrão.
            - **Fake News:** Costumam usar linguagem sensacionalista, muitos adjetivos e formatação irregular.
            """)
            st.write(f"**Texto processado que o modelo 'leu':**")
            st.caption(processed_text[:500] + "...")

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido para o Projeto DS - Análise de Fake News Americanas")