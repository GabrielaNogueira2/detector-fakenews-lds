# **Detector de Fake News - Liga de Data Science (LDS)**



#### Este projeto consiste em uma aplicação web capaz de classificar notícias americanas como Reais ou Falsas (Fake News) utilizando Processamento de Linguagem Natural (NLP) e Machine Learning. A aplicação foi desenvolvida como parte do ciclo de projetos da Liga de Data Science, focando na identificação de padrões linguísticos e de formatação (Nível 1).



###### **Equipe do Projeto**



Este projeto foi desenvolvido colaborativamente por:

João Pacolla: Estruturação e Coleta de Dados

Matheus Schartner: Limpeza de Dados e NLP

Victor Godoy: Modelagem e Treinamento

Renan Ribeiro: Análise de Métricas e Avaliação

Gabriela Nogueira: Documentação e Implementação Web 



###### **Sobre o Modelo**



O modelo utiliza Regressão Logística treinada sobre vetores TF-IDF.

Acurácia: ~99%Dados: Dataset de notícias americanas (Reuters vs Fake News).

Insight Principal: O modelo detecta com alta precisão o "estilo" da escrita. 

Notícias reais tendem a ser formais e padronizadas, enquanto Fake News abusam de sensacionalismo e formatação irregular. 



###### **Como Rodar Localmente**



Se você quiser rodar este projeto no seu computador, siga os passos abaixo:

Clone o repositório: git clone \[https://github.com/SEU-USUARIO/detector-fakenews-lds.git](https://github.com/SEU-USUARIO/detector-fakenews-lds.git)

Instale as dependências: pip install -r requirements.txt

Execute a aplicação: streamlit run app\_fakenews.py (Ou use python -m streamlit run app\_fakenews.py se estiver no Windows)

Upload dos Dados: Ao abrir a aplicação, faça o upload dos arquivos True.csv e Fake.csv na barra lateral para treinar o modelo.



###### **Stack Tecnológica**



Linguagem: Python

Interface: Streamlit

Machine Learning: Scikit-Learn

Manipulação de Dados: Pandas \& Numpy

