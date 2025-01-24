import streamlit as st
import psycopg2
import os
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Connexion à la base de données PostgreSQL
def get_pg_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

# Fonction pour exécuter une requête SQL
def query_as_list(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    res = cursor.fetchall()
    cursor.close()
    res = [el for sub in res for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Connexion à PostgreSQL
connection = get_pg_connection()

# Récupération des clients depuis la base de données
clients = query_as_list(connection, "SELECT name FROM clients")

# Création des embeddings et base de données vectorielle FAISS
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_texts(clients, embeddings)

# Création d'un retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Définition de l'outil de recherche
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description="Cherchez des noms approximatifs et obtenez la version correcte."
)

# Création de la connexion SQL pour LangChain
db = SQLDatabase.from_uri(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Initialisation du modèle OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Sélecteur d'exemples pour aider l'agent SQL
examples = [
    {"input": "Listez tous les clients.", "query": "SELECT * FROM clients;"},
    {"input": "Trouvez le client avec le nom 'John Doe'.", "query": "SELECT * FROM clients WHERE name = 'John Doe';"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
    FAISS,
    k=5,
    input_keys=["input"],
)

# Définition du prompt pour l'agent SQL
system_prefix = """Vous êtes un agent SQL. Créez et exécutez des requêtes SQL pour répondre aux questions. 
Ne modifiez pas la base de données (INSERT, UPDATE, DELETE)."""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template("Entrée: {input}\nRequête SQL: {query}"),
    input_variables=["input"],
    prefix=system_prefix,
    suffix="\nGénérez une requête SQL correcte en fonction de l'entrée."
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Création de l'agent SQL
agent_executor = create_sql_agent(llm, db=db, prompt=full_prompt, verbose=True, agent_type="openai-tools")

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Agent SQL avec LangChain", page_icon="📊")

st.title("🔍 Assistant SQL avec LangChain & FAISS")
st.write("Interrogez votre base de données PostgreSQL en langage naturel.")

# --- Recherche d'un client avec FAISS ---
st.header("📌 Recherche de clients")
query_name = st.text_input("Entrez un nom approximatif :", "")

if st.button("🔍 Rechercher"):
    if query_name:
        results = retriever_tool.invoke({"query": query_name})
        st.write("🔹 **Résultats les plus proches :**")
        st.write(results)
    else:
        st.warning("Veuillez entrer un nom.")

# --- Agent SQL pour interroger la base ---
st.header("📌 Requête SQL en langage naturel")
user_query = st.text_area("Posez une question sur la base de données :", "")

if st.button("🖥️ Exécuter la requête"):
    if user_query:
        response = agent_executor.invoke(user_query)
        st.write("📊 **Résultat :**")
        st.write(response)
    else:
        st.warning("Veuillez entrer une requête.")

# --- Fermeture de la connexion ---
connection.close()
