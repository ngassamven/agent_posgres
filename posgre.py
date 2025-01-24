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
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    return conn

# Fonction pour exécuter des requêtes SQL et récupérer les résultats sous forme de liste
def query_as_list(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    res = cursor.fetchall()
    cursor.close()  # Fermeture explicite du curseur
    res = [el for sub in res for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# Connexion à la base de données PostgreSQL
connection = get_pg_connection()

# Récupération des clients depuis la base de données
clients = query_as_list(connection, "SELECT name FROM clients")

# Création des embeddings OpenAI pour les clients
embeddings = OpenAIEmbeddings()

# Créer la base de données vectorielle avec FAISS
vector_db = FAISS.from_texts(clients, embeddings)

# Créer un retriever à partir de la base vectorielle
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Définir la description pour l'outil de recherche
description = """Utilisez cet outil pour rechercher des noms propres. 
L'entrée est une orthographe approximative d'un nom propre, et la sortie est un nom propre valide. 
Utilisez le nom le plus similaire à la recherche."""

retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

# Créer un agent SQL avec LangChain
db = SQLDatabase.from_uri(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Sélecteur d'exemples pour l'agent
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

# Définir le prompt pour l'agent
system_prefix = """Vous êtes un agent conçu pour interagir avec une base de données SQL. 
Lorsqu'une question est posée, créez une requête SQL correcte pour l'exécuter, 
puis analysez les résultats de la requête et renvoyez la réponse.
Ne jamais exécuter de requêtes DML (INSERT, UPDATE, DELETE, DROP etc.) sur la base de données.
"""

# Correction : Ajout du paramètre `suffix`
few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "Entrée de l'utilisateur: {input}\nRequête SQL: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix="\nVeuillez générer une requête SQL appropriée en fonction de l'entrée de l'utilisateur."
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Créer l'agent LangChain
agent_executor = create_sql_agent(llm, db=db, prompt=full_prompt, verbose=True, agent_type="openai-tools")

# Exemple d'utilisation de l'agent
response = agent_executor.invoke("Combien de clients y a-t-il dans la base de données ?")
print(response)

# Correction ici : Utiliser "query" au lieu de "input"
retriever_result = retriever_tool.invoke({"query": "John Doe"})
print(retriever_result)
