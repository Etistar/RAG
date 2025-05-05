from google import genai
from google.genai import types
from IPython.display import Markdown
from dotenv import load_dotenv
import os
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry
from google.genai import types
import chromadb

# Charge le .env situé dans le même dossier
load_dotenv()

# Vérifie que la variable est bien chargée
api = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api)

import re

def split_text_into_paragraphs(text):
    # Séparation des paragraphes en utilisant une ligne vide
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return paragraphs

# Lecture du fichier
with open("output_impact_all.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks = split_text_into_paragraphs(text)

Document = []  # Initialisation de la liste

for chunk in chunks:
    Document.append(chunk)  # Ajout du paragraphe à la liste


# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]


DB_NAME = "ILM3"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

db.add(documents=Document, ids=[str(i) for i in range(len(Document))])


def answer (query): 
    # Switch to query mode when generating embeddings.
    embed_fn.document_mode = False


    result = db.query(query_texts=[query], n_results=1)
    [all_passages] = result["documents"]

    query_oneline = query.replace("\n", " ")

    # This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below. 
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
    However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
    strike a friendly and converstional tone. If the passage is irrelevant to the answer, just say 
    Concact ILM on their phone numbers are +233559643898, +233268896068, and +233260909144 for more directions. Act like a human
    and do not talk about a text. Please also answer in the query language. Adresse , localisation are synonymes.

    QUESTION: {query_oneline}
    """

    # Add the retrieved documents to the prompt.
    for passage in all_passages:
        passage_oneline = passage.replace("\n", " ")
        prompt += f"PASSAGE: {passage_oneline}\n"

    answer = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt)

    return answer.text