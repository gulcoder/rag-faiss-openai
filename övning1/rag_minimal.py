import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Steg 1: Ladda API-nyckeln från .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Steg 2: Skapa dokument
documents = [
    "Solen är en stjärna i mitten av vårt solsystem.",
    "Jorden är den tredje planeten från solen.",
    "Månen kretsar runt jorden.",
    "Vatten kokar vid 100 grader Celsius vid havsnivå."
]

# Steg 3: Funktion för att skapa embedding med nya OpenAI v1.x
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Steg 4: Skapa embeddings för alla dokument
document_embeddings = [get_embedding(doc) for doc in documents]

# Steg 5: Bygg FAISS-index
dimension = len(document_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(document_embeddings).astype('float32'))

# Steg 6: Användarens fråga
query = "Kretsar månen kring jorden eller jorden kring månen?"

# Steg 7: Skapa embedding för frågan
query_embedding = get_embedding(query)

# Steg 8: Sök fram de 3 mest lika dokumenten
D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
retrieved_docs = [documents[i] for i in I[0]]

# Steg 9: Skapa prompt med kontext
context = "\n".join(retrieved_docs)
prompt = f"""
Du är en hjälpsam assistent. Använd kontexten nedan för att besvara användarens fråga så kortfattat som möjligt.

Kontext:
{context}

Fråga: {query}
Svar:
"""

# Steg 10: Skicka prompten till chat completions
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Du är en hjälpsam assistent."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

# Steg 11: Skriv ut svaret
answer = chat_response.choices[0].message.content.strip()
print("Svar:", answer)
