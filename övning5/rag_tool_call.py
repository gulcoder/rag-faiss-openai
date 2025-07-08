import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 1. Ladda API-nyckeln
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2. Dokument (kunskapsbas)
documents = [
    "Du återställer lösenord genom att klicka på 'Glömt lösenord'.",
    "För att spåra beställning, gå till 'Mina beställningar' på ditt konto.",
    "Vi accepterar betalning via kort, PayPal och Swish.",
    "Returer är möjliga inom 30 dagar från leverans.",
    "Kundtjänst nås via e-post eller telefon på hemsidan.",
]

# 3. Skapa embeddings och FAISS-index
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

doc_embeddings = [get_embedding(doc) for doc in documents]
dimension = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings).astype('float32'))

# 4. Definiera retrieval-funktion
def get_docs(query):
    query_emb = get_embedding(query)
    D, I = index.search(np.array([query_emb]).astype('float32'), k=3)
    return "\n".join([documents[i] for i in I[0]])

# 5. Användarfråga
user_question = "Hur fungerar returer?"

# 6. Systemprompt med instruktion
system_prompt = """
Du är en hjälpsam assistent. Om du är osäker eller behöver fakta, använd get_docs genom att skriva:
{"action": "get_docs", "query": "min fråga här"}

Om du redan vet svaret, svara direkt.
"""

# 7. Första AI-svar
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ],
    temperature=0.3
).choices[0].message.content.strip()

print("📝 AI-svar:", response)

# 8. Om AI väljer att anropa get_docs:
import json

try:
    action_json = json.loads(response)
    if action_json.get("action") == "get_docs":
        query = action_json.get("query")
        context = get_docs(query)

        follow_up_prompt = f"""
Här är informationen du bad om:

{context}

Besvara nu frågan: {user_question}
"""
        follow_up_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är en hjälpsam assistent."},
                {"role": "user", "content": follow_up_prompt}
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        print("\n📚 Slutgiltigt svar:", follow_up_response)

except json.JSONDecodeError:
    print("\n✅ Ingen retrieval behövdes. Svaret var direkt.")

