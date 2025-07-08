import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

# 1. Ladda API-nyckel
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2. FAQ-dokument (Exempel)
documents = [
    "Kunder kan återställa sina lösenord genom att klicka på 'Glömt lösenord' vid inloggning.",
    "För att spåra en beställning, logga in och klicka på 'Mina beställningar'.",
    "Vi accepterar betalning med kort, PayPal och direktbetalning via bank.",
    "Returer accepteras inom 30 dagar efter mottagandet av varan.",
    "Kundtjänst nås via e-post eller telefon som anges på vår hemsida.",
]

# 3. BM25-index
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# 4. Embedding-index (FAISS)
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

embeddings = [get_embedding(doc) for doc in documents]
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# 5. Fråga
user_question = "Hur byter jag lösenord?"

# 6. BM25 Retrieval
bm25_scores = bm25.get_scores(user_question.lower().split())
top_bm25_indices = np.argsort(bm25_scores)[::-1][:3]
bm25_results = [documents[i] for i in top_bm25_indices]

# 7. Embedding Retrieval
query_embedding = get_embedding(user_question)
D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
embedding_results = [documents[i] for i in I[0]]

# 8. Kombinera källor (ingen dubblett)
combined_sources = list(dict.fromkeys(bm25_results + embedding_results))

# 9. Skapa prompt med motivering
sources_text = "\n".join(f"- {src}" for src in combined_sources)
prompt = f"""
Besvara frågan nedan baserat på följande källor, som valts både med BM25 (sök på ord) och embeddings (semantisk likhet). Citera gärna källorna i svaret.

Källor:
{sources_text}

Fråga: {user_question}
Svar:
"""

# 10. Generera svar
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Du är en hjälpsam assistent som förklarar dina källor."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

# 11. Visa svaret
answer = chat_response.choices[0].message.content.strip()
print("Svar:\n", answer)

print("\n🔍 Använda källor:")
print("BM25:", [documents[i] for i in top_bm25_indices])
print("Embeddings:", [documents[i] for i in I[0]])
