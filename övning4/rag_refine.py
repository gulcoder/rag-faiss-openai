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

# 3. Embeddings och FAISS-index
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

# 4. Användarfråga
user_question = "Vad är kundtjänstens telefonnummer?"

# 5. Första vändan: Direkt svar (kort + JAG ÄR OSÄKER om oklart)
first_prompt = f"""
Svara kort på följande fråga. Om du är osäker, skriv exakt: JAG ÄR OSÄKER.

Fråga: {user_question}
Svar:
"""

first_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Du är en kortfattad assistent."},
        {"role": "user", "content": first_prompt}
    ],
    temperature=0.3
).choices[0].message.content.strip()

print("Första svar:", first_response)

# 6. Om osäker → hämta kontext och refine
if "JAG ÄR OSÄKER" in first_response.upper():
    query_embedding = get_embedding(user_question)
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=3)
    context = "\n".join([documents[i] for i in I[0]])

    refine_prompt = f"""
Här är mer kontext och ditt första svar. Skriv ett förbättrat, korrekt svar.

Kontext:
{context}

Första svar: {first_response}

Förbättrat svar:
"""

    refined_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Du är en hjälpsam assistent som förbättrar svar med hjälp av kontext."},
            {"role": "user", "content": refine_prompt}
        ],
        temperature=0.3
    ).choices[0].message.content.strip()

    print("\nFörbättrat svar:", refined_response)

else:
    print("\nSlutgiltigt svar (ingen refine behövdes).")

