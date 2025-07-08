import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 1. Ladda API-nyckeln
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 2. Skapa FAQ-lista (Exempel: Lägg till fler i praktiken)
faq_list = [
    {"question": "Hur återställer jag mitt lösenord?", "answer": "Du kan återställa ditt lösenord via länken 'Glömt lösenord' på inloggningssidan."},
    {"question": "Hur kontaktar jag kundtjänst?", "answer": "Du kan nå vår kundtjänst via e-post eller telefon, kontaktuppgifter finns på vår hemsida."},
    {"question": "Vilka betalningsmetoder accepterar ni?", "answer": "Vi accepterar kreditkort, PayPal och direktbetalning via bank."},
    {"question": "Hur spårar jag min beställning?", "answer": "Logga in på ditt konto och gå till 'Mina beställningar' för att spåra din order."},
    {"question": "Kan jag returnera en vara?", "answer": "Ja, du har 30 dagars returrätt från att du mottagit varan."},
    # Lägg till upp till 50 FAQ-par för verklig övning
]

# 3. Skapa embeddings för FAQ-poster
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

faq_texts = [f"Fråga: {item['question']} Svar: {item['answer']}" for item in faq_list]
faq_embeddings = [get_embedding(text) for text in faq_texts]

# 4. Bygg FAISS-index
dimension = len(faq_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(faq_embeddings).astype('float32'))

# 5. Användarens fråga
user_question = "Hur kan jag betala?"

# 6. Embedding för frågan
question_embedding = get_embedding(user_question)

# 7. Hitta de 5 mest lika FAQ-posterna
D, I = index.search(np.array([question_embedding]).astype('float32'), k=5)
retrieved_faqs = [faq_texts[i] for i in I[0]]

# 8. Skapa prompt med källor
sources = "\n\n".join(retrieved_faqs)
prompt = f"""
Besvara användarens fråga med hjälp av följande FAQ-rader. Citera de rader du använder.

FAQ:
{sources}

Fråga: {user_question}
Svar:
"""

# 9. Skicka prompten till chat completions
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Du är en hjälpsam assistent som alltid citerar FAQ-källor."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.3
)

# 10. Skriv ut svar + källor
answer = chat_response.choices[0].message.content.strip()
print("Svar:\n", answer)
print("\nAnvända FAQ-källor:")
for idx in I[0]:
    print("-", faq_list[idx]["question"])
