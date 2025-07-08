# 🚀 RAG OpenAI Exercises – Retrieval-Augmented Generation från grunden till avancerat

En praktisk samling av övningar som lär dig bygga kraftfulla **Retrieval-Augmented Generation (RAG)**-system med **OpenAI, FAISS, BM25** och **funktionella verktygsanrop**.

Det här projektet tar dig från grunderna i **embedding-baserad retrieval** till **hybrid-sökning** och **adaptiva RAG-loopar** – med fokus på verkliga tillämpningar som FAQ-botar, självkritiska AI-svar och AI-driven informationssökning.

---

## 🏗 Tech Stack

| Teknologi | Användning |
|-----------|------------|
| **Python 3.x** | Backend & scripting |
| **OpenAI API (gpt-4-turbo, embeddings)** | Språkmodell & vector embeddings |
| **FAISS** | Snabb vektorbaserad likhetssökning |
| **Rank-BM25 (Okapi BM25)** | Klassisk textbaserad retrieval |
| **NumPy** | Numeriska operationer |
| **dotenv** | Hantering av API-nycklar |
| **tiktoken** | Token-beräkningar (om tillämpligt) |

---

## 📚 Innehåll – Övningar

| Övning | Beskrivning |
|--------|-------------|
| **1. Minsta möjliga RAG** | Bygg en minimal RAG-pipeline: ingest → retrieval → generation. |
| **2. FAISS-FAQ-bot** | Skapa en FAQ-bot som hämtar de mest relevanta dokumenten och citerar sina källor. |
| **3. Hybrid-RAG (BM25 + Embedding)** | Kombinera klassisk BM25 och semantisk retrieval för bättre precision och transparens. |
| **4. RAG med Refine-strategi** | Implementera en självförbättrande loop där modellen markerar osäkerhet och ber om mer kontext. |
| **5. Funktionskallad RAG som verktyg** | Exponera retrieval som ett verktyg som AI:n själv kan anropa för förbättrade svar. |

---

## 🔑 Funktioner & Mål

✅ Förstå grunderna i **RAG** (Retrieval-Augmented Generation)  
✅ Bygg en **FAQ-bot med FAISS** och semantiska embeddings  
✅ Kombinera **BM25 och embeddings** för hybrid-retrieval  
✅ Implementera en **Refine-strategi** där modellen markerar osäkerhet och itererar sitt svar  
✅ Använd **OpenAI function calling** för retrieval-orkestrering  
✅ Skapa **transparens och källkritik** i AI-svar  
✅ Utveckla förståelse för **prompt engineering** och **vektorbaserad informationssökning**

---

## 🌍 Användningsområden

Denna uppsättning övningar visar hur RAG-principen kan användas i verkliga tillämpningar som:

- **FAQ-botar** som alltid ger källhänvisningar
- **Enterprise AI-assistenter** med intern kunskapsbas
- **Dokumentanalys** och **juridisk rådgivning**
- **Självkritiska AI-system** som flaggar osäkerhet innan de svarar
- **AI med tillgång till externa verktyg och datakällor**

---

## 🔮 Vidareutveckling & Framtida Möjligheter

Här är några förslag för hur du kan bygga vidare på projektet:

- 📄 **Lägg till fler dokumentdatabaser** (t.ex. PDF, webbsidor, Notion, Google Drive).
- 🧩 **Byt embeddingsmodell** till lokal transformer-modell från Hugging Face för ökad kontroll.
- 🔍 **Experimentera med vektordatabaser** som Pinecone, Weaviate eller Chroma för skalbarhet.
- 🌐 **Skapa en interaktiv webbfront** (Streamlit, React eller Next.js) som kopplas mot RAG-backend.
- 🧠 **Bygg multi-turn conversational RAG** där kontext hålls mellan användarfrågor.
- ⚙️ **Integrera fler verktygsanrop** (t.ex. sök i API:er, realtidsdata, databasanrop).
- 📝 **Tokenoptimering & kostnadsanalys** vid användning av LLM.

---

## 🛡 Säkerhet & Etik

Denna typ av AI-system kräver ansvarsfull användning:

- **Verifiera källor** – Använd alltid transparens med var informationen kommer från.
- **Skydda användardata & API-nycklar** – `.env` och `.gitignore` används för att undvika läckor.
- **Undvik hallucinationer** – Genom RAG minskar risken för falska svar, men den elimineras inte helt.
- **Förstå modellens begränsningar** – Ingen AI är perfekt; RAG hjälper men ersätter inte mänsklig granskning.

---

## 📜 Licens

MIT License  
Fritt att använda, modifiera och bygga vidare på – med hänsyn till ansvarsfull användning.

---

## 🙏 Tack & Inspiration

Den här serien av övningar är inspirerad av den snabba utvecklingen inom **AI Retrieval-Augmented Generation**, **function calling** och **open-source vector search**. Målet är att ge praktiska kunskaper som går att applicera i allt från små projekt till stora AI-drivna affärssystem.

---

