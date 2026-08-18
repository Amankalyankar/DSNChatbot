import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# =========================================================
# LOAD CSV DATA
# =========================================================

try:
    df = pd.read_csv("Test_data.csv")

    df["Questions"] = df["Questions"].astype(str)
    df["Answers"] = df["Answers"].astype(str)

    questions = df["Questions"].tolist()

    # Better TF-IDF:
    # Word n-grams understand phrases
    # Character n-grams help with differently-worded questions / spelling
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    question_vectors = vectorizer.fit_transform(questions)

    print(f"CSV loaded successfully: {len(df)} questions")

except FileNotFoundError:
    print("ERROR: Test_data.csv not found.")
    df = None


# =========================================================
# GROQ SETUP
# =========================================================

try:
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        client = None

    else:
        client = Groq(api_key=groq_api_key)
        print("Groq client initialized successfully.")

except Exception as e:
    print(f"Error initializing Groq: {e}")
    client = None


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def index():
    return render_template("index.html")


# =========================================================
# CHAT
# =========================================================

@app.route("/chat", methods=["POST"])
def chat():

    # -----------------------------------------------------
    # Check CSV
    # -----------------------------------------------------

    if df is None:
        return jsonify({
            "response": "The CSV knowledge base is missing.",
            "source": "Error",
            "similarity_score": 0
        })


    # -----------------------------------------------------
    # Get user message
    # -----------------------------------------------------

    user_message = request.json.get("message", "").strip()

    if not user_message:
        return jsonify({
            "response": "Please enter a message.",
            "source": "System",
            "similarity_score": 0
        })


    # =====================================================
    # STEP 1: FIND SIMILAR QUESTIONS
    # =====================================================

    user_vector = vectorizer.transform([user_message])

    similarities = cosine_similarity(
        user_vector,
        question_vectors
    )[0]

    # Get top 3 matches
    top_indices = similarities.argsort()[-3:][::-1]

    best_index = top_indices[0]
    best_score = similarities[best_index]

    matched_question = df["Questions"].iloc[best_index]
    matched_answer = df["Answers"].iloc[best_index]


    # =====================================================
    # RELEVANCE THRESHOLD
    # =====================================================

    # You can adjust this later.
    SIMILARITY_THRESHOLD = 0.10

    if best_score < SIMILARITY_THRESHOLD:

        return jsonify({
            "response": (
                "I'm sorry, but I couldn't find enough relevant "
                "information in my knowledge base to answer that."
            ),
            "source": "No relevant CSV match",
            "similarity_score": float(best_score)
        })


    # =====================================================
    # STEP 2: BUILD CONTEXT FROM TOP 3 CSV MATCHES
    # =====================================================

    context = ""

    for rank, index in enumerate(top_indices, start=1):

        score = similarities[index]

        # Only include reasonably relevant matches
        if score >= SIMILARITY_THRESHOLD:

            context += f"""
--- CSV RESULT {rank} ---
Question:
{df["Questions"].iloc[index]}

Answer:
{df["Answers"].iloc[index]}

Similarity:
{score:.2%}

"""


    # =====================================================
    # STEP 3: ASK GROQ
    # =====================================================

    if client:

        try:

            prompt = f"""
You are an assistant that answers questions using a CSV knowledge base.

USER QUESTION:
{user_message}

CSV KNOWLEDGE BASE:
{context}

IMPORTANT RULES:

1. Use ONLY the information contained in the CSV knowledge base.
2. Do NOT add facts from your own knowledge.
3. Do NOT invent statistics, diagnoses, symptoms, treatments,
   recommendations, or other information.
4. Combine information from the CSV results only when appropriate.
5. If the CSV does not contain enough information to answer the
   question, clearly say that the knowledge base does not contain
   enough information.
6. Answer the user's actual question directly.
7. Keep the answer clear, supportive, and conversational.
8. Do not mention "TF-IDF", similarity scores, or internal processing.
9. Do not say "according to result 1" or expose the retrieval process.
"""

            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful knowledge-base assistant. "
                            "You must stay strictly grounded in the "
                            "provided CSV information."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=300
            )

            response_message = completion.choices[0].message.content.strip()

            source = (
                f"Best CSV match: '{matched_question}' "
                f"(Score: {best_score:.2%})"
            )


        except Exception as e:

            print(f"Groq API error: {e}")

            # ---------------------------------------------
            # FALLBACK TO ORIGINAL CSV ANSWER
            # ---------------------------------------------

            response_message = matched_answer

            source = (
                f"CSV fallback: '{matched_question}' "
                f"(Score: {best_score:.2%})"
            )

    else:

        # =================================================
        # GROQ NOT CONFIGURED
        # =================================================

        response_message = matched_answer

        source = (
            f"CSV answer: '{matched_question}' "
            f"(Score: {best_score:.2%})"
        )


    # =====================================================
    # RETURN RESPONSE
    # =====================================================

    return jsonify({
        "response": response_message,
        "source": source,
        "similarity_score": float(best_score)
    })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)