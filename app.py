import os
import re # Import the regular expressions module
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

# --- Load and Preprocess CSV Data ---
try:
    # Load the FAQ data from the CSV file
    df = pd.read_csv('Test_data.csv')
    # Ensure the 'Questions' and 'Answers' columns are strings
    df['Questions'] = df['Questions'].astype(str)
    df['Answers'] = df['Answers'].astype(str) # Also convert answers to string
    questions = df['Questions'].tolist()

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    print("CSV data loaded and vectorized successfully.")
except FileNotFoundError:
    print("Error: Mental_Health_FAQ.csv not found. Please make sure the file is in the correct directory.")
    df = None
    vectorizer = None
    question_vectors = None

# --- Initialize Groq Client ---
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        client = None
    else:
        client = Groq(api_key=groq_api_key)
        print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None


# --- Route for the main page ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route for handling chat messages ---
# --- Route for handling chat messages ---
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    response_message = "I'm sorry, I encountered an error. Please try again."
    source = "Error"
    context_from_csv = ""  # Variable to hold context from the CSV

    # --- Step 1: Find relevant context in the CSV file ---
    if df is not None and vectorizer is not None:
        try:
            user_vector = vectorizer.transform([user_message])
            similarities = cosine_similarity(user_vector, question_vectors)
            most_similar_index = similarities.argmax()
            max_similarity = similarities[0, most_similar_index]

            similarity_threshold = 0.5

            # If a similar question is found, use its answer as context for the AI
            if max_similarity > similarity_threshold:
                context_from_csv = df.iloc[most_similar_index]['Answers']
                source = "Groq AI (from CSV context)"
                print(f"Found relevant context in CSV with similarity: {max_similarity}")
            else:
                source = "Groq AI (General)"
                print("No relevant context in CSV, using general knowledge.")

        except Exception as e:
            print(f"Error during CSV search: {e}")

    # --- Step 2: Always call Groq AI, but provide context if found ---
    if client:
        try:
            # The base system prompt
            system_content = (
                "You are a compassionate and helpful mental health assistant. "
                "Provide supportive and informative answers. Keep your answers concise "
                "and to the point, ideally in 2-3 sentences. Vary your wording each time. "
                "If the user's query is sensitive or indicates a crisis, strongly advise them "
                "to seek help from a professional mental health provider or contact a crisis hotline immediately. "
                "Do not provide medical advice."
            )

            # If we found context in the CSV, add it to the prompt
            if context_from_csv:
                system_content += (
                    "\n\nPlease use the following information to answer the user's question. "
                    f"Rephrase it in a natural and supportive way; do not copy it directly. Context: '{context_from_csv}'"
                )

            # First Groq call → generate the base answer
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_message},
                ],
                model="gemma2-9b-it",
            )
            response_message = chat_completion.choices[0].message.content

            # --- NEW: Refinement step ---
            refine_prompt = (
                "Please rephrase the following response into a slightly different wording, "
                "while keeping the supportive and concise style. Do not make it longer than 2-3 sentences. "
                "Here is the response to rephrase:\n\n"
                f"{response_message}"
            )

            refinement = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful rephrasing assistant."},
                    {"role": "user", "content": refine_prompt},
                ],
                model="gemma2-9b-it",
            )
            response_message = refinement.choices[0].message.content

        except Exception as e:
            print(f"Error calling Groq API: {e}")
            response_message = "I'm having trouble connecting to my AI service right now. Please try again later."
            source = "API Error"
    else:
        response_message = "My AI service is not configured. Please contact the administrator."
        source = "Configuration Error"

    return jsonify({'response': response_message, 'source': source})
