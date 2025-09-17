import os
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
    df['Answers'] = df['Answers'].astype(str)
    questions = df['Questions'].tolist()

    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    print("CSV data loaded and vectorized successfully.")
except FileNotFoundError:
    print("Error: Test_data.csv not found. Please make sure the file is in the correct directory.")
    df = None

# --- Initialize Groq Client ---
try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables.")
        client = None
    else:
        client = Groq(api_key=groq_api_key)
        print("Groq client initialized successfully.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main chat page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat logic: Find in CSV, then refine with AI."""
    if df is None:
        return jsonify({
            'response': 'I am unable to provide answers right now. The data file is missing.', 
            'source': 'Error', 
            'similarity_score': 0
        })

    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({
            'response': 'Please enter a message.', 
            'source': 'System', 
            'similarity_score': 0
        })

    # --- Step 1: Find the best answer from the CSV ---
    user_vector = vectorizer.transform([user_message])
    similarities = cosine_similarity(user_vector, question_vectors)
    most_similar_index = similarities.argmax()
    highest_similarity_score = similarities[0, most_similar_index]
    
    SIMILARITY_THRESHOLD = 0.24 # You can adjust this value

    if highest_similarity_score > SIMILARITY_THRESHOLD:
        # Retrieve the best answer from the CSV
        retrieved_answer = df['Answers'].iloc[most_similar_index]
        matched_question = df['Questions'].iloc[most_similar_index]
        
        # Format the source string to include the score
        source = f"Matched: '{matched_question}' (Score: {highest_similarity_score:.2%})"

        # --- Step 2: Use AI to refine the retrieved answer ---
        if client:
            try:
                refine_prompt = (
                    "Please rephrase the following text to be more supportive, empathetic, and conversational, "
                    "while keeping the core information intact. Keep the response to 2-3 sentences.\n\n"
                    "Original Text:\n"
                    f"\"{retrieved_answer}\""
                )

                refinement_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert at rephrasing text to be more empathetic and clear."},
                        {"role": "user", "content": refine_prompt},
                    ],
                    model="gemma2-9b-it",
                )
                response_message = refinement_completion.choices[0].message.content
            except Exception as e:
                print(f"Error calling Groq API for refinement: {e}")
                # Fallback to the original answer if AI fails
                response_message = retrieved_answer
                source += " (AI refinement failed)"
        else:
            # If the client isn't configured, just use the original answer
            response_message = retrieved_answer
            source += " (AI not configured)"
    else:
        # If no good match is found, provide a default response without calling AI
        response_message = "I'm not sure I have information on that. Could you please try asking in a different way?"
        source = "Default Fallback"

    return jsonify({
        'response': response_message, 
        'source': source,
        'similarity_score': float(highest_similarity_score) # Return the score
    })

if __name__ == '__main__':
    app.run(debug=True)