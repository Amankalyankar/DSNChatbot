# Mental Health Assistant Chatbot

This is a proof-of-concept chatbot designed to answer mental health-related questions. It's built with a Flask backend and a straightforward HTML, Tailwind CSS, and JavaScript frontend.

Initially, this project functions as a **retrieval-based model**. It uses a TF-IDF Vectorizer and Cosine Similarity to find the most relevant answer from a knowledge base (`Test_data.csv`). This README also guides you on how to upgrade it to a **Retrieval-Augmented Generation (RAG) model** using the Groq API.

***

##  Features

* **Simple & Clean UI**: A responsive and intuitive chat interface.
* **Similarity-Based Responses**: Leverages a TF-IDF Vectorizer and Cosine Similarity to find the best-matching question from the knowledge base.
* **RAG Upgrade Path**: Includes instructions and code to convert the chatbot into a generative model with Groq.
* **Easy to Customize**: The dataset and similarity threshold can be easily modified.

***

##  How It Works

The application is a web-based chatbot. When a user sends a message, it is sent to the Flask backend.

1.  **Retrieval**: The backend loads a CSV file of questions and answers. It uses a TF-IDF vectorizer to convert the user's query into a vector and calculates the cosine similarity against all question vectors in the CSV.
2.  **Response**:
    * **Retrieval-Only (Default)**: If the highest similarity score is above a threshold, the corresponding pre-written answer is returned directly.
    * **RAG (Upgraded)**: The best-matching answer from the CSV is used as **context**. This context is then sent to a Large Language Model (LLM) via the Groq API, which **generates** a new, conversational response.

```
##  Project Structure
├── .env                # To store your Groq API key
├── app.py              # The main Flask application
├── requirements.txt      # Python dependencies
├── Test_data.csv         # The knowledge base for the chatbot
├── templates
  └── index.html      # The HTML template for the chat interface
```

##  Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aman-627/DSNChatbot.git
    cd DSNChatbot
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Set up your API key:**
    * Create a file named `.env` in the root of your project directory.
    * Get a free API key from [GroqCloud](https://console.groq.com/keys).
    * Add the following line to your `.env` file:
        ```
        GROQ_API_KEY="YOUR_API_KEY_HERE"
        ```

***

##  Usage

1.  **Run the Flask application:**
    ```bash
    flask run
    ```
    or
    ```bash
    python app.py
    ```
2.  **Open your web browser and navigate to:**
    [http://127.0.0.1:5000](http://127.0.0.1:5000)

You should see the chat interface. Type a question to get a response.

***

##  From Retrieval to RAG: Upgrading to a Generative Model

This project is a perfect starting point for building a RAG model. The "Retrieval" part is already done. The only missing piece is the **"Augmented Generation,"** which we can add using `groq`.

### Why is this not a RAG model by default?

The default code performs **retrieval only**. It finds the most relevant pre-written answer from the CSV and returns it directly. It doesn't *generate* a new response.

A RAG model takes it one step further: it uses the retrieved information as context for a Large Language Model (LLM) to craft a brand-new, conversational answer.

### How to Implement RAG with Groq

To make the switch, you just need to modify `app.py`.

1.  **Import `groq` and `dotenv`:**
    Add these imports at the top of `app.py`.

    ```python
    import os
    from groq import Groq
    from dotenv import load_dotenv

    load_dotenv() # Load environment variables from .env
    ```

2.  **Initialize the Groq Client:**
    Add this after the `app = Flask(__name__)` line.

    ```python
    # --- Initialize Groq Client ---
    try:
        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print("Groq client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        groq_client = None
    ```

3.  **Update the `/chat` Route:**
    Replace the `else` block in the `/chat` function. Instead of returning a hardcoded fallback message, you'll call the Groq API.

    **Before (Retrieval only):**
    ```python
    if highest_similarity_score > SIMILARITY_THRESHOLD:
        # Retrieve the best answer directly from the CSV
        response_message = df['Answers'].iloc[most_similar_index]
        matched_question = df['Questions'].iloc[most_similar_index]
        source = f"Matched: '{matched_question}' (Score: {highest_similarity_score:.2%})"
    else:
        # If no good match is found, provide a default response
        response_message = "I'm not sure I have information on that. Could you please try asking in a different way?"
        source = "Default Fallback"

    return jsonify({
        'response': response_message,
        'source': source
    })
    ```

    **After (RAG implementation):**
    ```python
    if highest_similarity_score > SIMILARITY_THRESHOLD:
        # --- This is the RAG part ---
        retrieved_context = df['Answers'].iloc[most_similar_index]
        matched_question = df['Questions'].iloc[most_similar_index]
        source = f"Context: '{matched_question}' (Score: {highest_similarity_score:.2%})"

        if not groq_client:
             return jsonify({
                'response': 'Groq client not initialized. Cannot generate response.',
                'source': 'Error'
            })

        # Use the retrieved answer as context for the LLM
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful mental health assistant. Your role is to provide supportive and informative answers based on the context provided. Do not invent information. Be empathetic and clear."
                },
                {
                    "role": "user",
                    "content": f"Based on the following information: '{retrieved_context}', please answer this question: '{user_message}'"
                }
            ],
            model="llama3-8b-8192", # Or another model like "mixtral-8x7b-32768"
        )
        response_message = chat_completion.choices[0].message.content

    else:
        # If no good match is found, provide a default response
        response_message = "I'm not sure I have information on that. Could you please try asking in a different way?"
        source = "Default Fallback"

    return jsonify({
        'response': response_message,
        'source': source
    })
    ```

With these changes, your chatbot will now use the CSV to find relevant context and then use a powerful LLM to generate a helpful, natural-sounding response.

***

##  Dependencies

* Flask
* pandas
* scikit-learn
* groq
* python-dotenv

