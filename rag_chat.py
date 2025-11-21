import os
from openai import OpenAI
from retriever import retrieve_similar_chunks

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def determine_retrieval_query(user_query, history):
    """
    Use LLM to dynamically determine the best query for retrieval based on conversation context.
    The LLM analyzes the conversation and determines what should be searched in the vector database.
    """
    # Convert history to chat format (skip system messages)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") != "system"]
    
    # Create a prompt for the LLM to determine the retrieval query
    query_determination_prompt = {
        "role": "system",
        "content": """You are a query analyzer for a patient support information retrieval system. 
Your task is to analyze the conversation and determine the BEST search query to use for retrieving relevant information from the knowledge base.

CRITICAL RULES:
1. If the user is asking for MORE INFORMATION about something previously discussed (e.g., "more info", "tell me more", "more details", "what else"):
   - Look at the conversation history to identify what was discussed in previous messages
   - Extract the topic (disease, medication, symptom, etc.) that was mentioned earlier
   - Use that extracted term as the search query
   - Example: If user previously asked about "diabetes" and now says "more info", return "diabetes"
   - Example: If user previously asked about "medication reminders" and now says "tell me more", return "medication reminders"

2. If the user is asking about a disease or condition:
   - Extract the disease/condition name (diabetes, hypertension, asthma, depression, etc.)
   - Use the disease name for retrieval
   - Example: "tell me about diabetes" → Return: "diabetes"
   - Example: "what is high blood pressure" → Return: "hypertension"

3. If the user is asking about medications:
   - Extract medication names, brand names, or active ingredients
   - Use terms like "medication", "medicine", or specific drug names
   - Example: "how to take metformin" → Return: "metformin medication"
   - Example: "side effects of my medicine" → Return: "medication side effects"

4. If the user is asking about adherence or taking medications:
   - Use terms like "medication adherence", "medication reminders", "taking medication", "medication tracking"
   - Example: "I keep forgetting my pills" → Return: "medication reminders adherence"
   - Example: "how to remember to take medicine" → Return: "medication adherence reminders"

5. If the user is asking about symptoms:
   - Extract symptom names or condition names for symptom tracking
   - Use terms like "symptoms", "symptom tracking", or condition names
   - Example: "what symptoms should I track for diabetes" → Return: "diabetes symptom tracking"
   - Example: "when should I worry about my symptoms" → Return: "symptoms red flags"

6. If the user is asking about their journey or treatment stages:
   - Use terms like "patient journey", "treatment stages", "what to expect", or condition names
   - Example: "what happens after diagnosis" → Return: "patient journey diagnosis"
   - Example: "stages of diabetes treatment" → Return: "diabetes patient journey"

7. If the user is asking about support programs or resources:
   - Use terms like "support programs", "support groups", "resources", "help", or specific program types
   - Example: "are there support groups" → Return: "support programs groups"
   - Example: "where can I get help" → Return: "support programs resources"

8. If the user is referring to something mentioned earlier (using pronouns like "it", "that", "this"), extract the actual topic from the conversation history.

9. Always return ONLY the search query term(s) - no explanations, no questions, just the query string.

IMPORTANT: For follow-up queries like "more info", ALWAYS extract the topic from conversation history - never return the follow-up phrase itself.
IMPORTANT: Be specific - include disease names, medication names, or topic keywords.

Return ONLY the search query, nothing else."""
    }
    
    # Build messages for query determination
    messages = [query_determination_prompt] + chat_history + [
        {"role": "user", "content": f"Given the conversation above, what should be the search query for the current user message: '{user_query}'?\n\nReturn ONLY the search query:"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=50,
        )
        
        retrieval_query = response.choices[0].message.content.strip()
        
        # Clean up the response
        retrieval_query = retrieval_query.strip('"').strip("'").strip()
        
        if ":" in retrieval_query and len(retrieval_query.split(":")) > 1:
            retrieval_query = retrieval_query.split(":")[-1].strip()
        
        if not retrieval_query or len(retrieval_query) < 2:
            retrieval_query = user_query
        
        print(f"🤖 LLM determined retrieval query: '{retrieval_query}' (from user: '{user_query}')")
        return retrieval_query
        
    except Exception as e:
        print(f"⚠️  Error in LLM query determination: {e}. Using original query.")
        return user_query

def create_system_prompt(context_text):
    """Create the system prompt for the patient support assistant."""
    return {
        "role": "system",
        "content": (
            "You are a compassionate, empathetic, and supportive Patient Support Assistant. "
            "Your role is to help patients understand their conditions, manage their medications, track symptoms, "
            "navigate their health journey, and connect with support resources. "
            "Use the entire conversation history and the provided context to give accurate, helpful, and reassuring answers.\n\n"

            "CRITICAL: Response Style - Be CONCISE, CONFIDENT, and EMPATHETIC:\n"
            "- Keep responses SHORT (2-4 sentences maximum, unless user explicitly asks for more)\n"
            "- Be confident and direct - provide the essential information needed to answer the question\n"
            "- Use warm, understanding language but keep it brief\n"
            "- Acknowledge concerns briefly, then provide the answer\n"
            "- Break down complex information into simple terms, but keep it concise\n"
            "- Use 'you' and 'your' to personalize responses\n"
            "- Do NOT add unnecessary explanations or verbose elaborations\n"
            "- Do NOT dump raw data - provide concise, well-structured summaries\n"
            "- Do NOT list multiple items unless the question specifically asks for a list\n\n"

            "Response Guidelines (CONCISE):\n"
            "- For disease education: Give key facts in 2-3 sentences. Full details only if asked.\n"
            "- For medication questions: Explain how to take and key notes in 2-3 sentences. Full information only if asked.\n"
            "- For adherence questions: Provide 2-3 practical tips. Full strategies only if asked.\n"
            "- For symptom tracking: Explain what to track and red flags in 2-3 sentences. Full details only if asked.\n"
            "- For journey questions: Briefly explain the stage in 2-3 sentences. Full journey details only if asked.\n"
            "- For support programs: Mention available programs briefly. Full descriptions only if asked.\n\n"

            "ELABORATION RULES:\n"
            "- ONLY elaborate when user explicitly asks: 'more info', 'more details', 'tell me more', 'elaborate', 'explain more', 'give me more information', 'expand', 'detailed', 'full details'\n"
            "- When user asks to elaborate, THEN provide additional relevant information, examples, or detailed explanations\n"
            "- Default response should be SHORT and DIRECT - save detailed explanations for when explicitly requested\n\n"

            "Safety and Medical Disclaimers:\n"
            "- Always emphasize that you provide general information, not medical advice\n"
            "- Encourage users to consult their healthcare provider for personalized advice\n"
            "- For urgent symptoms or red flags, clearly state they should seek immediate medical attention\n"
            "- Never diagnose or recommend specific treatments - only provide educational information\n\n"

            "Conversation Flow:\n"
            "- Greet warmly and ask how you can help\n"
            "- Listen to concerns and provide relevant information\n"
            "- Check understanding and offer additional help\n"
            "- Be supportive throughout the conversation\n\n"

            f"Context from knowledge base:\n{context_text}\n\n"
            "Remember: Be CONCISE and DIRECT. Provide short, confident answers (2-4 sentences). "
            "Only elaborate when the user explicitly asks for more information. "
            "Be kind and supportive, but keep it brief. If information is not available in the context, say so briefly and suggest "
            "they speak with their healthcare provider."
        ),
    }

def generate_answer(user_query, history):
    """Generate context-aware answer using chat history and RAG."""
    # Check if user is asking for more information (increase top_k for more comprehensive retrieval)
    user_query_lower = user_query.lower().strip()
    is_more_data_request = any(phrase in user_query_lower for phrase in [
        "more info", "more information", "more details", "tell me more", 
        "what else", "anything else", "additional", "more about",
        "elaborate", "explain more", "give me more", "expand",
        "detailed", "full details", "complete information"
    ])
    
    # Use LLM to dynamically determine the best query for retrieval
    retrieval_query = determine_retrieval_query(user_query, history)
    
    # Retrieve chunks - use more chunks if user asks for "more info"
    top_k = 10 if is_more_data_request else 5
    retrieved_chunks = retrieve_similar_chunks(retrieval_query, top_k=top_k)
    context_text = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
    
    # If no context retrieved, still proceed but with empty context
    if not context_text.strip():
        context_text = "No relevant information found in the knowledge base for this query."
    
    # Convert Streamlit history to OpenAI chat format (exclude system messages from history)
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") != "system"]
    system_prompt = create_system_prompt(context_text)
    
    messages = [system_prompt] + chat_history
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,  # Slightly higher for more natural, empathetic responses
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"⚠️  Error generating answer: {e}")
        return "I'm sorry, I'm having trouble responding right now. Please try again or contact your healthcare provider for immediate assistance."

