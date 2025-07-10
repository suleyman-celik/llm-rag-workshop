
import os
import logging
import requests

from elasticsearch import Elasticsearch
from huggingface_hub import InferenceClient
from openai import OpenAI

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Global clients for OpenAI and Hugging Face
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
hf_client = InferenceClient(token=HUGGINGFACE_API_TOKEN) if HUGGINGFACE_API_TOKEN else None

# Initialize the Elasticsearch client
# Make sure to set the ELASTICSEARCH_URL environment variable or use the default    
# "http://localhost:9200"
es = Elasticsearch(hosts=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))

# Define the context template for formatting the retrieved documents
# This template will be used to format each document retrieved from the FAQ database    
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

# This template will be used to format the prompt for the LLM
# It includes the user's question and the context retrieved from the FAQ database
prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


# Document retrieval from Elasticsearch
def retrieve_documents(
        query,
        index_name=None,
        max_results=5,
        course="data-engineering-zoomcamp"
    ):
    """Retrieves matching documents from Elasticsearch based on the given query and course."""
    index_name = index_name or "course-questions"
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }
    try:
        response = es.search(index=index_name, body=search_query)
        # response['hits']['hits']
        hits = response.get('hits', {}).get('hits', [])
        return [hit['_source'] for hit in hits]
    except Exception as e:
        logging.error(f"Elasticsearch search error: {e}")
        return []


# Build context string from documents
def build_context(documents):
    """Builds a context string by formatting documents using the context_template."""
    context_result = ""
    for doc in documents:
        try:
            doc_str = context_template.format(
                section=doc.get("section", "N/A"),
                question=doc.get("question", "N/A"),
                text=doc.get("text", "N/A")
            )
            context_result += ("\n\n" + doc_str)
        except Exception as e:
            logging.warning(f"Skipping document due to formatting error: {e}")
    return context_result.strip()


# Build full prompt
def build_prompt(user_question, documents):
    """Formats the user question and context documents into a full prompt."""
    if not documents:
        return f"""You're a course assistant, but no FAQ documents were found to help with this question: {user_question}"""
    
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context,
    )
    return prompt


# Ask OpenAI
def ask_openai(prompt, model="gpt-4o"):
    """Query OpenAI's chat completion endpoint."""
    if not openai_client:
        return "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI request failed: {e}")
        return "Sorry, I couldn't generate an answer due to an OpenAI API error."


# Ask Ollama
# def ask_ollama(prompt, model="llama2", base_url=None):
#     """Query the Ollama API to generate a response using a specified model."""
#     base_url = base_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
#     # You can also switch to /api/generate if you're just doing single prompt/response generation.
#     endpoint = f"{base_url}/api/chat"

#     try:
#         response = requests.post(endpoint, json={
#             "model": model,
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ]
#         })
#         response.raise_for_status()
#         data = response.json()
#         return data.get("message", {}).get("content", "No response content.")
#     except Exception as e:
#         logging.error(f"Ollama request failed: {e}")
#         return "Sorry, I couldn't generate an answer due to an Ollama API error."


# Ask Hugging Face
# def ask_huggingface(prompt, model="HuggingFaceH4/zephyr-7b-beta"):
def ask_huggingface(prompt, model="HuggingFaceH4/zephyr-7b-beta"):
    """Query Hugging Face's text generation endpoint."""
    if not hf_client:
        return "Hugging Face token not found. Please set the HUGGINGFACE_API_TOKEN environment variable."
    try:
        response = hf_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Hugging Face request failed: {e}")
        return "Sorry, I couldn't generate an answer due to a Hugging Face API error."

# Ask Hugging Face for summarization
# def ask_huggingface_summarization(text, model="facebook/bart-large-cnn"):
#     """Query Hugging Face's text summarization endpoint."""
#     if not hf_client:
#         return "Hugging Face token not found. Please set the HUGGINGFACE_API_TOKEN environment variable."
#     try:
#         # The API might have a separate summarization endpoint or you can use text-generation
#         response = hf_client.summarization(text, model=model)
#         return response[0]['summary_text']  # Adjust depending on API response structure
#     except Exception as e:
#         logging.error(f"Hugging Face summarization failed: {e}")
#         return "Sorry, summarization failed due to API error."


# QA bot main function
def qa_bot(user_question, course, provider="huggingface", model=None, return_docs=False):
    """Main QA function: retrieves documents and queries the selected LLM."""
    context_docs = retrieve_documents(user_question, course=course)

    if not context_docs:
        msg = "Sorry, I couldn't find any relevant documents to answer your question."
        return (msg, []) if return_docs else msg

    prompt = build_prompt(user_question, context_docs)

    if provider == "openai":
        answer = ask_openai(prompt, model=model or "gpt-4o")
    elif provider == "huggingface":
        answer = ask_huggingface(prompt, model=model or "HuggingFaceH4/zephyr-7b-beta")
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'openai' or 'huggingface'.")

    return (answer, context_docs) if return_docs else answer

# Document summarization function
def summarize_document(text, provider="huggingface", model=None):
    """
    Summarizes a given text document using the selected provider.

    summary = summarize_document("Your long text here", provider="openai")
    print("Summary:", summary)    
    """
    prompt = f"Summarize the following document briefly:\n\n{text}"
    
    if provider == "openai":
        return ask_openai(prompt, model or "gpt-3.5-turbo")
    elif provider == "huggingface":
        return ask_huggingface_summarization(prompt, model or "HuggingFaceH4/zephyr-7b-beta")
    else:
        raise ValueError(f"Unknown provider: {provider}")


# CLI usage
# Example:
# python qa_bot.py --question "What is Parquet?" --course data-engineering-zoomcamp --provider openai

# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Course FAQ QA Bot")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--course", default="data-engineering-zoomcamp", help="Course name to filter Elasticsearch.")
    parser.add_argument("--provider", default="huggingface", choices=["openai", "huggingface"],
                        help="Which LLM provider to use.")
    parser.add_argument("--model", default=None, help="(Optional) Specific model name to use.")
    args = parser.parse_args()

    answer, docs = qa_bot(
        user_question=args.question,
        course=args.course,
        provider=args.provider,
        model=args.model,
        return_docs=True
    )

    print("\nAnswer:\n", answer)
    print("\n--- Retrieved Context Documents ---")
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Section: {doc.get('section', 'N/A')}\nQ: {doc.get('question', 'N/A')}\nA: {doc.get('text', 'N/A')}\n")
