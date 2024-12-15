from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import wikipediaapi
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Load the Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')


class QueryRequest(BaseModel):
    url: str
    question: str
    top_k: int = 5


def extract_title_from_url(url: str) -> str:
    """Extracts the title from a Wikipedia URL."""
    if "wikipedia.org/wiki/" in url:
        return url.split("wikipedia.org/wiki/")[-1]
    raise ValueError("Invalid Wikipedia URL")


def get_wikipedia_content(title: str) -> str:
    """Fetches the content of a Wikipedia page."""
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(title)
    if not page.exists():
        raise ValueError("Page does not exist.")
    return page.text


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Chunks the text into smaller pieces."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_relevant_chunks(question: str, chunks: list, top_k: int) -> list:
    """Returns the top K relevant chunks based on the question."""
    question_embedding = model.encode(question)
    chunk_embeddings = model.encode(chunks)

    # Calculate cosine similarities
    similarities = np.dot(chunk_embeddings, question_embedding)
    top_indices = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_indices]


@app.post("/query/")
async def query_wikipedia(request: QueryRequest):
    try:
        title = extract_title_from_url(request.url)
        content = get_wikipedia_content(title)
        chunks = chunk_text(content)
        relevant_chunks = get_relevant_chunks(request.question, chunks, request.top_k)
        return {"relevant_documents": relevant_chunks}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run the app, use the command: uvicorn main:app --reload
