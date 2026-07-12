
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag_processor import RagProcessor, RagProcessorError

app = FastAPI()


class SearchQuery(BaseModel):
    query: str
    k: int = Field(default=5, ge=1)


@app.post("/search")
async def search(args: SearchQuery) -> dict[str, Any] | HTTPException:
    # Load RagProcessor
    rag = RagProcessor()

    # Start searching using RagProcessor
    try:
        results = rag.search(query=args.query, k=args.k)
    except RagProcessorError as e:
        return HTTPException(400, e)

    # Print results
    return results.model_dump()


@app.post("/answer")
async def answer(args: SearchQuery) -> dict[str, Any] | HTTPException:
    # Load RagProcessor
    rag = RagProcessor()

    try:
        answer = rag.answer(query=args.query, k=args.k)
    except RagProcessorError as e:
        return HTTPException(400, e)

    # Print results
    return {'response': answer}
