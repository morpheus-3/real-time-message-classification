import json
from functools import lru_cache
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from . import messages
from .common import ROOT, metadata, groq_explain
from pathlib import Path

app = FastAPI(title='Message Intelligence', version="1.0.0")

@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

@app.get("/status")
def status():
    report = ROOT / "messages" / "report.json"
    return metadata(report) if report.exists() else {"trained": False}

@app.exception_handler(FileNotFoundError)
async def missing_model(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={"detail": "Model not trained. Run the documented training command first."})

@app.exception_handler(ValueError)
async def invalid_input(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": str(exc)})

class MessageInput(BaseModel):
    text: str = Field(min_length=1, max_length=4000)

class ExplanationInput(BaseModel):
    context: dict

@lru_cache(maxsize=1)
def classifier():
    return messages.Classifier()

@app.post("/messages/classify")
def classify(body: MessageInput):
    return classifier().predict(body.text)

@app.post("/messages/stream")
def stream(body: MessageInput):
    result = classifier().predict(body.text)
    return StreamingResponse(iter(["event: classification\ndata: " + json.dumps(result) + "\n\n"]), media_type="text/event-stream")

@app.post("/explain")
def explain(body: ExplanationInput):
    if len(json.dumps(body.context)) > 12000:
        raise HTTPException(413, "Context too large.")
    try:
        return {"provider": "Groq", "explanation": groq_explain(body.context)}
    except ValueError as exc:
        raise HTTPException(503, str(exc))
    except Exception:
        raise HTTPException(502, "Groq request failed; check server configuration and quota.")
