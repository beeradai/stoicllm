from fastapi import FastAPI
from pydantic import BaseModel
from inference import generate_response

app = FastAPI()

class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 150

@app.post("/generate")
def generate(req: Request):
    result = generate_response(req.prompt, req.max_new_tokens)
    return {"response": result}
