from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    prompt: str
    image_url: str | None = None
    stream: bool = False

@app.post("/generate")
async def generate(req: Request):
    await asyncio.sleep(0.5)
    
    prompt = req.prompt.lower()
    
    if "python" in prompt and "server" in prompt:
        response = """Here is a simple and fast Python web server using `FastAPI`!

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, TensorGrid Cluster!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Instructions to run:
1. Install FastAPI and Uvicorn: `pip install fastapi uvicorn`
2. Save the code above in a file named `server.py`
3. Run the server: `python server.py`
"""
        return {"text": response}
        
    return {"text": f"I processed your query: `{req.prompt}`. Because this is running in Mock Mode (no GPUs detected), I am unable to generate a full LLM response. Try asking me to write a python web server!"}

@app.post("/o1_generate")
async def o1_generate(req: Request):
    await asyncio.sleep(2)
    return {"text": f"**[o1-Reasoning Mock]**\nI thought deeply about `{req.prompt}` for 2 seconds.\n\n### Self-Verification\n- Checked mathematical bounds.\n- Verified logic.\n\n**Final Answer:** You are running the TensorGrid Cluster."}

@app.post("/rag_generate")
async def rag_generate(req: Request):
    await asyncio.sleep(1)
    return {"text": f"**[GraphRAG Mock]**\n- Neo4j Node Found: `User`\n- Qdrant Vector Match: `99%`\n\nBased on the Knowledge Graph, the answer to `{req.prompt}` is verified."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
