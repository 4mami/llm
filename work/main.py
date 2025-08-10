import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class StreamRequest(BaseModel):
    messages: list

async def send_token(messages: list):
    for token in messages[-1]["content"]:
        yield token
        await asyncio.sleep(0.1)

@app.post("/streaming")
async def ask_stream(request: StreamRequest) -> StreamingResponse:
    # print(request)
    return StreamingResponse(
        send_token(request.messages),
        media_type="text/event-stream",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
