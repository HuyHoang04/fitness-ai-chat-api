from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

model_id = "Soorya03/Llama-3.2-1B-Instruct-FitnessAssistant"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)


class ChatRequest(BaseModel):
    message: str
    history: list[str] = []


@app.post("/chat")
def chat(req: ChatRequest):
    history = req.history
    message = req.message

    full_prompt = ""
    for msg in history:
        full_prompt += msg + "\n"
    full_prompt += f"User: {message}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_new_tokens=150,
                                 temperature=0.7,
                                 top_k=50,
                                 top_p=0.9,
                                 repetition_penalty=1.1,
                                 pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(full_prompt):].strip()
    return {"response": answer}
