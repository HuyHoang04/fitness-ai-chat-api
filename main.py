from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import torch
import threading
import time

app = FastAPI()

# Global variables to store model and tokenizer
tokenizer = None
model = None
is_model_loaded = False
is_loading = False

# Function to load model in background
def load_model_in_background():
    global tokenizer, model, is_model_loaded, is_loading
    is_loading = True
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = "Soorya03/Llama-3.2-1B-Instruct-FitnessAssistant"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        is_model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
    finally:
        is_loading = False

# Start loading the model in a background thread when the app starts
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()

class ChatRequest(BaseModel):
    message: str
    history: list[str] = []

@app.get("/")
def read_root():
    if is_model_loaded:
        return {"status": "ready", "message": "Model loaded and ready to use"}
    elif is_loading:
        return {"status": "loading", "message": "Model is still loading, please try again later"}
    else:
        # Start loading if not already loading
        if not is_loading:
            thread = threading.Thread(target=load_model_in_background)
            thread.daemon = True
            thread.start()
        return {"status": "initializing", "message": "Starting model loading process"}

@app.post("/chat")
def chat(req: ChatRequest):
    global is_model_loaded
    
    # Check if model is loaded
    if not is_model_loaded:
        if is_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again later")
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded, please visit the root endpoint first")
    
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


# Troubleshooting 504 Gateway Timeout Error

It looks like you're experiencing a 504 Gateway Timeout error when trying to access your FastAPI application deployed on Azure. This error typically occurs when the server acting as a gateway or proxy did not receive a timely response from the upstream server.

## Likely Causes for Your Application

Based on your FastAPI application that loads the Llama 3.2 model, there are several potential causes:

1. **Model Loading Time**: The most likely cause is that loading the large language model (`Soorya03/Llama-3.2-1B-Instruct-FitnessAssistant`) is taking too long, causing Azure to time out before your application fully starts.

2. **Resource Limitations**: Your Azure App Service plan might not have enough resources (memory/CPU) to handle the large model.

3. **Startup Timeout**: Azure App Service has a default startup timeout (usually around 230 seconds), which your application might be exceeding.

## Recommended Solutions

Here are some solutions to address the 504 Gateway Timeout:

### 1. Modify Your Application to Load the Model Asynchronously

Update your `main.py` to load the model after the application starts:
```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import torch
import threading
import time

app = FastAPI()

# Global variables to store model and tokenizer
tokenizer = None
model = None
is_model_loaded = False
is_loading = False

# Function to load model in background
def load_model_in_background():
    global tokenizer, model, is_model_loaded, is_loading
    is_loading = True
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_id = "Soorya03/Llama-3.2-1B-Instruct-FitnessAssistant"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        is_model_loaded = True
    except Exception as e:
        print(f"Error loading model: {e}")
    finally:
        is_loading = False

# Start loading the model in a background thread when the app starts
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=load_model_in_background)
    thread.daemon = True
    thread.start()

class ChatRequest(BaseModel):
    message: str
    history: list[str] = []

@app.get("/")
def read_root():
    if is_model_loaded:
        return {"status": "ready", "message": "Model loaded and ready to use"}
    elif is_loading:
        return {"status": "loading", "message": "Model is still loading, please try again later"}
    else:
        # Start loading if not already loading
        if not is_loading:
            thread = threading.Thread(target=load_model_in_background)
            thread.daemon = True
            thread.start()
        return {"status": "initializing", "message": "Starting model loading process"}

@app.post("/chat")
def chat(req: ChatRequest):
    global is_model_loaded
    
    # Check if model is loaded
    if not is_model_loaded:
        if is_loading:
            raise HTTPException(status_code=503, detail="Model is still loading, please try again later")
        else:
            raise HTTPException(status_code=503, detail="Model is not loaded, please visit the root endpoint first")
    
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
```
