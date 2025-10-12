from contextlib import asynccontextmanager
import torch
import uvicorn
from fastapi import FastAPI
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"
ADAPTER_PATH = "./trained_models/nyan_Adapter_full_sample"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=ADAPTER_PATH,
        device_map={"": "cuda"},
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
    )
    tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

    # パディングトークンが設定されていない場合、EOSトークンを設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # パディングを右側に設定(fp16を使う際のオーバーフロー対策)
    tokenizer.padding_side = "right"

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    models['model'] = model
    models['tokenizer'] = tokenizer
    models['conversation_history'] = []

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/answer")
async def answer(q: str):
    models['conversation_history'].append({"role": "user", "content": q})
    prompt = models['tokenizer'].apply_chat_template(
        conversation=models['conversation_history'],
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = models['tokenizer']([prompt], return_tensors="pt").to("cuda")

    generated_ids = models['model'].generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=300
    )
        # 生成された回答部分のみ抽出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = models['tokenizer'].batch_decode(generated_ids, skip_special_tokens=True)[0]
    models['conversation_history'].append({"role": "assistant", "content": response})

    return {"question": q, "answer": response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
