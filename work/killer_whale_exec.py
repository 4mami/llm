import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import time

print("Start: ", datetime.datetime.now())
tokenizer = AutoTokenizer.from_pretrained("DataPilot/ArrowPro-7B-KillerWhale")
model = AutoModelForCausalLM.from_pretrained(
  "DataPilot/ArrowPro-7B-KillerWhale",
  torch_dtype="auto",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.pad_token_id

model.eval()
print("Model loaded ", datetime.datetime.now())

if torch.cuda.is_available():
    model = model.to("cuda")

def build_prompt(user_query):
    sys_msg = "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。"
    template = """[INST] <<SYS>>
{}
<</SYS>>

{}[/INST]"""
    return template.format(sys_msg,user_query)

# Infer with prompt without any additional input
count = 0
while True:
    user_query = input("質問を入力してください (終了するには 'quit' と入力): ")
    if user_query.lower() == 'quit':
        break

    start_time = time.time()
    user_inputs = {
        "user_query": user_query,
    }
    prompt = build_prompt(**user_inputs)

    inputs = tokenizer(
        prompt, 
        add_special_tokens=True, 
        return_tensors="pt"
    ).to(device=model.device)
    input_ids = inputs["input_ids"]

    tokens = model.generate(
        input_ids,
        attention_mask=inputs["attention_mask"],
        max_new_tokens=500,
        temperature=1,
        top_p=0.95,
        do_sample=True,
    )

    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"{count+1} Question: {user_query}")
    print(f"{count+1} Response: {out}")
    print(f"{count+1} Response time: {time.time() - start_time} sec")
    count += 1
