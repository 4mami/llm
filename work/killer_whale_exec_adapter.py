# 参考にした記事:
#  https://zenn.dev/yumefuku/articles/llm-finetuning-qlora

import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"

parser = argparse.ArgumentParser()
parser.add_argument('--adapter_path', default='./trained_models/nyan_Adapter_full_sample')
args = parser.parse_args()

model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.adapter_path, 
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

question_list = [
    "名前を教えてください",
    "日本の首都はどこですか", 
    "ジョークを言ってください", 
    "東北の観光地について教えてください" 
]

for i, question in enumerate(question_list, 1):
    print(f"\nchat_{i}----------------------------------------------------")
    print(f"質問: {question}")

    messages = [
        {"role": "user", "content": question}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=300
    )
    # 生成された回答部分のみ抽出
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"回答: {response}")
    print("----------------------------------------------------------")
