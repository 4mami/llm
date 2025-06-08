# 参考にした記事:
#  https://zenn.dev/yumefuku/articles/llm-finetuning-qlora

import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"

parser = argparse.ArgumentParser()
parser.add_argument('--adapter_path', default='./nyan_Adapter_full_sample')
args = parser.parse_args()

model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.adapter_path, 
    device_map={"": "cuda"},
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID, 
)
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}" # todo 訓練時と同じテンプレートを設定

# パディングトークンが設定されていない場合、EOSトークンを設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# パディングを右側に設定(fp16を使う際のオーバーフロー対策)
tokenizer.padding_side = "right"

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
