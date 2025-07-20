# 参考にした記事:
#  https://zenn.dev/yumefuku/articles/llm-finetuning-qlora

import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='./trained_models/nyan_Merged_model')
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.model_path,
    local_files_only=True,
    device_map={"": 'cuda'},
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

conversation_history = []
while True:
    user_input = input("質問: ")
    if user_input.lower() == 'exit':
        break

    start_time = time.time()
    conversation_history.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(
        conversation=conversation_history,
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

    print("回答:", response)
    print(f"回答時間: {time.time() - start_time} 秒")
    conversation_history.append({"role": "assistant", "content": response})
