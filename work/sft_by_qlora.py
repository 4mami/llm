import torch
from datasets import load_dataset, DatasetDict
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"
DATASET_ID = "DataPilot/databricks-dolly-15k-Nyan-ja"
PROMPT_FORMAT = """<start_of_turn>user
{system}

{instruction}
<end_of_turn>
<start_of_turn>model
{output}
<end_of_turn>
"""

def generate_text_field(data):
    system = "あなたは優秀なアシスタントです。指示に対して適切な回答を行なってください。"

    instruction = data["instruction"]
    output = data["output"]

    full_prompt = PROMPT_FORMAT.format(system=system, instruction=instruction, output=output) 
    return {"text": full_prompt}

dataset = load_dataset(DATASET_ID)
dataset = DatasetDict({
    split: dataset[split].select(range(1500)) 
    for split in dataset.keys()
})
train_dataset = dataset.map(generate_text_field)
# 学習データの構造次第では意図した学習が行われない可能性があるため、不要なフィールドを削除
train_dataset = train_dataset.remove_columns(['output', 'index', 'input', 'category', 'instruction'])

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # 4ビット量子化を使用
    bnb_4bit_quant_type="nf4", # 4ビット量子化の種類にnf4（NormalFloat4）を使用
    bnb_4bit_use_double_quant=True, # 二重量子化を使用
    bnb_4bit_compute_dtype=torch.float16 # 量子化のデータ型をfloat16に設定
)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    device_map="auto",
    quantization_config=quantization_config,
    # attn_implementation="eager", # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
)

# キャッシュを無効化（メモリ使用量を削減）
model.config.use_cache = False 
# テンソル並列ランクを１に設定（テンソル並列化を使用しない）
model.config.pretraining_tp = 1 

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_ID,
    # attn_implementation="eager", # 注意機構に"eager"を設定（Gemma2モデルの学習で推奨されているため）
    add_eos_token=True,
)

# パディングトークンが設定されていない場合、EOSトークンを設定
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# パディングを右側に設定(fp16を使う際のオーバーフロー対策)
tokenizer.padding_side = "right"

def find_all_linear_names(model):
    target_class = bnb.nn.Linear4bit
    linear_layer_names = set()
    for name_list, module in model.named_modules():
        if isinstance(module, target_class):
            names = name_list.split('.')
            layer_name = names[-1] if len(names) > 1 else names[0]
            linear_layer_names.add(layer_name)
    if 'lm_head' in linear_layer_names:
        linear_layer_names.remove('lm_head')
    return list(linear_layer_names)

target_modules = find_all_linear_names(model)
Lora_config = LoraConfig(
    lora_alpha=8, # LoRAによる学習の影響力を調整（スケーリング)
    lora_dropout=0.1,
    r=4,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

training_arguments = SFTConfig(
    output_dir="./train_logs",
    fp16=True,
    logging_strategy='epoch', # 各エポックごとにログを保存（デフォルトは"steps"）
    save_strategy='epoch', # 各エポックごとにチェックポイントを保存（デフォルトは"steps"）
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3, # 勾配の最大ノルムを制限（クリッピング）
    warmup_ratio=0.03, # 学習を増加させるウォームアップ期間の比率
    weight_decay=0.001, # 重み減衰率
    group_by_length=True,# シーケンスの長さが近いものをまとめてバッチ化
    report_to="tensorboard",
    dataset_text_field="text",
    max_seq_length=512,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset["train"],
    peft_config=Lora_config,
    args=training_arguments,
)

# 正規化層をfloat32に変換(学習を安定させるため)
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
trainer.model.save_pretrained("./nyan_Adapter")
