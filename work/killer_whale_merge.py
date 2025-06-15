# 参考にした記事:
#  https://zenn.dev/yumefuku/articles/llm-finetuning-qlora

import argparse
import torch
from peft import AutoPeftModelForCausalLM

MODEL_ID = "DataPilot/ArrowPro-7B-KillerWhale"

parser = argparse.ArgumentParser()
parser.add_argument('--adapter_path', default='./trained_models/nyan_Adapter_full_sample')
parser.add_argument("--output_dir", default="./trained_models/nyan_Merged_model")
args = parser.parse_args()

model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=args.adapter_path, 
    device_map={"": "cuda"},
    torch_dtype=torch.float16,
)

model = model.merge_and_unload()
model.save_pretrained(
    args.output_dir, 
    safe_serialization=True
)

print("マージ完了")
