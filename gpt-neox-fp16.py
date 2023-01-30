# example script to run (a locally downloaded copy of) gpt-neox on a cuda-based gpu such as the 1080ti in 16-bit floating point precision
# by restricting to fp16, much larger lengths of text can be generated without running into memory limit-related issues
# used in example: https://huggingface.co/EleutherAI/gpt-neo-2.7B
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse    

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", help="input prompt as string", default="Text-based generative AI has so much potential because")
parser.add_argument("-l", "--length_max", help="maximum length as integer", default=100, type=int)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("./gpt-neo-2.7B", torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True) # low_cpu_mem_usage=False
tokenizer = AutoTokenizer.from_pretrained("./gpt-neo-2.7B")

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

input_ids = tokenizer(args.prompt, return_tensors="pt", padding=True).input_ids
input_ids = input_ids.to('cuda')

gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=args.length_max)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

# output
print(f"\nPrompt: {args.prompt}\n")
print(f"Output: {gen_text}")