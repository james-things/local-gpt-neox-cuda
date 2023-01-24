# example script to run (a locally downloaded copy of) gpt-neox on a cuda-based gpu such as the 1080ti
# used in example: https://huggingface.co/EleutherAI/gpt-neo-2.7B
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch    

model = AutoModelForCausalLM.from_pretrained("./gpt-neo-2.7B", revision="float32", torch_dtype=torch.float32, device_map="auto", low_cpu_mem_usage=True) # low_cpu_mem_usage=False
tokenizer = AutoTokenizer.from_pretrained("./gpt-neo-2.7B")

# prompt
context = """Text-based generative AI is"""

input_ids = tokenizer(context, return_tensors="pt").input_ids
input_ids = input_ids.to('cuda')

gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100)
gen_text = tokenizer.batch_decode(gen_tokens)[0]

# output
print(f"\nPrompt: {context}\n")
print(f"Output: {gen_text}")