import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

device = 'cuda'

transformers.utils.move_cache()

if os.getcwd() == '/workspace/hshm':
    print('Setting TRANSFORMERS_CACHE to /workspace/hshm')
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/hshm'

transformers.utils.move_cache()

if os.getcwd() == '/workspace/hshm':
    print('Setting TRANSFORMERS_CACHE to /workspace/hshm')
    os.environ['TRANSFORMERS_CACHE'] = '/workspace/hshm'

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B").to(device)
input_text = "Miles Davis plays the"

print('tokeninzing')
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Forward pass
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

# Get last layer, last time slot attention
last_layer_attention = outputs.attentions[-1]  # Last layer
last_time_slot_attention = last_layer_attention[0, :, -1, :]  # Last time slot

# TODO: find correlation for specific "Miles Davis plays the" and compare against paper