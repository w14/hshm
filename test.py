import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

device = 'cuda'

cache_dir = '/workspace/hshm'

print('load model')
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir = cache_dir).to(device)
print('load tokenizer')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir = cache_dir).to(device)
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