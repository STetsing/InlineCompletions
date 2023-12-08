import json
import time
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else 'cpu'

model_path = 'bigcode/starcoder'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = T5ForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True).to(device)


def infer(data):
    data = json.loads(data)
    context = data['context']
    mnt = data['max_new_words']
    temp = data['temperature']
    start = time.time()
    input_ids = tokenizer(context, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_new_tokens=int(mnt), temperature=float(temp), pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return json.dumps(generated_text)