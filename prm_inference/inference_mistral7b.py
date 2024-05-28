# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")

# model_path = "/workspace/dujh22/models/mistral-7B-sft"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

# def get_mistral_response(question):
#     inputs = tokenizer(question, return_tensors="pt").to(device)
#     outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# if __name__ == "__main__":
#     question = "What is the capital of France?"
#     response = get_mistral_response(question)
#     print(response)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = "/workspace/dujh22/models/mistral-7B-sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model once into CPU memory then move it to all available GPUs using DataParallel
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)  # This line wraps the model to use multiple GPUs

def get_mistral_response(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    # Ensure that model.module is used to access the underlying model for generation
    if isinstance(model, torch.nn.DataParallel):
        outputs = model.module.generate(**inputs, max_new_tokens=2048, do_sample=True)
    else:
        outputs = model.generate(**inputs, max_new_tokens=2048, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    question = "What is the capital of France?"
    response = get_mistral_response(question)
    print(response)