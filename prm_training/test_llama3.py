import transformers
import torch

model_id = "/workspace/dujh22/models/llama-3-8B/"

pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={
        "torch_dtype": torch.bfloat16
    }, 
    device_map="auto"
) 

output = pipeline("Hey how are you doing today?")
print(output)