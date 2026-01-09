from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
model_path = "models/extracted_vlm"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    _attn_implementation="eager",
   device_map="cuda").to(DEVICE)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/pinkocelot/code/lerobot/examples/vlm_test/images/inbox.png"},
            {"type": "text", "text": "纸巾在盒子里吗"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])