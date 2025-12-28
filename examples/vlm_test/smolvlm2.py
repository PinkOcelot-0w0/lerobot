from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
 
model_path = "models/extracted_vlm"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2"
).to("cuda")
 
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "描述一下这张图片"},
            {"type": "image", "path": "vlm_test/images/1.jpg"}
        ]
    },
]
 
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
 
generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
 
