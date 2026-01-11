import json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def main():
    # 硬编码配置
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    device = "auto"
    max_new_tokens = 256
    
    # 硬编码测试数据（只需要图片和问题）
    test_cases = [
        {
            "image": "/home/pinkocelot/code/lerobot/examples/vlm_test/images/1.jpg",
            "question": "You are a robot arm. How should you grasp the green object and place it into the box? Describe each step."
        }
    ]

    # 加载模型和处理器
    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    vlm = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    vlm.eval()

    print(f"Total samples: {len(test_cases)}\n")

    for idx, case in enumerate(test_cases):
        image_path = case["image"]
        question = case["question"]

        # 读取图像
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Failed to load image {image_path}: {e}\n")
            continue

        # SmolVLM 对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # 准备输入
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(vlm.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}

        # 生成文本
        with torch.no_grad():
            output_ids = vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 解码并提取 Assistant 回答
        full_response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # 提取 "Assistant:" 之后的内容
        if "Assistant:" in full_response:
            pred_answer = full_response.split("Assistant:")[-1].strip()
        else:
            pred_answer = full_response

        # 终端输出
        print(f"[{idx}] Q: {question}")
        print(f"     A: {pred_answer}")
        print()


if __name__ == "__main__":
    main()