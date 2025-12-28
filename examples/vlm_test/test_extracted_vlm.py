#!/usr/bin/env python

"""
Quick test for an extracted SmolVLM model:
1) Text generation from image + prompt
2) Attention visualization (vision self-attn + text-conditioned cross-attn)

Usage:
  python examples/Bluekiwi/test_extracted_vlm.py \
    --vlm outputs/extracted_vlm \
    --image examples/Bluekiwi/1.png \
    --prompt "<image> Describe the green tissue." \
    --out outputs/attn

Offline:
  export HF_HUB_OFFLINE=1
"""

import argparse
import os
from pathlib import Path
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_vlm(
    vlm_dir: str,
    device: str = "cpu",
    dtype: str = "bf16",
    attn_impl: str = "sdpa",
    local_only: bool = True,
    hf_token: str | None = None,
):
    processor = AutoProcessor.from_pretrained(vlm_dir, local_files_only=local_only, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_dir,
        local_files_only=local_only,
        device_map=device,
        attn_implementation=attn_impl,
        token=hf_token,
    )
    # cast dtype to reduce memory if on CUDA
    if device == "cuda":
        if dtype == "bf16":
            model.to(dtype=torch.bfloat16)
        elif dtype == "fp16":
            model.to(dtype=torch.float16)
    # prefer enabling attentions at config level for libraries like BertViz
    try:
        model.config.output_attentions = True
    except Exception:
        pass
    model.eval()
    return processor, model


def ensure_eager_attention(model):
    """Try to enable eager attention to allow output_attentions=True to work."""
    # Global on the model wrapper
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass
    # On the inner multimodal model
    try:
        if hasattr(model, "model"):
            model.model.set_attn_implementation("eager")
    except Exception:
        pass
    # On vision submodules if available
    try:
        if hasattr(model, "model"):
            vm = None
            if hasattr(model.model, "vision_tower"):
                vm = model.model.vision_tower
            elif hasattr(model.model, "vision_model"):
                vm = model.model.vision_model
            if vm is not None:
                vm.set_attn_implementation("eager")
    except Exception:
        pass


def visualize_map(image, attn_map, title: str, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(attn_map, cmap="inferno", alpha=0.75)
    plt.axis("off")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[OK] Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", default="outputs/extracted_vlm", help="VLM path or HF hub id (e.g. HuggingFaceTB/SmolVLM2-1.7B)")
    parser.add_argument("--image", default="examples/Bluekiwi/1.png", help="Image path")
    parser.add_argument("--prompt", default="<image> Describe the green tissue.", help="Prompt text")
    parser.add_argument("--out", default="outputs/attn", help="Output folder for visualizations")
    parser.add_argument("--token", default="green", help="Target word for cross-attention")
    parser.add_argument("--attn_layer", type=int, default=-1, help="Layer index for attentions")
    parser.add_argument("--head", default="mean", help="Head fuse: 'mean' or a number")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to run on")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Precision (CUDA only)")
    parser.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa", help="Attention implementation")
    parser.add_argument("--online", action="store_true", help="Load model/processor from HF Hub (disable local-only)")
    parser.add_argument("--hf_token", default=None, help="HF token for private models (optional)")
    parser.add_argument("--bertviz_html", default=None, help="If set, export BertViz cross-attn HTML to this path")
    parser.add_argument("--cross_loc", choices=["auto", "start", "end", "placeholder"], default="auto", help="How to locate image tokens in cross-attn src")
    parser.add_argument("--skip_img_prefix", type=int, default=-1, help="Skip N prefix tokens in image segment (-1 = auto)")
    parser.add_argument("--skip_img_suffix", type=int, default=0, help="Skip N suffix tokens in image segment")
    parser.add_argument("--simple_html", default=None, help="Export a simple interactive HTML for cross-attn (no external deps)")
    parser.add_argument("--html_max_layers", type=int, default=12, help="Limit number of layers exported in HTML (from last)")
    parser.add_argument("--html_heads", choices=["mean", "all"], default="mean", help="Export mean head or all heads in HTML")
    # Generation controls
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling instead of greedy")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    processor, model = load_vlm(
        args.vlm,
        device=args.device,
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        local_only=(not args.online),
        hf_token=args.hf_token,
    )
    # If we plan to capture attentions, enforce eager impl
    if args.attn_impl != "eager":
        print("[INFO] Switching to eager attention implementation for capturing attentions")
        ensure_eager_attention(model)

    image = Image.open(args.image).convert("RGB")

    # =========================
    # 1) Text generation
    # =========================
    inputs_mm = processor(text=args.prompt, images=image, return_tensors="pt")
    inputs_mm = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs_mm.items()}
    # Ensure tensor dtype matches model dtype on CUDA to avoid mismatches
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    for k in list(inputs_mm.keys()):
        if isinstance(inputs_mm[k], torch.Tensor) and inputs_mm[k].dtype.is_floating_point:
            inputs_mm[k] = inputs_mm[k].to(model_dtype)
    # Simple greedy generation (note: some models ignore output_attentions during generate)
    gen_ids = model.generate(
        **inputs_mm,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    # 显示“新增”的生成，不重复输入提示
    prompt_len = inputs_mm.get("input_ids", None)
    if isinstance(prompt_len, torch.Tensor):
        prompt_len = prompt_len.shape[1]
    else:
        prompt_len = 0
    new_ids = gen_ids[0][prompt_len:]
    text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"[GEN] {text}")

    # =========================
    # 2) Attention visualization
    # =========================
    # Vision self-attention (CLS→patch)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    # Match dtype for vision forward
    for k in list(inputs.keys()):
        if isinstance(inputs[k], torch.Tensor) and inputs[k].dtype.is_floating_point:
            inputs[k] = inputs[k].to(model_dtype)
    pixel_values = inputs["pixel_values"]
    if pixel_values.ndim == 5:
        pixel_values = pixel_values[:, 0]

    # Locate vision encoder
    if hasattr(model.model, "vision_tower"):
        vision_model = model.model.vision_tower
    elif hasattr(model.model, "vision_model"):
        vision_model = model.model.vision_model
    else:
        raise RuntimeError("Cannot find vision encoder")

    with torch.no_grad():
        vision_outputs = vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=inputs.get("patch_attention_mask", None),
            output_attentions=True,
            return_dict=True,
        )

    attn = vision_outputs.attentions[args.attn_layer][0]  # (H, N, N)
    attn = attn.mean(0) if args.head == "mean" else attn[int(args.head)]
    cls_attn = attn[0, 1:]
    patch_mask = inputs.get("patch_attention_mask", None)
    if patch_mask is not None:
        pm = patch_mask[0].to(cls_attn.device).bool()
        cls_attn = cls_attn[: pm.shape[0]]
        cls_attn = torch.where(pm, cls_attn, torch.zeros_like(cls_attn[: pm.shape[0]]))
    num_patches = cls_attn.shape[0]
    # Try to use vision config grid if available
    grid_h = None
    grid_w = None
    try:
        img_size = getattr(model.config.vision_config, "image_size", None)
        patch_size = getattr(model.config.vision_config, "patch_size", None)
        if isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
            grid_h = img_size // patch_size
            grid_w = img_size // patch_size
    except Exception:
        pass
    if grid_h is None or grid_w is None or grid_h * grid_w != num_patches:
        grid = int(math.sqrt(num_patches))
        cls_attn = cls_attn[: grid * grid]
        attn_map = cls_attn.reshape(grid, grid)
    else:
        attn_map = cls_attn.reshape(grid_h, grid_w)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
    attn_map = torch.nn.functional.interpolate(
        attn_map[None, None], size=image.size[::-1], mode="bilinear", align_corners=False
    )[0, 0].cpu()
    visualize_map(image, attn_map, "Vision Self-Attn (CLS→patch)", out_dir / "vlm_vision_self_attn.png")

    # Text-conditioned cross-attention
    # Use autocast on CUDA to reduce memory
    autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if args.device == "cuda" and args.dtype == "bf16" else None
    if autocast_ctx is not None:
        with autocast_ctx:
            with torch.no_grad():
                outputs_mm = model(**inputs_mm, output_attentions=True, return_dict=True)
    else:
        with torch.no_grad():
            outputs_mm = model(**inputs_mm, output_attentions=True, return_dict=True)

    # Determine image token length in text space
    vision_hidden = vision_outputs.last_hidden_state
    if hasattr(model.model, "connector"):
        vision_embeds = model.model.connector(vision_hidden)
        n_img_tokens = vision_embeds.shape[1]
    else:
        n_img_tokens = vision_hidden.shape[1]
    # Use patch count from vision self-attn for grid reshape
    n_patches = num_patches

    att_src = None
    if hasattr(outputs_mm, "cross_attentions") and outputs_mm.cross_attentions:
        att_src = outputs_mm.cross_attentions[args.attn_layer]
    elif hasattr(outputs_mm, "attentions") and outputs_mm.attentions:
        att_src = outputs_mm.attentions[args.attn_layer]
    else:
        print("[WARN] No cross/attentions returned; skip")
        return

    att_text_src = att_src[0].mean(0) if args.head == "mean" else att_src[0, int(args.head)]
    tokens = processor.tokenizer.tokenize(args.prompt)
    try:
        tgt_idx = next(i for i, t in enumerate(tokens) if args.token.lower() in t.lower())
    except StopIteration:
        tgt_idx = len(tokens) - 1

    # Locate the image token segment in src axis
    src_len = att_text_src.shape[1]
    start_idx = 0
    if args.cross_loc == "start":
        start_idx = 0
    elif args.cross_loc == "end":
        start_idx = max(0, src_len - n_img_tokens)
    elif args.cross_loc == "placeholder":
        try:
            image_tid = processor.tokenizer.convert_tokens_to_ids(["<image>"])[0]
            input_ids = inputs_mm.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                ids = input_ids[0].tolist()
                if image_tid in ids:
                    ph = ids.index(image_tid)
                    start_idx = min(max(ph, 0), max(0, src_len - n_img_tokens))
                else:
                    start_idx = 0
            else:
                start_idx = 0
        except Exception:
            start_idx = 0
    else:  # auto: find the contiguous window of length n_img_tokens with max attention mass
        import torch.nn.functional as F
        # ensure weight dtype matches att_text_src dtype (fp16/bf16 on CUDA)
        w = torch.ones(n_img_tokens, device=att_text_src.device, dtype=att_text_src.dtype)
        # moving sum via conv1d
        x = att_text_src[tgt_idx][None, None]  # (1,1,src_len)
        scores = F.conv1d(x, w[None, None], stride=1)  # (1,1,src_len - n_img_tokens + 1)
        start_idx = int(torch.argmax(scores).item())
    # Slice full image segment then trim potential special/prefix/suffix tokens
    att_seg_full = att_text_src[tgt_idx, start_idx : start_idx + n_img_tokens]
    extra = max(n_img_tokens - n_patches, 0)
    skip_prefix = args.skip_img_prefix if args.skip_img_prefix >= 0 else (extra if extra <= 3 else 0)
    skip_suffix = args.skip_img_suffix
    left = min(skip_prefix, att_seg_full.shape[0])
    right = max(att_seg_full.shape[0] - skip_suffix, 0)
    att_vec = att_seg_full[left:right]
    if att_vec.shape[0] >= n_patches:
        att_vec = att_vec[:n_patches]
    else:
        pad = torch.zeros(n_patches - att_vec.shape[0], device=att_vec.device, dtype=att_vec.dtype)
        att_vec = torch.cat([att_vec, pad], dim=0)

    # Reshape using grid from vision config if available
    grid_h2 = None
    grid_w2 = None
    try:
        img_size = getattr(model.config.vision_config, "image_size", None)
        patch_size = getattr(model.config.vision_config, "patch_size", None)
        if isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
            grid_h2 = img_size // patch_size
            grid_w2 = img_size // patch_size
    except Exception:
        pass
    if grid_h2 is None or grid_w2 is None or grid_h2 * grid_w2 != n_patches:
        grid = int(math.sqrt(n_patches))
        att_vec = att_vec[: grid * grid]
        attn_map2 = att_vec.reshape(grid, grid)
    else:
        attn_map2 = att_vec.reshape(grid_h2, grid_w2)
    attn_map2 = (attn_map2 - attn_map2.min()) / (attn_map2.max() - attn_map2.min() + 1e-6)
    attn_map2 = torch.nn.functional.interpolate(
        attn_map2[None, None], size=image.size[::-1], mode="bilinear", align_corners=False
    )[0, 0].cpu()
    visualize_map(
        image, attn_map2, f"Cross-Attn for token '{args.token}'", out_dir / "vlm_cross_attn.png"
    )

    # Optional: export interactive BertViz cross-attention HTML
    if args.bertviz_html is not None:
        try:
            from bertviz import head_view
            # Build per-layer attention list limited to prompt (query) and image segment (key)
            att_layers = []
            if hasattr(outputs_mm, "cross_attentions") and outputs_mm.cross_attentions:
                raw_layers = outputs_mm.cross_attentions
            else:
                raw_layers = outputs_mm.attentions
            # prompt tokens and length (use input prompt length to avoid oversized target len)
            tokens_q_raw = tokens
            q_len = min(len(tokens_q_raw), att_text_src.shape[0])
            for layer_att in raw_layers:
                # Expect original layer_att shape: (batch, heads, tgt_len, src_len)
                la = layer_att[0, :, :q_len, start_idx : start_idx + n_img_tokens]  # (heads, q, k)
                la = la.detach().to(torch.float32).cpu()  # (heads, q, k)
                att_layers.append(la)
            try:
                att_tensor = torch.stack(att_layers, dim=0)  # (layers, heads, q, k)
            except Exception as e:
                print(f"[WARN] BertViz: failed to stack attentions ({e}); skip export")
                return
            print(f"[INFO] BertViz att shape: {tuple(att_tensor.shape)} (layers, heads, q, k)")
            # tokens
            tokens_q = tokens_q_raw[:q_len]
            if len(tokens_q) < q_len:
                tokens_q = tokens_q + ["[PAD]"] * (q_len - len(tokens_q))
            # simple image patch labels
            try:
                img_size = getattr(model.config.vision_config, "image_size", None)
                patch_size = getattr(model.config.vision_config, "patch_size", None)
                if isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
                    gh = img_size // patch_size
                    gw = img_size // patch_size
                else:
                    gh = gw = int(math.sqrt(n_img_tokens))
            except Exception:
                gh = gw = int(math.sqrt(n_img_tokens))
            tokens_k = [f"P[{r},{c}]" for r in range(gh) for c in range(gw)][: n_img_tokens]
            import inspect
            sig = inspect.signature(head_view)
            if "sentence_b" in sig.parameters:
                html = head_view(att_tensor, tokens_q, tokens_k, html_action='return', sentence_b=tokens_k)
            else:
                print("[WARN] This bertviz version lacks sentence_b (cross-attn) support; please upgrade: pip install -U bertviz>=1.4.0")
                return
            out_html = Path(args.bertviz_html)
            ensure_dir(out_html.parent)
            with open(out_html, 'w', encoding='utf-8') as f:
                f.write(html.data if hasattr(html, 'data') else str(html))
            print(f"[OK] Saved BertViz HTML: {out_html}")
        except ImportError:
            print("[WARN] bertviz not installed. Install with: pip install bertviz")
        except Exception as e:
            print(f"[WARN] Failed to export BertViz HTML: {e}")

            # Simple interactive HTML export without external libs
            if args.simple_html is not None:
                try:
                    import io, base64
                    from matplotlib.figure import Figure
                    # Prepare layer range (from last)
                    if hasattr(outputs_mm, "cross_attentions") and outputs_mm.cross_attentions:
                        raw_layers = outputs_mm.cross_attentions
                    else:
                        raw_layers = outputs_mm.attentions
                    total_layers = len(raw_layers)
                    max_layers = max(1, min(args.html_max_layers, total_layers))
                    start_layer = max(0, total_layers - max_layers)
                    layer_indices = list(range(start_layer, total_layers))

                    # Head count
                    heads = att_src.shape[1]
                    head_indices = ["mean"] if args.html_heads == "mean" else list(range(heads))

                    # Utility to render attention vector to base64 PNG
                    def render_map(vec: torch.Tensor, title: str) -> str:
                        # reshape to grid
                        try:
                            img_size = getattr(model.config.vision_config, "image_size", None)
                            patch_size = getattr(model.config.vision_config, "patch_size", None)
                            if isinstance(img_size, int) and isinstance(patch_size, int) and patch_size > 0:
                                gh = img_size // patch_size
                                gw = img_size // patch_size
                            else:
                                gh = gw = int(math.sqrt(n_patches))
                        except Exception:
                            gh = gw = int(math.sqrt(n_patches))
                        m = vec[: gh * gw].reshape(gh, gw)
                        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
                        # upscale to image size for nicer display
                        m_up = torch.nn.functional.interpolate(m[None, None], size=image.size[::-1], mode="bilinear", align_corners=False)[0,0].cpu().numpy()
                        fig = Figure(figsize=(6, 6))
                        ax = fig.subplots(1, 1)
                        ax.imshow(image)
                        ax.imshow(m_up, cmap="inferno", alpha=0.75)
                        ax.set_title(title)
                        ax.axis("off")
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches="tight")
                        buf.seek(0)
                        b64 = base64.b64encode(buf.read()).decode("ascii")
                        return f"data:image/png;base64,{b64}"

                    # Build images dict
                    images_dict = {}
                    for li in layer_indices:
                        layer_att = raw_layers[li][0]  # (heads, tgt_len, src_len)
                        # choose token index
                        q_idx = tgt_idx
                        # slice image segment
                        seg = layer_att[:, q_idx, start_idx : start_idx + n_img_tokens]  # (heads, k)
                        # trim prefix/suffix to patches length
                        extra = max(n_img_tokens - n_patches, 0)
                        left = min(args.skip_img_prefix if args.skip_img_prefix >= 0 else (extra if extra <= 3 else 0), seg.shape[1])
                        right = max(seg.shape[1] - args.skip_img_suffix, 0)
                        seg = seg[:, left:right]
                        if seg.shape[1] >= n_patches:
                            seg = seg[:, :n_patches]
                        else:
                            pad = torch.zeros((seg.shape[0], n_patches - seg.shape[1]), device=seg.device, dtype=seg.dtype)
                            seg = torch.cat([seg, pad], dim=1)
                        # render
                        if "mean" in head_indices:
                            vec = seg.mean(0)
                            images_dict[f"layer{li}-head_mean"] = render_map(vec, f"Layer {li} - head mean - token '{args.token}'")
                        if args.html_heads == "all":
                            for hi in range(heads):
                                vec = seg[hi]
                                images_dict[f"layer{li}-head{hi}"] = render_map(vec, f"Layer {li} - head {hi} - token '{args.token}'")

                    # Write simple HTML with dropdowns
                    out_html = Path(args.simple_html)
                    ensure_dir(out_html.parent)
                    # generate options
                    layer_opts = "".join([f"<option value='layer{li}'>Layer {li}</option>" for li in layer_indices])
                    if args.html_heads == "mean":
                        head_opts = "<option value='head_mean'>mean</option>"
                    else:
                        head_opts = "<option value='head_mean'>mean</option>" + "".join([f"<option value='head{hi}'>head {hi}</option>" for hi in range(heads)])
                    # initial key
                    init_key = f"layer{layer_indices[-1]}-head_mean"
                    # images JSON (simple)
                    import json
                    images_json = json.dumps(images_dict)
                    html = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset='utf-8'/>
          <title>Cross-Attn Viewer</title>
          <style> body {{ font-family: sans-serif; }} img {{ max-width: 90vw; height: auto; }} </style>
          <script>
            const IMAGES = {images_json};
            function updateImage() {{
              const layerSel = document.getElementById('layerSel').value;
              const headSel = document.getElementById('headSel').value;
              const key = layerSel + '-' + headSel;
              const img = document.getElementById('attImg');
              img.src = IMAGES[key] || '';
            }}
            window.onload = () => updateImage();
          </script>
        </head>
        <body>
          <h3>Cross-Attention for token '{args.token}'</h3>
          <div>
            Layer: <select id='layerSel' onchange='updateImage()'>{layer_opts}</select>
            Head: <select id='headSel' onchange='updateImage()'>{head_opts}</select>
          </div>
          <div>
            <img id='attImg' src='{images_dict.get(init_key, '')}'/>
          </div>
        </body>
        </html>
        """
                    with open(out_html, 'w', encoding='utf-8') as f:
                        f.write(html)
                    print(f"[OK] Saved Simple HTML: {out_html}")
                except Exception as e:
                    print(f"[WARN] Failed to export simple HTML: {e}")


if __name__ == "__main__":
    main()
