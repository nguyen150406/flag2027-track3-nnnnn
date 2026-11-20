"""Convert the FaRL ViT-B/16 checkpoint (OpenAI-CLIP layout) to a HF CLIPVisionModelWithProjection, so t3_ft_clip.py can put
the same LoRA layout on it. Verifies the HF model reproduces FaRL's encode_image (cos 1.0) and the FLAG FaRL cache.
Usage: python t3_farl_to_hf.py  -> .tools/farl/farl_hf/"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / ".tools/farl/farl_hf"
payload = torch.load(ROOT / ".tools/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", map_location="cpu", weights_only=False)
S = {k.removeprefix("module."): v.float() for k, v in payload.get("state_dict", payload).items() if k.removeprefix("module.").startswith("visual.")}
L = len({k.split(".")[3] for k in S if k.startswith("visual.transformer.resblocks.")}); D = S["visual.conv1.weight"].shape[0]
cfg = CLIPVisionConfig(hidden_size=D, intermediate_size=4 * D, num_hidden_layers=L, num_attention_heads=D // 64, image_size=224,
                       patch_size=S["visual.conv1.weight"].shape[-1], projection_dim=S["visual.proj"].shape[1], hidden_act="quick_gelu", layer_norm_eps=1e-5)
H = {"vision_model.embeddings.class_embedding": S["visual.class_embedding"],
     "vision_model.embeddings.position_embedding.weight": S["visual.positional_embedding"],
     "vision_model.embeddings.patch_embedding.weight": S["visual.conv1.weight"],
     "vision_model.pre_layrnorm.weight": S["visual.ln_pre.weight"], "vision_model.pre_layrnorm.bias": S["visual.ln_pre.bias"],
     "vision_model.post_layernorm.weight": S["visual.ln_post.weight"], "vision_model.post_layernorm.bias": S["visual.ln_post.bias"],
     "visual_projection.weight": S["visual.proj"].T.contiguous()}
for i in range(L):
    o, h = f"visual.transformer.resblocks.{i}.", f"vision_model.encoder.layers.{i}."
    for (w, b), n in zip(zip(S[o + "attn.in_proj_weight"].chunk(3), S[o + "attn.in_proj_bias"].chunk(3)), ("q", "k", "v")):
        H[h + f"self_attn.{n}_proj.weight"], H[h + f"self_attn.{n}_proj.bias"] = w, b
    for src, dst in (("attn.out_proj", "self_attn.out_proj"), ("ln_1", "layer_norm1"), ("ln_2", "layer_norm2"), ("mlp.c_fc", "mlp.fc1"), ("mlp.c_proj", "mlp.fc2")):
        H[h + dst + ".weight"], H[h + dst + ".bias"] = S[o + src + ".weight"], S[o + src + ".bias"]
model = CLIPVisionModelWithProjection(cfg).eval()
missing, unexpected = model.load_state_dict(H, strict=False)
assert not unexpected and all("position_ids" in m for m in missing), (missing, unexpected)

# check 1: identical to FaRL's own encode_image on random pixels
sys.path.insert(0, str(ROOT / ".tools/farl/vendor")); sys.path.insert(0, str(ROOT / ".tools/farl/CLIP"))
import clip
st = {k.removeprefix("module."): v for k, v in payload.get("state_dict", payload).items()}
st = {k: v for k, v in st.items() if not k.startswith(("visual.mask_token", "visual.lm_transformer.", "visual.ln_lm.", "visual.lm_head."))}
ref = clip.model.build_model(st).float().eval()
x = torch.randn(4, 3, 224, 224)
with torch.no_grad():
    c = F.cosine_similarity(model(pixel_values=x).image_embeds, ref.encode_image(x)).min().item()
print(f"HF vs FaRL encode_image: min cos {c:.6f}"); assert c > 0.9999
# check 2: FLAG train faces through the t3_ft_clip preprocessing vs the FLAG FaRL cache used by production
sys.path.insert(0, str(Path(__file__).parent)); import t3_clipft_encode as E, t3data as D
meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id"); tr = meta[meta.partition.eq("train")]
j = np.random.default_rng(0).choice(len(tr), 48, replace=False)
with torch.no_grad():
    e = F.normalize(model(pixel_values=torch.stack([E.load(ROOT / "data/train_set/train_set" / p) for p in tr.face_relpath.iloc[j]])).image_embeds, dim=1).numpy()
cache = np.asarray(D.face("farl"), np.float32)[tr.row_id.to_numpy()[j]]; cache /= np.linalg.norm(cache, axis=1, keepdims=True)
cs = (e * cache).sum(1); print(f"HF FaRL vs FLAG FaRL cache: mean cos {cs.mean():.4f} min {cs.min():.4f}"); assert cs.mean() > 0.99
model.save_pretrained(OUT); print("saved", OUT)
