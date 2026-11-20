"""Frozen FaRL (ViT-B/16 LAION-Face) and CLIP ViT-L/14 image embeddings for a list of face crops.

Same preprocessing as the FLAG caches (FaRL: resize224 bicubic + centre crop + CLIP norm;
CLIP: official CLIPProcessor). Each image independent; L2-normalised.
Usage: python extract_faces.py <list.csv with column path> <out_dir> [farl] [clip] [--flip]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_farl():
    sys.path.insert(0, str(ROOT / ".tools/farl/vendor")); sys.path.insert(0, str(ROOT / ".tools/farl/CLIP"))
    import clip
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
    payload = torch.load(ROOT / ".tools/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", map_location="cpu", weights_only=False)
    state = {k.removeprefix("module."): v for k, v in payload.get("state_dict", payload).items()}
    mim = ("visual.mask_token", "visual.lm_transformer.", "visual.ln_lm.", "visual.lm_head.")
    state = {k: v for k, v in state.items() if not k.startswith(mim)}
    model = clip.model.build_model(state).float().eval().to(DEV)
    pre = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
                   Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    return lambda imgs: model.encode_image(torch.stack([pre(i) for i in imgs]).to(DEV))


def load_clip():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(DEV)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return lambda imgs: model.get_image_features(**proc(images=imgs, return_tensors="pt").to(DEV))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flip = "--flip" in sys.argv
    paths = pd.read_csv(args[0]).path.tolist(); out = Path(args[1]); out.mkdir(parents=True, exist_ok=True)
    for enc in args[2:]:
        fn = dict(farl=load_farl, clip=load_clip)[enc]()
        feats = []
        with torch.inference_mode():
            for s in range(0, len(paths), 64):
                imgs = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths[s:s+64]]
                if flip:
                    imgs = [ImageOps.mirror(i) for i in imgs]
                e = fn(imgs).float().cpu().numpy()
                feats.append(e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8))
                if s % 6400 == 0:
                    print(enc, s, len(paths), flush=True)
        np.save(out / f"{enc}{'_flip' if flip else ''}.float32.npy", np.concatenate(feats).astype(np.float32))


if __name__ == "__main__":
    main()
