"""Track 3 frozen-encoder extraction library (local Mac or Kaggle GPU).

All encoders are frozen, pretrained, and process every file independently.
Faces: farl (FaRL ViT-B/16 LAION-Face), clip (OpenAI CLIP ViT-L/14).
Voices: simam (WeSpeaker VoxBlink2 SimAM-R100), ecapa (WeSpeaker VoxCeleb
ECAPA-TDNN1024-LM), campplus (WeSpeaker VoxCeleb CAM++-LM), wavlm4 (WavLM-Base+
layer-4 mean, 8 s centre crop). Recipes match the FLAG caches used in Track 2.
"""
from __future__ import annotations
import math, os, sys, types, wave
from pathlib import Path
import numpy as np, torch

DEV = os.environ.get("T3_DEVICE") or ("cuda" if torch.cuda.is_available() else
                                      "mps" if torch.backends.mps.is_available() else "cpu")
TOOLS = Path(os.environ.get("T3_TOOLS", Path(__file__).resolve().parents[2] / ".tools"))
HF_SPK = {"ecapa": "Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM",
          "campplus": "Wespeaker/wespeaker-voxceleb-campplus-LM",
          "r293": "Wespeaker/wespeaker-voxceleb-resnet293-LM",
          "sam100ft": "bzikst/voxblink2_samresnet100_ft",
          "r221": "Wespeaker/wespeaker-voxceleb-resnet221-LM"}  # RF reproduction (2026-11-05); repo files named voxceleb_resnet221_LM.*
HF_FILES = {"r221": ("voxceleb_resnet221_LM.yaml", "voxceleb_resnet221_LM.pt", "26732f43821f238bfb258c5ef54924a798e62118")}


def _unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


# ----------------------------------------------------------------- faces
def face_encoder(name, turbo=False):  # turbo: decode JPEGs with OpenCV's libjpeg-turbo (as the Kaggle-made caches, e.g. FaRL)
    from PIL import Image, ImageOps
    if name == "farl":
        sys.path.insert(0, str(TOOLS / "farl/vendor")); sys.path.insert(0, str(TOOLS / "farl/CLIP"))
        import clip
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor
        payload = torch.load(TOOLS / "farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth", map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in payload.get("state_dict", payload).items()}
        mim = ("visual.mask_token", "visual.lm_transformer.", "visual.ln_lm.", "visual.lm_head.")
        model = clip.model.build_model({k: v for k, v in state.items() if not k.startswith(mim)}).float().eval().to(DEV)
        pre = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224), ToTensor(),
                       Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        fn = lambda imgs: model.encode_image(torch.stack([pre(i) for i in imgs]).to(DEV))
    elif name == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(DEV)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
        def fn(imgs):  # == get_image_features (transformers 4.x), explicit for 5.x compatibility
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(DEV)
            return model.visual_projection(model.vision_model(pixel_values=px).pooler_output)
    else:
        raise ValueError(name)

    def run(paths, batch=64, log=None):
        out = []
        with torch.inference_mode():
            for s in range(0, len(paths), batch):
                imgs = [Image.fromarray(__import__("cv2").cvtColor(__import__("cv2").imread(str(p)), 4)) if turbo else  # 4 = COLOR_BGR2RGB
                        ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths[s:s + batch]]
                out.append(_unit(fn(imgs).float().cpu().numpy()))
                if log and s % (batch * 50) == 0:
                    log(f"{name} {s}/{len(paths)}")
        return np.concatenate(out)
    return run


# ----------------------------------------------------------------- voices
def _pcm16(path):
    with wave.open(str(path)) as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        assert w.getsampwidth() == 2, path
        x = np.frombuffer(w.readframes(n), "<i2").reshape(n, ch).T.astype(np.float32)
    x = x.mean(0, keepdims=True) if ch > 1 else x
    if sr != 16000:
        from scipy.signal import resample_poly
        g = math.gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g, axis=1).astype(np.float32)
    return x  # int16 scale


def _wespeaker(name):
    import yaml, torchaudio
    root = TOOLS / "wespeaker/wespeaker"
    if "wespeaker" not in sys.modules or not hasattr(sys.modules["wespeaker"], "__path__"):
        pkg = types.ModuleType("wespeaker"); pkg.__path__ = [str(root)]; sys.modules["wespeaker"] = pkg
        fe = types.ModuleType("wespeaker.frontend"); fe.__path__ = [str(root / "frontend")]; sys.modules["wespeaker.frontend"] = fe
    from wespeaker.models.speaker_model import get_speaker_model
    if name == "simam":  # T3_TOOLS/simam/unpacked/voxblink2_samresnet100/avg_model.pt
        cfg = dict(model="SimAM_ResNet100_ASP", model_args=dict(in_planes=64, embed_dim=256, acoustic_dim=80))
        weights = TOOLS / "simam/unpacked/voxblink2_samresnet100/avg_model.pt"
    else:
        from huggingface_hub import hf_hub_download
        cf, wf, rev = HF_FILES.get(name, ("config.yaml", "avg_model.pt", None))
        cfg = yaml.safe_load(open(hf_hub_download(HF_SPK[name], cf, revision=rev)))
        weights = hf_hub_download(HF_SPK[name], wf, revision=rev)
    model = get_speaker_model(cfg["model"])(**cfg["model_args"])
    missing, _ = model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    assert not [k for k in missing if "projection" not in k], missing
    model.eval().to(DEV)

    def one(path):
        x = torch.from_numpy(_pcm16(path))
        n = x.shape[1]
        k = max(1, math.ceil(n / (30 * 16000)))  # balanced <=30 s segments (RAW-S01 policy)
        bounds = np.linspace(0, n, k + 1, dtype=np.int64)
        embs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            fb = torchaudio.compliance.kaldi.fbank(x[:, a:b], num_mel_bins=80, frame_length=25, frame_shift=10,
                                                   dither=0.0, sample_frequency=16000, window_type="hamming")
            fb = fb - fb.mean(0, keepdim=True)
            with torch.inference_mode():
                e = model(fb[None].to(DEV))
            e = e[-1] if isinstance(e, (tuple, list)) else e
            e = e[0].float().cpu(); embs.append(e / e.norm())
        e = torch.stack(embs).mean(0)
        return (e / e.norm()).numpy()
    return one


def _wavlm4():
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(DEV)

    def one(path):
        a = _pcm16(path)[0] / 32768.0
        m = 8 * 16000
        if len(a) > m:
            s = (len(a) - m) // 2; a = a[s:s + m]
        with torch.inference_mode():
            h = model(torch.from_numpy(a)[None].to(DEV), output_hidden_states=True).hidden_states[4][0].float()
        v = h.mean(0)
        return (v / v.norm()).cpu().numpy()
    return one


def voice_encoder(name):
    one = _wavlm4() if name == "wavlm4" else _wespeaker(name)

    def run(paths, log=None):
        out = []
        for i, p in enumerate(paths):
            out.append(one(p))
            if log and i % 1000 == 0:
                log(f"{name} {i}/{len(paths)}")
        return _unit(np.stack(out))
    return run
