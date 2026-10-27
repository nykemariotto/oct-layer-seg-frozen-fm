"""Frozen ViT-L encoders (RETFound, DINOv2) with the two trainable decoders compared in the paper.

SETR-PUP: skip-free; four progressive-upsampling stages on the last-block patch tokens.
UNETR (2D adaptation): skips from blocks 6, 12, 18 and 24 (indices 5, 11, 17, 23) plus a convolutional
stem on the input image.
The encoders are frozen and always kept in eval mode; checkpoints store only the decoder.
RETFound weights: https://huggingface.co/YukunZhou/RETFound_mae_natureOCT (requires accepting its terms).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download

SIZE = 224
RETFOUND_REPO = "YukunZhou/RETFound_mae_natureOCT"
RETFOUND_FILE = "RETFound_mae_natureOCT.pth"
DINO_MODEL = "vit_large_patch14_dinov2"   # timm, LVD-142M weights
TAP_BLOCKS = [5, 11, 17, 23]


def retfound_vit():
    """RETFound ViT-L/16 encoder (MAE decoder and mask token discarded)."""
    ck = hf_hub_download(repo_id=RETFOUND_REPO, filename=RETFOUND_FILE)
    vit = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0, global_pool="")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    sd = c["model"] if "model" in c else c
    enc = {k: v for k, v in sd.items() if not k.startswith("decoder") and not k.startswith("mask_token")}
    vit.load_state_dict(enc, strict=False)
    return vit


def dinov2_vit():
    return timm.create_model(DINO_MODEL, pretrained=True, num_classes=0, global_pool="", img_size=SIZE)


def setr_pup(n_classes):
    def up(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))

    return nn.Sequential(up(1024, 512), up(512, 256), up(256, 128), up(128, 64), nn.Conv2d(64, n_classes, 1))


class RETFoundSeg(nn.Module):
    """Frozen RETFound -> 14x14x1024 patch tokens -> SETR-PUP -> 224x224 logits."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.vit = retfound_vit()
        for p in self.vit.parameters():
            p.requires_grad = False
        self.decoder = setr_pup(n_classes)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.vit.forward_features(x)[:, 1:, :]       # drop CLS: (B, 196, 1024)
        B, N, C = f.shape
        g = int(N ** 0.5)
        feat = f.transpose(1, 2).reshape(B, C, g, g)
        return self.decoder(feat)


class DINOv2Seg(nn.Module):
    """Frozen DINOv2 -> 16x16x1024 patch tokens -> the same SETR-PUP -> 256x256, interpolated to 224x224."""

    def __init__(self, n_classes=6):
        super().__init__()
        self.vit = dinov2_vit()
        self.n_prefix = self.vit.num_prefix_tokens
        for p in self.vit.parameters():
            p.requires_grad = False
        self.decoder = setr_pup(n_classes)

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.vit.forward_features(x)[:, self.n_prefix:, :]
        B, N, C = f.shape
        g = int(N ** 0.5)
        feat = f.transpose(1, 2).reshape(B, C, g, g)
        out = self.decoder(feat)
        return F.interpolate(out, size=(SIZE, SIZE), mode="bilinear", align_corners=False)


def build_encoder(enc):
    if enc == "ret":
        return retfound_vit()
    if enc == "dino":
        return dinov2_vit()
    raise ValueError(enc)


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNETRDecoder(nn.Module):
    """Skips fused progressively from the token grid (14x14 or 16x16) up to 224x224, plus a stem on the image."""

    def __init__(self, n_classes=6, c_in=1024):
        super().__init__()
        self.pd = cbr(c_in, 192)
        self.p17 = nn.Conv2d(c_in, 96, 1)
        self.p11 = nn.Conv2d(c_in, 96, 1)
        self.p5 = nn.Conv2d(c_in, 96, 1)
        self.d1 = cbr(192 + 96, 128)
        self.d2 = cbr(128 + 96, 96)
        self.d3 = cbr(96 + 96, 64)
        self.stem = cbr(3, 32)
        self.d4 = cbr(64 + 32, 48)
        self.head = nn.Conv2d(48, n_classes, 1)

    @staticmethod
    def _up2(t):
        return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)

    @staticmethod
    def _match(s, ref):
        return F.interpolate(s, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, skips, image):
        z5, z11, z17, z23 = skips
        x = self.pd(z23)
        u = self._up2(x); x = self.d1(torch.cat([u, self._match(self.p17(z17), u)], 1))
        u = self._up2(x); x = self.d2(torch.cat([u, self._match(self.p11(z11), u)], 1))
        u = self._up2(x); x = self.d3(torch.cat([u, self._match(self.p5(z5), u)], 1))
        x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        x = self.d4(torch.cat([x, self.stem(image)], 1))
        return self.head(x)


class ViTUNetr(nn.Module):
    def __init__(self, enc, n_classes=6):
        super().__init__()
        self.vit = build_encoder(enc)
        self.n_prefix = self.vit.num_prefix_tokens
        for p in self.vit.parameters():
            p.requires_grad = False
        self._feats = []
        for bi in TAP_BLOCKS:
            self.vit.blocks[bi].register_forward_hook(self._mk_hook())
        self.decoder = UNETRDecoder(n_classes=n_classes)

    def _mk_hook(self):
        def hook(m, i, o):
            self._feats.append(o)
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, x):
        self._feats = []
        with torch.no_grad():
            self.vit.forward_features(x)
        skips = []
        for f in self._feats:
            t = f[:, self.n_prefix:, :]
            B, N, C = t.shape
            g = int(N ** 0.5)
            skips.append(t.transpose(1, 2).reshape(B, C, g, g).float())
        return self.decoder(skips, x)


def n_dec_params(model):
    return sum(p.numel() for p in model.decoder.parameters())
