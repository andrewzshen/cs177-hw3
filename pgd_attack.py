#!/usr/bin/env python

import io
import requests
import torch

from PIL import Image
from torch import Tensor
from torchvision import transforms

SERVER_IP: str = "cs177.seclab.cs.ucsb.edu"
PORT: int = 0

SAMPLE_IMAGE: str = "image.png"  

def to_vect(path: str) -> Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(transforms.Resize(256)(img))
    _, h, w = t.shape
    y, x = (h - 224) // 2, (w - 224) // 2
    clean = t[:, y:y+224, x:x+224].clone()
    adv = clean.clone()

def to_png(vect: Tensor) -> bytes:
    buf = io.BytesIO()
    transforms.ToPILImage()(vect.clamp(0,1)).save(buf, format="PNG")
    return buf.getvalue()

def predict(img_vect):
    png = to_png(img_vect)

    r = requests.post(
        f"https://{SERVER_IP}:{PORT}/api/predict",
        files={"file": (SAMPLE_IMAGE, png, "image/png")}
    )

def gradient(im):
    pass 

def main():
    pass 
    
if __name__ == "__main__":
    main()
