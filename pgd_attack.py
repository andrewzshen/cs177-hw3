#!/usr/bin/env python

import sys
import io
import requests
import torch

from typing import Any
from PIL import Image
from torch import Tensor
from torchvision import transforms

SERVER_IP: str = "cs177.seclab.cs.ucsb.edu"
port: int = 0 

SAMPLE_IMAGE: str = "images/cat_001.jpg"  

EPSILON: float = 16 / 255
ALPHA: float = 2 / 255
STEPS: int = 100

MEAN: Tensor = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

STD: Tensor = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)

def to_vect(path: str) -> Tensor:
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(transforms.Resize(256)(img))
    _, h, w = t.shape
    y, x = (h - 224) // 2, (w - 224) // 2
    clean = t[:, y: y + 224, x: x + 224].clone()
    return clean

def to_png(vect: Tensor) -> bytes:
    buf = io.BytesIO()
    transforms.ToPILImage()(vect.clamp(0,1)).save(buf, format="PNG")
    return buf.getvalue()

def predict(img_vect: Tensor) -> dict[str, Any]: 
    png: bytes = to_png(img_vect)

    r: requests.Response = requests.post(
        f"http://{SERVER_IP}:{port}/api/predict",
        files={"image": (SAMPLE_IMAGE, png, "image/png")}
    )

    r.raise_for_status()
    
    return r.json()

def gradient(img_vect: Tensor, label: str) -> dict[str, Any]:
    png: bytes = to_png(img_vect)

    r: requests.Response = requests.post(
        f"http://{SERVER_IP}:{port}/api/predict",
        files={"image": (SAMPLE_IMAGE, png, "image/png")},
        data={"target": label}
    )

    r.raise_for_status()

    return r.json()

def verify(img_vect: Tensor) -> dict[str, Any]:
    png: bytes = to_png(img_vect)

    r: requests.Response = requests.post(
        f"http://{SERVER_IP}:{port}/api/verify",
        files={"image": (SAMPLE_IMAGE, png, "image/png")}
    )
    
    r.raise_for_status()

    return r.json()

def pgd_attack(clean: Tensor) -> Tensor:
    adv = clean.clone()
    
    pred_json: dict[str, Any] = predict(clean)

    pred = pred_json["prediction"]
    target: str = "cat" if pred == "dog" else "dog" 

    print(f"Original prediction: {pred}")
    print(f"Target: {target}")

    for i in range(STEPS):
        grad_json: dict[str, Any] = gradient(adv, target)
        
        grad: Tensor = torch.tensor(grad_json["gradient"]).view(3, 224, 224) 
        grad = grad / STD

        nudged: Tensor = adv - ALPHA * grad.sign()
        delta: Tensor = nudged - clean
        delta = torch.clamp(delta, -EPSILON, EPSILON)
        adv = torch.clamp(clean + delta, 0, 1)

        curr_pred: str = grad_json["prediction"]
        confidence: float = grad_json["confidence"]

        print(f"Iteration {i + 1:03d}: Target: {target}, Prediction: {curr_pred}, Confidence: {confidence}")

        if curr_pred == target:
            print("\nSUCCESS")
            break

    return adv

def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python dns_amplify.py <port>") 
    
    global port
    port = int(sys.argv[1])
     
    clean: Tensor = to_vect(SAMPLE_IMAGE)

    adv: Tensor = pgd_attack(clean)

    with open("adv.png", "wb") as f:
        f.write(to_png(adv))

    print("\nSaved: adv.png")

    result: dict[str, Any] = verify(adv)

    success: bool = result["success"]
    message: str = result["message"]

    print(message)

    if success:
        print(result["flag"])
    
if __name__ == "__main__":
    main()
