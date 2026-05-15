#!/usr/bin/env python

import sys
import io
import torch

from PIL import Image
from torchvision import transforms

SERVER_IP = "cs177.seclab.cs.ucsb.edu"

def to_vect(path):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()(transforms.Resize(256)(img))
    _, h, w = t.shape
    y, x = (h - 224) // 2, (w - 224) // 2
    clean = t[:, y:y+224, x:x+224].clone()
    adv = clean.clone()

def to_png(vect):
    buf = io.BytesIO()
    transforms.ToPILImage()(vect.clamp(0,1)).save(buf, format="PNG")
    return buf.getvalue()

def gradient():
    pass 

def predict():
    pass

def main():
    if len(sys.argv) != 2:
        print("Usage: python pgd_attack.py <port>")
        return 1 

    port = int(sys.argv[1])

    packet = IP(dst=SERVER_IP) / UDP(dport=port) / DNS(qd=DNSQR(qname=DOMAIN, qtype=1))

    response = sr1(packet, timeout=3, verbose=0)

    if response:
        response.show()

        request_size = len(bytes(packet[DNS]))
        response_size = len(bytes(response[DNS])) if response else 0

        print(f"Request Size: {request_size}")
        print(f"Response Size: {response_size}")
        print(f"BAF: {response_size / request_size:.2f}")
    else:
        print("No response")
    
if __name__ == "__main__":
    main()
