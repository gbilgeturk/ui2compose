# SPDX-FileCopyrightText: 2026 Murat Saran <saran@cankaya.edu.tr>
# SPDX-FileCopyrightText: 2026 Göktürk Bilgetürk <gbilgeturk@yahoo.com>
#
# SPDX-License-Identifier: MIT

"""Checks whether fine-tuned YOLOv8s checkpoints were initialised from the COCO-pretrained yolov8s.pt.

Usage: python scripts/coco_init_check.py <coco yolov8s.pt> <checkpoint.pt>...
Prints, per checkpoint, the train_args start model and the mean cosine similarity of every
weight tensor shared with yolov8s.pt, against the COCO weights and against a random yolov8s init.
"""
import sys, json, torch
import torch.nn.functional as F
from ultralytics import YOLO

coco_path, *ckpts = sys.argv[1:]
def sd(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = (ck.get('ema') or ck['model']).float()
    return {k: v for k, v in m.state_dict().items() if v.ndim >= 2}, ck.get('train_args', {})
coco, _ = sd(coco_path)
torch.manual_seed(0)
rand = {k: v for k, v in YOLO('yolov8s.yaml').model.float().state_dict().items() if v.ndim >= 2}

def cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()
def layer(k):
    return int(k.split('.')[1])
out = {}
for path in ckpts:
    w, args = sd(path)
    rows = []
    for k, v in w.items():
        if k in coco and coco[k].shape == v.shape:
            rows.append((layer(k), cos(v, coco[k]), cos(v, rand[k]) if k in rand and rand[k].shape == v.shape else None))
    by = {}
    for l, c, r in rows:
        by.setdefault(l, []).append((c, r))
    summary = {l: (round(sum(c for c, _ in v) / len(v), 3), round(sum(r for _, r in v if r is not None) / max(1, sum(1 for _, r in v if r is not None)), 3)) for l, v in sorted(by.items())}
    out[path.split('/')[-1]] = {'train_args_model': args.get('model'), 'pretrained': args.get('pretrained'), 'resume': args.get('resume'),
                                'epochs': args.get('epochs'), 'data': args.get('data'),
                                'mean_cos_vs_coco_backbone_0_9': round(sum(summary[l][0] for l in summary if l <= 9) / sum(1 for l in summary if l <= 9), 3),
                                'mean_cos_vs_random_backbone_0_9': round(sum(summary[l][1] for l in summary if l <= 9) / sum(1 for l in summary if l <= 9), 3),
                                'per_layer_cos_vs_coco_vs_random': summary}
print(json.dumps(out, indent=1))
