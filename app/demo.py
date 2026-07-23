#!/usr/bin/env python3
"""
Streamlit Demo — UI Screenshot → Jetpack Compose Code
=====================================================
Kullanım:
     .venv/bin/python -m streamlit run app/demo.py
"""

import streamlit as st
import streamlit.components.v1 as components
import sys
import json
import tempfile
from pathlib import Path
from collections import Counter
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ── Path setup ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.infer.detect_components import ComponentDetector
from src.infer.build_ui_graph import UIGraphBuilder
from src.infer.generate_compose_code import ComposeCodeGenerator

# ── Constants ───────────────────────────────────────────────
CLASS_COLORS = {
    "Button": "#FF6B6B", "TextView": "#4ECDC4", "EditText": "#45B7D1",
    "ImageView": "#96CEB4", "CheckBox": "#FFEAA7", "Switch": "#DDA0DD",
    "RadioButton": "#98D8C8", "ProgressBar": "#85C1E9", "SeekBar": "#85C1E9",
    "Spinner": "#BB8FCE", "ListView": "#F7DC6F", "RecyclerView": "#F8B500",
    "WebView": "#58D68D", "CardView": "#F8B500",
}

EDGE_COLORS = {
    "parent_child": "#2ECC71", "above": "#3498DB", "below": "#9B59B6",
    "left_of": "#E74C3C", "right_of": "#F39C12",
}


# ── Helpers ─────────────────────────────────────────────────
def find_models() -> list[Path]:
    """runs/ altındaki eğitilmiş model ağırlıklarını (best.pt) listeler.

    Girdi:  yok
    Çıktı:  sıralı best.pt Path listesi; runs/ klasörü yoksa boş liste
    """
    runs_dir = PROJECT_ROOT / "runs"
    if not runs_dir.exists():
        return []
    return sorted(runs_dir.glob("*/weights/best.pt"))


LAYOUT_COLORS = {
    "Column": (255, 165, 0),    # Orange
    "Row": (0, 200, 0),         # Green
    "Card": (100, 100, 255),    # Blue
    "Box": (180, 180, 0),       # Yellow-ish
}


def _get_layout_type(node: dict, children: list) -> str:
    """Düğümün sınıfına ve çocuklarının konum dağılımına göre yerleşim tipini belirler.

    Girdi:  node — hiyerarşi düğümü sözlüğü
            children — düğümün çocuk düğümlerinin listesi
    Çıktı:  "Row", "Column" veya "Card" yerleşim tipi dizgesi
    """
    cls = node.get("class", "")
    if cls == "SyntheticRow":
        return "Row"
    if cls == "SyntheticColumn":
        return "Column"
    if cls == "CardView":
        return "Card"
    if len(children) < 2:
        return "Column"
    bboxes = [c.get("bbox", [0, 0, 0, 0]) for c in children]
    cy_values = [(b[1] + b[3]) / 2 for b in bboxes]
    cx_values = [(b[0] + b[2]) / 2 for b in bboxes]
    y_spread = max(cy_values) - min(cy_values)
    x_spread = max(cx_values) - min(cx_values)
    return "Row" if x_spread > y_spread else "Column"


def draw_layout_overlay(image: Image.Image, graph: dict) -> Image.Image:
    """Yerleşim gruplarını (Row, Column, Card) görüntü üzerine yarı saydam çerçevelerle çizer.

    Girdi:  image — PIL görüntüsü
            graph — hierarchy bilgisi içeren UI grafı sözlüğü
    Çıktı:  yerleşim çerçeveleri ve etiketleri eklenmiş yeni PIL görüntüsü
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_cv.copy()

    def draw_node(node, depth=0):
        """Bir düğümün çocuklarını kapsayan yerleşim çerçevesini özyinelemeli olarak çizer.

        Girdi:  node — hiyerarşi düğümü sözlüğü
                depth — özyineleme derinliği (çerçeve ofsetini büyütmek için)
        Çıktı:  yok (img_cv ve overlay üzerine çizim yan etkisi)
        """
        children = node.get("children", [])
        if not children:
            return
        layout = _get_layout_type(node, children)
        color = LAYOUT_COLORS.get(layout, (180, 180, 180))

        # Compute bounding box that covers all children
        all_bboxes = []
        def collect_bboxes(n):
            """Düğümün ve tüm alt düğümlerinin bbox'larını all_bboxes listesine toplar.

            Girdi:  n — hiyerarşi düğümü sözlüğü
            Çıktı:  yok (all_bboxes listesine ekleme yan etkisi)
            """
            if "bbox" in n:
                all_bboxes.append(n["bbox"])
            for c in n.get("children", []):
                collect_bboxes(c)
        for c in children:
            collect_bboxes(c)

        if all_bboxes:
            x1 = int(min(b[0] for b in all_bboxes)) - 4 - depth * 3
            y1 = int(min(b[1] for b in all_bboxes)) - 4 - depth * 3
            x2 = int(max(b[2] for b in all_bboxes)) + 4 + depth * 3
            y2 = int(max(b[3] for b in all_bboxes)) + 4 + depth * 3

            # Semi-transparent fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # Solid border
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            # Label
            cv2.putText(img_cv, layout, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for c in children:
            draw_node(c, depth + 1)

    for root in graph.get("hierarchy", {}).get("roots", []):
        draw_node(root)

    # Blend overlay
    alpha = 0.15
    cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0, img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Tespit kutularını sınıf renkleri ve güven etiketleriyle görüntü üzerine çizer.

    Girdi:  image — PIL görüntüsü
            detections — tespit sözlüklerinin listesi
    Çıktı:  kutular ve etiketler eklenmiş yeni PIL görüntüsü
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color_hex = CLASS_COLORS.get(det["class_name"], "#808080")
        color_bgr = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 2)
        cv2.putText(img_cv, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 2)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def plot_graph(graph: dict, detections: list, img_w: int, img_h: int) -> plt.Figure:
    """UI grafını düğüm daireleri ve ilişki oklarıyla matplotlib figürü olarak çizer.

    Girdi:  graph — edges listesi içeren UI grafı sözlüğü
            detections — düğüm merkezleri için tespit listesi
            img_w — görüntü genişliği (piksel)
            img_h — görüntü yüksekliği (piksel)
    Çıktı:  matplotlib Figure nesnesi
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.set_aspect("equal")
    ax.set_title("Graph: Nodes & Edges", fontsize=13, fontweight="bold")

    node_centers = {}
    for det in detections:
        cx, cy = det["center"]
        node_centers[det["id"]] = (cx, cy)
        color = CLASS_COLORS.get(det["class_name"], "#808080")
        r = min(img_w, img_h) * 0.018
        circle = plt.Circle((cx, cy), r, color=color, ec="black", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, str(det["id"]), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    for edge in graph.get("edges", []):
        sid, tid = edge["source"], edge["target"]
        if sid in node_centers and tid in node_centers:
            x1, y1 = node_centers[sid]
            x2, y2 = node_centers[tid]
            color = EDGE_COLORS.get(edge["relation"], "#808080")
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=2)

    handles = []
    present = {e["relation"] for e in graph.get("edges", [])}
    for rel, color in EDGE_COLORS.items():
        if rel in present:
            handles.append(mpatches.Patch(color=color, label=rel))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax.set_facecolor("#f5f5f5")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_adjacency(graph: dict, detections: list) -> plt.Figure:
    """Graf kenarlarından ilişki tipine göre renklendirilmiş komşuluk matrisi figürü oluşturur.

    Girdi:  graph — edges listesi içeren UI grafı sözlüğü
            detections — matris boyutu ve eksen etiketleri için tespit listesi
    Çıktı:  matplotlib Figure nesnesi; tespit yoksa "No detections" mesajlı figür
    """
    n = len(detections)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No detections", ha="center", va="center")
        return fig

    edge_types = {"parent_child": 1, "above": 2, "below": 3, "left_of": 4, "right_of": 5}
    matrix = np.zeros((n, n))
    for edge in graph.get("edges", []):
        s, t = edge["source"], edge["target"]
        if s < n and t < n:
            matrix[s][t] = edge_types.get(edge["relation"], 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#FFFFFF", "#2ECC71", "#3498DB", "#9B59B6", "#E74C3C", "#F39C12"]
    cmap = ListedColormap(colors)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=5)

    limit = min(n, 25)
    labels = [f"{d['id']}:{d['class_name'][:5]}" for d in detections[:limit]]
    ax.set_xticks(range(limit))
    ax.set_yticks(range(limit))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Adjacency Matrix", fontsize=13, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(["None", "Parent-Child", "Above", "Below", "Left", "Right"])
    plt.tight_layout()
    return fig


def format_hierarchy(node: dict, indent: int = 0) -> str:
    """Hiyerarşi ağacını girintili çok satırlı metin gösterimine dönüştürür.

    Girdi:  node — hiyerarşi düğümü sözlüğü ('class', 'id', 'children' içerir)
            indent — mevcut girinti seviyesi
    Çıktı:  ağacın girintili metin gösterimi (dizge)
    """
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    line = f"{prefix}{node['class']} (id={node['id']})\n"
    for child in node.get("children", []):
        line += format_hierarchy(child, indent + 1)
    return line


# ── Rendered preview (compiled Compose look) ────────────────
_PREVIEW_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;}
  .legend{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 14px;}
  .chip{display:flex;align-items:center;gap:6px;font-size:11.5px;color:#555;}
  .sw{width:12px;height:12px;border-radius:3px;border:1px solid rgba(0,0,0,.3);}
  .row{display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start;}
  .col{flex:0 0 auto;}
  .col h3{font-size:13px;margin:0 0 4px;font-weight:600;}
  .col .note{font-size:11px;color:#777;margin:0 0 10px;max-width:240px;line-height:1.45;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;overflow:auto;}
  .stack{display:flex;flex-direction:column;align-items:stretch;padding:8px;gap:8px;}
  .leaf{font-size:12px;}
  .l-text{color:#222;padding:4px 6px;}
  .l-img{height:64px;background:repeating-linear-gradient(45deg,#dfe6ee,#dfe6ee 8px,#e9eef4 8px,#e9eef4 16px);border:1px solid #c7d0db;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#6b7686;font-size:11px;}
  .l-btn{align-self:center;background:#6750a4;color:#fff;border:none;border-radius:20px;padding:8px 22px;font-size:12px;font-weight:600;}
  .l-field{border:1px solid #b9c2cf;border-radius:6px;padding:8px;color:#8b95a3;font-size:12px;background:#fff;}
  .l-toggle{display:flex;align-items:center;gap:8px;color:#222;}
  .l-generic{border:1px dashed #b9c2cf;border-radius:6px;padding:10px;font-size:11.5px;text-align:center;color:#6b7686;background:#f1f4f8;}
  .container-box{border:1.5px dashed #c2b8e6;border-radius:8px;padding:8px;display:flex;flex-direction:column;align-items:center;gap:6px;background:#f7f4fe;}
  .cbadge{font-size:9.5px;color:#7a5bd0;align-self:flex-start;font-weight:700;letter-spacing:.3px;}
  .bb{position:absolute;border-radius:3px;border:1.5px solid rgba(0,0,0,.35);overflow:hidden;display:flex;align-items:flex-start;}
  .bb span{background:rgba(255,255,255,.72);padding:0 3px;font-size:8.5px;line-height:1.3;color:#11141a;}
</style></head><body>
<div class="legend" id="legend"></div>
<div class="row">
  <div class="col"><h3>Derlenmiş Compose — Koda Sadık</h3><div class="note">Üretilen .kt kodunun Compose'da derlenince oluşturacağı dikey akış.</div>
    <div class="phone" id="p_stack"><div class="screen" id="s_stack"></div></div></div>
  <div class="col"><h3>Konum-temelli Rekonstrüksiyon</h3><div class="note">Tespit edilen bbox konumlarına göre ekranın yeniden inşası (hedeflenen yerleşim).</div>
    <div class="phone" id="p_bbox"><div class="screen" id="s_bbox"></div></div></div>
</div>
<script>
const DATA = __DATA__; const COLORS = __COLORS__;
const W = 240, H = Math.round(W / DATA.aspect);
for(const id of ["p_stack","p_bbox"]){ const el=document.getElementById(id); el.style.width=W+"px"; el.style.height=H+"px"; }
function leafHTML(cls){
  switch(cls){
    case 'ImageView': return '<div class="leaf l-img">\\uD83D\\uDDBC Image</div>';
    case 'TextView': return '<div class="leaf l-text">Text Content</div>';
    case 'Button': return '<button class="leaf l-btn">Button</button>';
    case 'EditText': return '<div class="leaf l-field">TextField\\u2026</div>';
    case 'CheckBox': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Checkbox</label>';
    case 'Switch': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Switch</label>';
    case 'RadioButton': return '<label class="leaf l-toggle"><input type="radio" checked disabled> RadioButton</label>';
    default: return '<div class="leaf l-generic">'+cls+'</div>';
  }
}
function renderNode(n){
  const k=n.children||[];
  if(k.length){ return '<div class="container-box"><span class="cbadge">Column \\u00B7 '+n.class+'</span>'+k.map(renderNode).join('')+'</div>'; }
  return leafHTML(n.class);
}
document.getElementById("s_stack").innerHTML='<div class="stack">'+DATA.roots.map(renderNode).join('')+'</div>';
document.getElementById("s_bbox").innerHTML=DATA.boxes.map(function(b){
  const c=COLORS[b.cls]||'#bbb';
  const l=((b.cx-b.w/2)*100).toFixed(2),t=((b.cy-b.h/2)*100).toFixed(2),w=(b.w*100).toFixed(2),h=(b.h*100).toFixed(2);
  return '<div class="bb" style="left:'+l+'%;top:'+t+'%;width:'+w+'%;height:'+h+'%;background:'+c+'cc;border-color:'+c+'"><span>'+b.cls+'</span></div>';
}).join('');
const cnt={}; DATA.boxes.forEach(function(b){cnt[b.cls]=(cnt[b.cls]||0)+1;});
document.getElementById("legend").innerHTML=Object.keys(cnt).map(function(c){
  return '<div class="chip"><span class="sw" style="background:'+(COLORS[c]||'#bbb')+'"></span>'+c+' \\u00D7'+cnt[c]+'</div>';
}).join('');
</script></body></html>"""


def build_preview_html(roots: list, detections: list, img_w: int, img_h: int) -> str:
    """Derlenmiş Compose akışı ile bbox rekonstrüksiyonunu yan yana gösteren HTML üretir.

    Girdi:  roots — hiyerarşi kök düğümlerinin listesi
            detections — tespit sözlüklerinin listesi
            img_w — görüntü genişliği (piksel)
            img_h — görüntü yüksekliği (piksel)
    Çıktı:  şablona veriler gömülmüş önizleme HTML dizgesi
    """
    boxes = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append({
            "cls": d["class_name"],
            "cx": ((x1 + x2) / 2) / img_w,
            "cy": ((y1 + y2) / 2) / img_h,
            "w": (x2 - x1) / img_w,
            "h": (y2 - y1) / img_h,
        })
    data = {"roots": roots, "boxes": boxes, "aspect": round(img_w / img_h, 4)}
    return (_PREVIEW_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__COLORS__", json.dumps(CLASS_COLORS)))


# ── Single-phone compiled Compose preview (for Compose Preview tab) ──
_COMPILED_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;display:flex;justify-content:center;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;overflow:auto;}
  .stack{display:flex;flex-direction:column;align-items:stretch;padding:8px;gap:8px;}
  .leaf{font-size:12px;}
  .l-text{color:#222;padding:4px 6px;}
  .l-img{height:64px;background:repeating-linear-gradient(45deg,#dfe6ee,#dfe6ee 8px,#e9eef4 8px,#e9eef4 16px);border:1px solid #c7d0db;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#6b7686;font-size:11px;}
  .l-btn{align-self:center;background:#6750a4;color:#fff;border:none;border-radius:20px;padding:8px 22px;font-size:12px;font-weight:600;}
  .l-field{border:1px solid #b9c2cf;border-radius:6px;padding:8px;color:#8b95a3;font-size:12px;background:#fff;}
  .l-toggle{display:flex;align-items:center;gap:8px;color:#222;}
  .l-generic{border:1px dashed #b9c2cf;border-radius:6px;padding:10px;font-size:11.5px;text-align:center;color:#6b7686;background:#f1f4f8;}
  .container-box{border:1.5px dashed #c2b8e6;border-radius:8px;padding:8px;display:flex;flex-direction:column;align-items:center;gap:6px;background:#f7f4fe;}
  .cbadge{font-size:9.5px;color:#7a5bd0;align-self:flex-start;font-weight:700;letter-spacing:.3px;}
</style></head><body>
<div class="phone" id="phone"><div class="screen" id="screen"></div></div>
<script>
const DATA = __DATA__;
const W = 260, H = Math.round(W / DATA.aspect);
const p=document.getElementById("phone"); p.style.width=W+"px"; p.style.height=H+"px";
function leafHTML(cls){
  switch(cls){
    case 'ImageView': return '<div class="leaf l-img">\\uD83D\\uDDBC Image</div>';
    case 'TextView': return '<div class="leaf l-text">Text Content</div>';
    case 'Button': return '<button class="leaf l-btn">Button</button>';
    case 'EditText': return '<div class="leaf l-field">TextField\\u2026</div>';
    case 'CheckBox': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Checkbox</label>';
    case 'Switch': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Switch</label>';
    case 'RadioButton': return '<label class="leaf l-toggle"><input type="radio" checked disabled> RadioButton</label>';
    default: return '<div class="leaf l-generic">'+cls+'</div>';
  }
}
function renderNode(n){
  const k=n.children||[];
  if(k.length){ return '<div class="container-box"><span class="cbadge">Column \\u00B7 '+n.class+'</span>'+k.map(renderNode).join('')+'</div>'; }
  return leafHTML(n.class);
}
document.getElementById("screen").innerHTML='<div class="stack">'+DATA.roots.map(renderNode).join('')+'</div>';
</script></body></html>"""


def build_compiled_html(roots: list, img_w: int, img_h: int) -> str:
    """Tek telefon çerçeveli derlenmiş Compose görünümü HTML'i üretir.

    Girdi:  roots — hiyerarşi kök düğümlerinin listesi
            img_w — görüntü genişliği (piksel)
            img_h — görüntü yüksekliği (piksel)
    Çıktı:  şablona veriler gömülmüş HTML dizgesi
    """
    data = {"roots": roots, "aspect": round(img_w / img_h, 4)}
    return _COMPILED_TEMPLATE.replace("__DATA__", json.dumps(data))


# ── Synthesized nested Row/Column layout (mirrors the generated code) ──
# Renders the same XY-cut tree the code generator now produces, so the
# preview reflects the actual compiled structure (nested Row/Column/Box).
_SYNTH_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;display:flex;justify-content:center;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;overflow:auto;}
  .screenwrap{padding:6px;}
  .box{display:flex;border-radius:8px;padding:6px;gap:6px;position:relative;}
  .row{flex-direction:row;align-items:center;border:1.4px dashed rgba(79,140,255,.55);background:rgba(79,140,255,.05);}
  .col{flex-direction:column;align-items:stretch;border:1.4px dashed rgba(175,82,222,.45);background:rgba(175,82,222,.04);}
  .badge{position:absolute;top:-7px;left:6px;font-size:8px;font-weight:700;letter-spacing:.3px;padding:0 4px;border-radius:4px;background:#11141a;color:#fff;}
  .cell{display:flex;align-items:center;justify-content:center;}
  .leaf{font-size:11px;}
  .l-text{color:#222;padding:4px 6px;}
  .l-img{height:48px;width:100%;background:repeating-linear-gradient(45deg,#dfe6ee,#dfe6ee 8px,#e9eef4 8px,#e9eef4 16px);border:1px solid #c7d0db;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#6b7686;font-size:10px;}
  .l-btn{background:#6750a4;color:#fff;border:none;border-radius:20px;padding:7px 18px;font-size:11px;font-weight:600;}
  .l-field{border:1px solid #b9c2cf;border-radius:6px;padding:7px;color:#8b95a3;font-size:11px;background:#fff;width:100%;}
  .l-toggle{display:flex;align-items:center;gap:6px;color:#222;}
  .l-generic{border:1px dashed #b9c2cf;border-radius:6px;padding:8px;font-size:10.5px;text-align:center;color:#6b7686;background:#f1f4f8;}
</style></head><body>
<div class="phone" id="phone"><div class="screen" id="screen"></div></div>
<script>
const DATA = __DATA__;
const W = 260, H = Math.round(W / DATA.aspect);
const p=document.getElementById("phone"); p.style.width=W+"px"; p.style.height=H+"px";
function leafHTML(cls){
  switch(cls){
    case 'ImageView': return '<div class="leaf l-img">\\uD83D\\uDDBC Image</div>';
    case 'TextView': return '<div class="leaf l-text">Text Content</div>';
    case 'Button': return '<button class="leaf l-btn">Button</button>';
    case 'EditText': return '<div class="leaf l-field">TextField\\u2026</div>';
    case 'CheckBox': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Checkbox</label>';
    case 'Switch': return '<label class="leaf l-toggle"><input type="checkbox" checked disabled> Switch</label>';
    case 'RadioButton': return '<label class="leaf l-toggle"><input type="radio" checked disabled> RadioButton</label>';
    default: return '<div class="leaf l-generic">'+cls+'</div>';
  }
}
function render(node, inRow){
  const style = inRow ? ' style="flex:1;min-width:0"' : '';
  if(node.type==='leaf'){
    return '<div class="cell"'+style+'>'+leafHTML(node.node.class)+'</div>';
  }
  const isRow = node.type==='Row';
  const inner = (node.children||[]).map(function(c){return render(c,isRow);}).join('');
  const cls = isRow ? 'box row' : 'box col';
  return '<div class="'+cls+'"'+style+'><span class="badge">'+node.type+'</span>'+inner+'</div>';
}
document.getElementById("screen").innerHTML='<div class="screenwrap">'+render(DATA.tree,false)+'</div>';
</script></body></html>"""


def build_synth_html(roots: list, img_w: int, img_h: int) -> str:
    """Kod üreticinin ürettiği XY-cut iç içe Row/Column ağacını HTML olarak çizer.

    Girdi:  roots — hiyerarşi kök düğümlerinin listesi
            img_w — görüntü genişliği (piksel)
            img_h — görüntü yüksekliği (piksel)
    Çıktı:  şablona ağaç verisi gömülmüş HTML dizgesi
    """
    gen = ComposeCodeGenerator()
    leaves = gen._collect_leaves(roots)
    tree = gen._xy_cut(leaves, prefer="y") if leaves else {"type": "Column", "children": []}
    data = {"tree": tree, "aspect": round(img_w / img_h, 4)}
    return _SYNTH_TEMPLATE.replace("__DATA__", json.dumps(data))


# ── Single-phone bbox reconstruction (resembles the screenshot) ──
_BBOX_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#16181d;background:transparent;display:flex;justify-content:center;}
  .phone{position:relative;background:#fff;border:9px solid #11141a;border-radius:24px;overflow:hidden;box-shadow:0 6px 22px rgba(0,0,0,.25);}
  .screen{position:relative;width:100%;height:100%;background:#fafbfc;}
  .bb{position:absolute;border-radius:3px;border:1.5px solid rgba(0,0,0,.35);overflow:hidden;display:flex;align-items:flex-start;}
  .bb span{background:rgba(255,255,255,.72);padding:0 3px;font-size:8.5px;line-height:1.3;color:#11141a;}
</style></head><body>
<div class="phone" id="phone"><div class="screen" id="screen"></div></div>
<script>
const DATA = __DATA__; const COLORS = __COLORS__;
const W = 260, H = Math.round(W / DATA.aspect);
const p=document.getElementById("phone"); p.style.width=W+"px"; p.style.height=H+"px";
document.getElementById("screen").innerHTML=DATA.boxes.map(function(b){
  const c=COLORS[b.cls]||'#bbb';
  const l=((b.cx-b.w/2)*100).toFixed(2),t=((b.cy-b.h/2)*100).toFixed(2),w=(b.w*100).toFixed(2),h=(b.h*100).toFixed(2);
  return '<div class="bb" style="left:'+l+'%;top:'+t+'%;width:'+w+'%;height:'+h+'%;background:'+c+'cc;border-color:'+c+'"><span>'+b.cls+'</span></div>';
}).join('');
</script></body></html>"""


def build_bbox_html(detections: list, img_w: int, img_h: int) -> str:
    """YOLO tespit koordinatlarından ekranın bbox rekonstrüksiyonu HTML'ini üretir.

    Girdi:  detections — tespit sözlüklerinin listesi
            img_w — görüntü genişliği (piksel)
            img_h — görüntü yüksekliği (piksel)
    Çıktı:  şablona kutu verileri gömülmüş HTML dizgesi
    """
    boxes = []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        boxes.append({
            "cls": d["class_name"],
            "cx": ((x1 + x2) / 2) / img_w,
            "cy": ((y1 + y2) / 2) / img_h,
            "w": (x2 - x1) / img_w,
            "h": (y2 - y1) / img_h,
        })
    data = {"boxes": boxes, "aspect": round(img_w / img_h, 4)}
    return (_BBOX_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__COLORS__", json.dumps(CLASS_COLORS)))


# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="UI → Compose Demo",
    page_icon="📱",
    layout="wide",
)

st.title("📱 UI Screenshot → Jetpack Compose")
st.caption("Android ekran görüntüsünden otomatik Compose kodu üretimi")

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Ayarlar")

    models = find_models()
    if not models:
        st.error("Model bulunamadı! `runs/*/weights/best.pt` kontrol edin.")
        st.stop()

    model_labels = [str(m.relative_to(PROJECT_ROOT)) for m in models]
    selected_model_idx = st.selectbox(
        "Model", range(len(models)),
        format_func=lambda i: model_labels[i],
    )
    model_path = str(models[selected_model_idx])

    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

    st.divider()
    st.markdown("**Pipeline akışı:**")
    st.markdown("1. YOLO Detection\n2. Graph Building\n3. Code Generation")

# ── Main area ───────────────────────────────────────────────
uploaded = st.file_uploader("Screenshot yükle", type=["png", "jpg", "jpeg"])

# Also allow picking test images
test_dir = PROJECT_ROOT / "examples"
test_images = sorted(test_dir.glob("*.png")) if test_dir.exists() else []

if not uploaded and test_images:
    st.markdown("**veya test görsellerinden birini seç:**")
    cols = st.columns(len(test_images))
    for i, img_path in enumerate(test_images):
        with cols[i]:
            thumb = Image.open(img_path)
            thumb.thumbnail((150, 300))
            st.image(thumb, caption=img_path.name)
            if st.button(f"Seç", key=f"test_{i}"):
                st.session_state["test_image"] = str(img_path)

# Resolve which image to use
image_path = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(uploaded.read())
    tmp.flush()
    image_path = tmp.name
elif "test_image" in st.session_state:
    image_path = st.session_state["test_image"]

if not image_path or not Path(image_path).exists():
    st.info("Başlamak için bir screenshot yükleyin veya test görsellerinden birini seçin.")
    st.stop()

# ── Run pipeline ────────────────────────────────────────────
img = Image.open(image_path)
img_w, img_h = img.size

run_btn = st.button("Pipeline Çalıştır", type="primary", use_container_width=True)

if run_btn or st.session_state.get("ran"):
    st.session_state["ran"] = True

    # ── Stage 1: Detection ──────────────────────────────────
    with st.status("Stage 1/3: YOLO Detection...", expanded=True) as status:
        dataset_yaml = str(PROJECT_ROOT / "configs" / "dataset.yaml")
        detector = ComponentDetector(model_path, dataset_yaml)
        detections = detector.detect(image_path, conf)

        status.update(label=f"Stage 1: {len(detections)} component bulundu", state="complete")

    # ── Stage 2: Graph ──────────────────────────────────────
    with st.status("Stage 2/3: Graph oluşturuluyor...", expanded=True) as status:
        builder = UIGraphBuilder()
        graph = builder.build_graph(detections)
        n_edges = len(graph["edges"])
        n_roots = len(graph["hierarchy"]["roots"])
        status.update(label=f"Stage 3: {n_edges} edge, {n_roots} root", state="complete")

    # ── Stage 3: Code generation ────────────────────────────
    with st.status("Stage 3/3: Kod üretiliyor...", expanded=True) as status:
        generator = ComposeCodeGenerator()
        code = generator.generate(graph)
        n_lines = len(code.split("\n"))
        status.update(label=f"Stage 3: {n_lines} satır kod üretildi", state="complete")

    # ── Results ─────────────────────────────────────────────
    st.divider()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Components", len(detections))
    c3.metric("Edges", n_edges)
    c4.metric("Code Lines", n_lines)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Detection", "Graph", "Hierarchy", "Compose Code", "Compose Preview"
    ])

    # ── Tab 1: Detection ────────────────────────────────────
    with tab1:
        col_orig, col_det, col_layout = st.columns(3)
        with col_orig:
            st.subheader("Orijinal")
            st.image(img, use_container_width=True)
        with col_det:
            st.subheader("Detections")
            annotated = draw_detections(img, detections)
            st.image(annotated, use_container_width=True)
        with col_layout:
            st.subheader("Layout")
            layout_img = draw_layout_overlay(img, graph)
            st.image(layout_img, use_container_width=True)
            st.caption("🟠 Column  🟢 Row  🔵 Card")

        st.subheader("Class Dağılımı")
        class_dist = Counter(d["class_name"] for d in detections)
        chart_data = {k: v for k, v in sorted(class_dist.items(), key=lambda x: -x[1])}
        st.bar_chart(chart_data)

        with st.expander("Detection JSON"):
            st.json(detections)

    # ── Tab 2: Graph ────────────────────────────────────────
    with tab2:
        col_g, col_a = st.columns(2)
        with col_g:
            st.subheader("Node-Edge Graph")
            fig_graph = plot_graph(graph, detections, img_w, img_h)
            st.pyplot(fig_graph)
            plt.close(fig_graph)
        with col_a:
            st.subheader("Adjacency Matrix")
            fig_adj = plot_adjacency(graph, detections)
            st.pyplot(fig_adj)
            plt.close(fig_adj)

        st.subheader("Edge Dağılımı")
        edge_dist = Counter(e["relation"] for e in graph["edges"])
        st.bar_chart({k: v for k, v in sorted(edge_dist.items(), key=lambda x: -x[1])})

        with st.expander("Graph JSON"):
            st.json(graph)

    # ── Tab 3: Hierarchy ────────────────────────────────────
    with tab3:
        st.subheader("UI Hierarchy Tree")
        for i, root in enumerate(graph["hierarchy"]["roots"]):
            with st.expander(f"Root {i+1}: {root['class']} (id={root['id']})", expanded=i == 0):
                st.code(format_hierarchy(root), language=None)

    # ── Tab 4: Code ─────────────────────────────────────────
    with tab4:
        st.subheader("Jetpack Compose Code")
        st.code(code, language="kotlin")

        st.download_button(
            "GeneratedScreen.kt indir",
            data=code,
            file_name="GeneratedScreen.kt",
            mime="text/plain",
        )

    # ── Tab 5: Compose Preview (original + synthesized layout + code) ───
    with tab5:
        st.subheader("Compose Preview")
        st.caption(
            "Soldan sağa: seçilen ekran görüntüsü, üretilen Jetpack Compose kodunun "
            "derlenince oluşturacağı iç içe Row/Column yerleşimi ve kodun kendisi. "
            "Yerleşim, tespit edilen bileşenlerin uzamsal ilişkilerinden (XY-cut) "
            "sentezlenir; mor kesikli çerçeve Column, mavi kesikli çerçeve Row'dur."
        )
        roots = graph["hierarchy"]["roots"]
        ph_h = max(560, int(260 / (img_w / img_h)) + 60)
        col_orig, col_layout, col_code = st.columns([1, 1, 1.2])
        with col_orig:
            st.markdown("**Seçilen ekran görüntüsü**")
            st.image(img, use_container_width=True)
        with col_layout:
            st.markdown("**Derlenmiş yerleşim (Row/Column)**")
            components.html(build_synth_html(roots, img_w, img_h), height=ph_h)
        with col_code:
            st.markdown("**Üretilen kod — GeneratedScreen.kt**")
            st.code(code, language="kotlin")
            st.download_button(
                "GeneratedScreen.kt indir",
                data=code,
                file_name="GeneratedScreen.kt",
                mime="text/plain",
                key="dl_preview",
            )

        with st.expander("Tespit overlay'i (bbox — bileşenler nerede bulundu)"):
            st.caption(
                "Bu görünüm üretilen koddan DEĞİL, doğrudan YOLO tespit koordinatlarından "
                "çizilir; modelin bileşenleri ekranda nerede bulduğunu gösterir. "
                "Soldaki 'Derlenmiş yerleşim' ise üretilen kodun yapısını yansıtır."
            )
            components.html(build_bbox_html(detections, img_w, img_h), height=ph_h)
