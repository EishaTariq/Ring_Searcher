"""
Ring Visual Search — CLIP version
===================================
CLIP (ViT-B-32) use karta hai jo jewelry ko properly samajhta hai.
ResNet se kaafi better results milte hain.
"""

import os, cv2, pickle, numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# ── CLIP model ────────────────────────────────────────────────────────────────
_clip_model = None

def _load_clip():
    global _clip_model
    if _clip_model is not None:
        return _clip_model
    try:
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        model.eval()
        _clip_model = (model, preprocess)
        print("✅ CLIP model loaded")
        return _clip_model
    except Exception as e:
        print(f"⚠️  CLIP failed: {e}")
        return None


# ── stone mask ────────────────────────────────────────────────────────────────

def stone_mask(bgr):
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    for lo, hi in [
        (np.array([100,60,60]),  np.array([140,255,255])),
        (np.array([0,120,70]),   np.array([10,255,255])),
        (np.array([170,120,70]),np.array([180,255,255])),
        (np.array([35,60,60]),   np.array([85,255,255])),
        (np.array([125,60,60]),  np.array([155,255,255])),
        (np.array([0,0,210]),    np.array([180,25,255])),
    ]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


# ── feature extractors ────────────────────────────────────────────────────────

def clip_embed(pil_img):
    """CLIP embedding of a PIL image."""
    import torch
    result = _load_clip()
    if result is None:
        return np.array([], dtype=np.float32)
    model, preprocess = result
    tensor = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten().astype(np.float32)


def feat_edge(bgr):
    """Canny edge histogram on top 65% of ring (design area)."""
    h = bgr.shape[0]
    crop  = bgr[:int(h*0.65), :]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 30, 90)
    hist, _ = np.histogram(edges, bins=64, range=(0,256))
    hist = hist.astype(np.float32)
    if hist.sum() > 0: hist /= hist.sum()
    return hist


def feat_metal_color(bgr):
    """Metal color histogram (stone pixels masked out)."""
    smask  = stone_mask(bgr)
    mmask  = cv2.bitwise_not(smask)
    hsv    = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv],[0],mmask,[32],[0,180]).flatten()
    s_hist = cv2.calcHist([hsv],[1],mmask,[32],[0,256]).flatten()
    v_hist = cv2.calcHist([hsv],[2],mmask,[32],[0,256]).flatten()
    feat   = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    if feat.sum() > 0: feat /= feat.sum()
    return feat


def extract_features(bgr):
    from PIL import Image

    img = cv2.resize(bgr, (512, 512))

    # CLIP on full image
    pil_full  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    clip_full = clip_embed(pil_full)  # 512-d

    # CLIP on top 65% (design/stone area)
    h = img.shape[0]
    top_crop  = img[:int(h*0.65), :]
    pil_top   = Image.fromarray(cv2.cvtColor(top_crop, cv2.COLOR_BGR2RGB))
    clip_top  = clip_embed(pil_top)   # 512-d

    # Edge + metal color
    edge  = feat_edge(img)        # 64-d
    metal = feat_metal_color(img) # 96-d

    if len(clip_full) > 0:
        # CLIP full 50% + CLIP top 30% + edge 15% + metal 5%
        feat = np.concatenate([
            clip_full * 0.50,
            clip_top  * 0.30,
            edge      * 0.15,
            metal     * 0.05,
        ])
    else:
        # fallback: no CLIP
        feat = np.concatenate([edge, metal])

    norm = np.linalg.norm(feat)
    if norm > 0: feat /= norm
    return feat.astype(np.float32)


def quality_score(bgr):
    if bgr is None: return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── main class ────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

class RingVisualSearch:

    def __init__(self, catalog_path="catalog", db_path="ring_features.pkl"):
        self.catalog_path = Path(catalog_path)
        self.db_path  = db_path
        self.images   = []
        self.features = []
        self.metadata = []
        self.catalog_path.mkdir(exist_ok=True)

    def build_catalog_index(self):
        files = [p for p in self.catalog_path.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if not files:
            print(f"⚠️  No images in {self.catalog_path}"); return

        # load CLIP once before indexing
        _load_clip()

        print(f"📸 Indexing {len(files)} images…")
        self.images, self.features, self.metadata = [], [], []
        for i, p in enumerate(files, 1):
            try:
                bgr  = cv2.imread(str(p))
                if bgr is None: raise ValueError("unreadable")
                feat = extract_features(bgr)
                self.images.append(str(p))
                self.features.append(feat)
                self.metadata.append({"filename":p.name,"path":str(p),"index":i-1})
                print(f"  [{i}/{len(files)}] {p.name}")
            except Exception as e:
                print(f"  ❌ {p.name}: {e}")

        with open(self.db_path,"wb") as f:
            pickle.dump({"images":self.images,"features":self.features,"metadata":self.metadata},f)
        print(f"✅ Done. {len(self.features)} rings indexed.")

    def _load_features_db(self):
        if not os.path.exists(self.db_path): return False
        try:
            with open(self.db_path,"rb") as f:
                db = pickle.load(f)
            self.images, self.features, self.metadata = db["images"], db["features"], db["metadata"]
            print(f"✅ Loaded {len(self.features)} rings from DB")
            return True
        except Exception as e:
            print(f"⚠️  DB load failed: {e}"); return False

    def search(self, query_path, top_k=10, similarity_threshold=85.0):
        if not self.features:
            if not self._load_features_db(): return []

        bgr = cv2.imread(str(query_path))
        if bgr is None: raise ValueError(f"Cannot read: {query_path}")

        qfeat      = extract_features(bgr).reshape(1,-1)
        cat_matrix = np.vstack(self.features)
        sims       = cosine_similarity(qfeat, cat_matrix)[0]

        results = sorted([
            {"index":idx, "match_percentage":float(sims[idx]*100),
             "image_path":self.images[idx], "metadata":self.metadata[idx]}
            for idx in range(len(self.features))
        ], key=lambda x: x["match_percentage"], reverse=True)

        results = self._deduplicate(results, similarity_threshold)
        return results[:top_k]

    def _deduplicate(self, results, threshold):
        used, unique = set(), []
        for i, res in enumerate(results):
            if i in used: continue
            group = [res]
            fi = self.features[res["index"]].reshape(1,-1)
            for j, other in enumerate(results[i+1:], i+1):
                if j in used: continue
                fj = self.features[other["index"]].reshape(1,-1)
                if float(cosine_similarity(fi,fj)[0][0])*100 >= threshold:
                    group.append(other); used.add(j)
            try:
                best = max(group, key=lambda r: quality_score(cv2.imread(r["image_path"])))
            except Exception:
                best = group[0]
            best["duplicate_count"] = len(group)
            unique.append(best)
            used.add(i)
        return unique

    def get_customization_options(self, ring_index):
        return {
            "ring_index":  ring_index,
            "stone_types": ["Diamond","Sapphire","Ruby","Emerald","Amethyst","Topaz","Aquamarine","Citrine","Garnet","Peridot"],
            "stone_sizes": ["0.25ct","0.5ct","0.75ct","1.0ct","1.5ct","2.0ct","2.5ct","3.0ct"],
            "positions":   ["Center","Side","Halo","Band","Accent"],
            "base_design": self.metadata[ring_index] if ring_index < len(self.metadata) else None,
        }

if __name__ == "__main__":
    engine = RingVisualSearch()
    if not engine._load_features_db():
        engine.build_catalog_index()
    print("Ready!")