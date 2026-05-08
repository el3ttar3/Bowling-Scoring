#!/usr/bin/env python3
"""Label the ball across the bowling video.

Hybrid pipeline (no GPU needed for the motion path; YOLO uses CPU here):
  1. COCO YOLO11x sports_ball at low conf — anchor labels.
  2. cv2.BackgroundSubtractorMOG2 with a long history — motion mask.
  3. Per frame: extract round, small, bright candidates from motion mask.
  4. Combine all candidates; pick one per frame using:
       - YOLO confidence (highest weight),
       - Roundness * brightness (motion candidate),
       - Distance to predicted position from previous frame (temporal continuity).
  5. Linear interpolation across short gaps (≤ 6 frames).
  6. Outlier rejection via velocity sanity (drop candidates with > 250 px/frame jumps).

Outputs:
  labels/<frame_basename>.txt   YOLO format: "0 cx cy w h" (just ball, class 0).
  preview/<frame_basename>.jpg  annotated for visual sanity check.
"""
import os, sys, glob, json, math
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMES = sorted(glob.glob(os.path.join(ROOT, "frames", "*.jpg")))
LBL_DIR = os.path.join(ROOT, "labels")
PREV_DIR = os.path.join(ROOT, "preview")
os.makedirs(LBL_DIR, exist_ok=True)
os.makedirs(PREV_DIR, exist_ok=True)

assert FRAMES, "No frames found. Re-run ffmpeg to populate ./frames/."
H0, W0 = cv2.imread(FRAMES[0]).shape[:2]
print(f"Frames: {len(FRAMES)}  size {W0}x{H0}")

# ---------- 1) YOLO anchors ----------
print("Loading YOLO11x for ball anchor labels (CPU; ~3 sec/frame, ~10 min total)...")
from ultralytics import YOLO
y = YOLO("yolo11x.pt")
yolo_dets = []  # list per frame: [(x1,y1,x2,y2,conf), ...]
from time import time
t0 = time()
for i, p in enumerate(FRAMES):
    bgr = cv2.imread(p)
    res = y.predict(bgr, imgsz=1280, conf=0.05, classes=[32], verbose=False, device="cpu")[0]
    cands = []
    if res.boxes is not None and len(res.boxes) > 0:
        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()
        for b, c in zip(boxes, confs):
            cands.append((int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(c)))
    yolo_dets.append(cands)
    if (i + 1) % 20 == 0:
        print(f"  YOLO {i+1}/{len(FRAMES)}  {time()-t0:.1f}s")
n_yolo = sum(1 for d in yolo_dets if d)
print(f"YOLO seeded {n_yolo}/{len(FRAMES)} frames")

# ---------- 2) Motion mask over the whole sequence ----------
print("Computing motion masks (MOG2)...")
bg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=20, detectShadows=False)
masks = []
for p in FRAMES:
    masks.append(bg.apply(cv2.imread(p)))

# ---------- 3) Round bright motion candidates ----------
def motion_cands(bgr, mask):
    H, W = bgr.shape[:2]
    img_area = H * W
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < img_area * 8e-5 or area > img_area * 8e-3:  # ball-ish size
            continue
        ar = max(ww, hh) / max(1, min(ww, hh))
        if ar > 2.0:
            continue
        roundness = area / max(1, ww * hh)
        if roundness < 0.55:
            continue
        # mean V (brightness) inside the blob — ball is whitish
        crop = hsv[y:y+hh, x:x+ww]
        comp = (labels[y:y+hh, x:x+ww] == i)
        if comp.sum() < 10:
            continue
        v_mean = float(crop[..., 2][comp].mean())
        s_mean = float(crop[..., 1][comp].mean())
        if v_mean < 110:
            continue
        score = roundness * (v_mean / 255.0) * (1.0 - min(s_mean, 100) / 200.0)
        out.append((x, y, x + ww, y + hh, score))
    return out

motion_per = [motion_cands(cv2.imread(p), masks[i]) for i, p in enumerate(FRAMES)]

# ---------- 4) Combine + temporal selection ----------
def cx_cy(b): return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

def pick_best(yolo, motion, predicted_xy):
    cands = []
    for b in yolo:
        cands.append((b, b[4] * 1.5, "yolo"))   # weight YOLO conf x 1.5
    for b in motion:
        cands.append((b, b[4],       "motion"))
    if not cands:
        return None
    if predicted_xy is None:
        # earliest pick: highest score
        cands.sort(key=lambda c: -c[1])
        return cands[0]
    px, py = predicted_xy
    def cost(c):
        bx, by = cx_cy(c[0])
        d = math.hypot(bx - px, by - py)
        # large d penalises; cap at 250 (anything beyond is implausible per frame)
        return d - 100 * c[1]   # higher score reduces cost
    cands.sort(key=cost)
    return cands[0]

ball_track = [None] * len(FRAMES)
last_xy = None; last_idx = -1
for i in range(len(FRAMES)):
    if last_xy is not None and (i - last_idx) <= 4:
        # extrapolate from previous if any
        pred = last_xy
    else:
        pred = None
    pick = pick_best(yolo_dets[i], motion_per[i], pred)
    if pick is None:
        continue
    box, score, src = pick
    bx, by = cx_cy(box)
    if last_xy is not None and math.hypot(bx - last_xy[0], by - last_xy[1]) > 250 * max(1, i - last_idx):
        # too far from expected — likely a noise blob. Skip this frame.
        continue
    ball_track[i] = box
    last_xy = (bx, by); last_idx = i

n_pre = sum(1 for b in ball_track if b is not None)
print(f"Ball labeled in {n_pre}/{len(FRAMES)} frames before interpolation")

# ---------- 5) Interpolate across short gaps ----------
def interp(box_a, box_b, alpha):
    return tuple(int(a + (b - a) * alpha) for a, b in zip(box_a[:4], box_b[:4])) + (0.4,)

last_seen = -1; last_box = None
for i in range(len(FRAMES)):
    if ball_track[i] is not None:
        if last_seen >= 0 and i - last_seen > 1 and (i - last_seen) <= 6:
            # interpolate between last_seen and i
            for k in range(last_seen + 1, i):
                a = (k - last_seen) / (i - last_seen)
                ball_track[k] = interp(last_box, ball_track[i], a)
        last_seen = i; last_box = ball_track[i]

n_post = sum(1 for b in ball_track if b is not None)
print(f"Ball labeled in {n_post}/{len(FRAMES)} frames after interpolation")

# ---------- 6) Write YOLO labels (ball ONLY; pins are auto-labeled in the notebook) ----------
for i, p in enumerate(FRAMES):
    bn = os.path.splitext(os.path.basename(p))[0]
    out_txt = os.path.join(LBL_DIR, bn + ".txt")
    if ball_track[i] is None:
        # write empty file (no ball this frame)
        open(out_txt, "w").close()
        continue
    x1, y1, x2, y2 = ball_track[i][:4]
    H, W = cv2.imread(p).shape[:2]
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H
    with open(out_txt, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

# ---------- 7) Render previews every 5 frames ----------
for i in range(0, len(FRAMES), 5):
    bgr = cv2.imread(FRAMES[i])
    if ball_track[i] is not None:
        x1, y1, x2, y2 = ball_track[i][:4]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(bgr, "ball", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(bgr, f"f{i:04d}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(PREV_DIR, os.path.basename(FRAMES[i])), bgr)

print(f"Done. Labels in {LBL_DIR}, previews in {PREV_DIR}")
