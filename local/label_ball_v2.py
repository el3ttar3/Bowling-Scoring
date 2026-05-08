#!/usr/bin/env python3
"""Ball labeler v2 — Hough Circle detection + temporal trajectory tracking.

Approach:
  1. For each frame run cv2.HoughCircles with tight params on a smoothed grayscale.
  2. Filter circles to keep only "ball-shaped":
     - radius 12..30 px
     - region inside circle is bright (V > 130) and unsaturated (S < 60) — white ball
     - region is NOT on top of a colored cone (HSV mask of red/green/blue does NOT overlap)
  3. Temporal: keep only candidates that form a smooth trajectory (per-frame jump < 200 px).
  4. Output: only frames with HIGH-CONFIDENCE detections get a label. All others empty.
"""
import os, glob, math
import warnings; warnings.filterwarnings("ignore")
import numpy as np, cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
FRAMES = sorted(glob.glob(os.path.join(ROOT, "frames", "*.jpg")))
OUT_LBL = os.path.join(ROOT, "labels")
OUT_PRV = os.path.join(ROOT, "preview_v2")
os.makedirs(OUT_LBL, exist_ok=True); os.makedirs(OUT_PRV, exist_ok=True)

# --- parameters tuned for 960×960 frames ---
MIN_R, MAX_R = 10, 36
HOUGH_PARAM2 = 22           # accumulator threshold — higher = stricter
DP = 1.2
MIN_DIST = 40
MAX_JUMP_PX = 220           # max ball motion between consecutive frames

# Brightness / saturation cutoffs for "ball-like" patches.
V_MIN = 115                 # ball is bright (cream/white)
S_MAX = 70                  # ball is unsaturated

# Pin-color HSV mask (so we don't pick up colored caps as the ball).
def color_mask(hsv):
    masks = [
        cv2.inRange(hsv, np.array([0,110,70], dtype=np.uint8),  np.array([8,255,255], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([172,110,70], dtype=np.uint8),np.array([180,255,255], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([38,70,50], dtype=np.uint8),  np.array([85,255,255], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([95,90,60], dtype=np.uint8),  np.array([130,255,255], dtype=np.uint8)),
    ]
    m = masks[0]
    for k in masks[1:]: m = m | k
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

def candidates_for_frame(bgr):
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cmask = color_mask(hsv)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=DP, minDist=MIN_DIST,
        param1=80, param2=HOUGH_PARAM2, minRadius=MIN_R, maxRadius=MAX_R,
    )
    out = []
    if circles is None: return out
    for x, y, r in np.round(circles[0]).astype(int):
        if x - r < 0 or y - r < 0 or x + r >= W or y + r >= H: continue
        # disk mask of this candidate
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(x), int(y)), int(r), 255, -1)
        if cv2.countNonZero(m) < 30: continue
        # reject if overlaps a colored cone area
        overlap = cv2.bitwise_and(m, cmask)
        if cv2.countNonZero(overlap) / cv2.countNonZero(m) > 0.15: continue
        v_mean = float(hsv[..., 2][m > 0].mean())
        s_mean = float(hsv[..., 1][m > 0].mean())
        # ball is bright + low saturation
        if v_mean < V_MIN or s_mean > S_MAX: continue
        score = (v_mean / 255.0) * (1.0 - s_mean / 100.0) * (1.0 - abs(r - 22) / 30.0)
        out.append((int(x - r), int(y - r), int(x + r), int(y + r), float(score)))
    return out

def run():
    print(f"Frames: {len(FRAMES)}")
    cands = [candidates_for_frame(cv2.imread(p)) for p in FRAMES]
    n_any = sum(1 for c in cands if c)
    print(f"Frames with ANY Hough candidate (after filters): {n_any}")

    # ---- pick best per frame using temporal continuity ----
    track = [None] * len(FRAMES)
    last_xy = None; last_idx = -1
    for i in range(len(FRAMES)):
        if not cands[i]: continue
        if last_xy is None:
            # earliest pick: highest standalone score
            cands[i].sort(key=lambda c: -c[4])
            box = cands[i][0]
        else:
            # pick candidate closest to predicted position
            best = None; best_cost = 1e9
            for c in cands[i]:
                cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2
                d = math.hypot(cx - last_xy[0], cy - last_xy[1])
                if d > MAX_JUMP_PX * max(1, i - last_idx): continue
                cost = d - 50 * c[4]
                if cost < best_cost: best_cost = cost; best = c
            if best is None: continue
            box = best
        track[i] = box
        last_xy = ((box[0]+box[2])/2, (box[1]+box[3])/2); last_idx = i
    print(f"Tracked ball in {sum(1 for t in track if t is not None)}/{len(FRAMES)} frames")

    # ---- write labels (high precision: only labelled frames get a box; others empty) ----
    for i, p in enumerate(FRAMES):
        bn = os.path.splitext(os.path.basename(p))[0]
        out_txt = os.path.join(OUT_LBL, bn + ".txt")
        if track[i] is None:
            open(out_txt, "w").close(); continue
        H, W = cv2.imread(p).shape[:2]
        x1, y1, x2, y2 = track[i][:4]
        cx, cy, w, h = (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H
        with open(out_txt, "w") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # ---- previews ----
    for i in range(0, len(FRAMES), 3):
        bgr = cv2.imread(FRAMES[i])
        if track[i] is not None:
            x1, y1, x2, y2 = track[i][:4]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(bgr, "ball", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(bgr, f"f{i:04d}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.imwrite(os.path.join(OUT_PRV, os.path.basename(FRAMES[i])), bgr)

    print("Done.")

if __name__ == "__main__":
    run()
