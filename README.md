# 🎳 Bowling Scoring

End-to-end Colab pipeline for scoring a bowling-style game (small ball + Schweppes-bottle pins with red/green/blue caps). Trains **YOLO11n** on a single ~30 s gameplay video and ships a **Gradio app** that takes any video and returns:

- annotated MP4 (bboxes, pin labels P1..Pn, drop chip strip with per-pin timing, ball trajectory, top HUD)
- JSON drop log
- plain-English summary

## Authors

- **Abdelrahman Elattar** — ID 202201353
- **Abdelrahman Omar** — ID 202202254
- **Reem Ehab** — ID 202201373

## Resources

| Resource | Link |
|---|---|
| Inputs (video, weights, ball labels) | <https://drive.google.com/drive/folders/1agz6FRHjGQKde5q8tsUOq8VkbaPlnho0?usp=share_link> |
| Trained notebook (with run outputs) | [`notebook/Bowling_Scoring.ipynb`](notebook/Bowling_Scoring.ipynb) |

## Repo layout

```
CV-bowling-proj/
├── README.md
├── notebook/
│   └── Bowling_Scoring.ipynb      # 25 cells — full pipeline + Gradio, run outputs included
└── local/                         # local pre-processing
    ├── label_ball.py              # v1 ball labeler (motion + roundness, kept for reference)
    ├── label_ball_v2.py           # v2 ball labeler — Hough Circles + HSV + temporal continuity (used)
    ├── ball_labels.zip            # 20 high-precision ball labels + 163 empty frames (upload to Drive)
    ├── labels/                    # YOLO-format ball labels: "0 cx cy w h" per frame (.txt)
    └── check_all/all.jpg          # contact sheet of all 20 labeled frames for visual sanity
```

(Frames, model weights, and preview directories are reproducible from the inputs and excluded via `.gitignore`.)

## How to run

1. **Get the inputs** from the Drive folder linked above and place them in your own
   `/MyDrive/Bowling_Scoring/inputs/`:
   - `bowling.mp4` — gameplay video (~30 s)
   - `ball_labels.zip` — pre-made YOLO ball labels (built locally by `label_ball_v2.py`)
   - `best.pt` *(optional)* — previous Attar Scoring weights (pin classes 1/2/3 are identical, transfers for free)

2. **Open** `notebook/Bowling_Scoring.ipynb` in Colab.
   **Runtime → Change runtime type → T4 GPU.** *Don't pick TPU.*

3. **Run all cells**. The notebook will:
   - Mount Drive, verify all three inputs exist
   - Extract frames at 5 fps
   - Auto-label pins via HSV + PCA principal-axis bbox extension (deterministic, works for upright + fallen)
   - Load ball labels from the uploaded zip
   - Build a YOLO dataset with cross-temporal split (70 % train / 15 % val / 15 % test)
   - Train YOLO11n in two phases (frozen backbone → full fine-tune), checkpointing every 5 epochs to Drive
   - Evaluate on the held-out tail
   - Mirror final `best.pt` to `/MyDrive/Bowling_Scoring/release/`
   - Launch a **Gradio app** with a public share link

4. **Open the share link** (printed by the last cell) in any browser → drop a video → get the annotated MP4 + JSON drop log.

## Class IDs

```
0: ball
1: pin_red
2: pin_green
3: pin_blue
```

## What's reused vs. new (vs. the previous Attar Scoring project)

| Component | Status |
|---|---|
| Pin auto-labeler (HSV color + PCA bbox extension) | ✅ reused — works for upright + fallen pins |
| Drop detection (cone-vs-body angle, hysteresis) | ✅ reused |
| Per-pin timing (frame-PTS based) | ✅ reused |
| ByteTrack-lite tracker | ✅ reused, inlined |
| **Ball labels** | 🆕 hand-curated locally (Hough circles + HSV + temporal continuity, then visually verified) |
| Mobile UI / Flutter / IPA | ❌ not in this project — Colab + Gradio only |
| Model weights | ✅ transfer-learn from previous `best.pt` (pin classes 1/2/3 identical) |

## Producing `ball_labels.zip` (already done — kept for repeatability)

```bash
cd local/
# 1. Frames at 5 fps from the gameplay video
ffmpeg -i bowling.mp4 -vf 'fps=5,scale=960:-2' -q:v 3 frames/f_%04d.jpg
# 2. Hough-circle ball labeler with HSV + temporal-continuity filtering (~10 min on Mac CPU)
python3 label_ball_v2.py
# 3. Manually drop a handful of false positives (visual contact sheet), then bundle
cd labels && zip -j ../ball_labels.zip *.txt
```

The labeler combines:
1. `cv2.HoughCircles` (ball is the only round object — pins are tall rectangles, caps are trapezoids).
2. HSV brightness + saturation filter (white ball: V ≥ 115, S ≤ 70).
3. Pin-color overlap rejection (kills cap-edge false positives).
4. Temporal continuity (per-frame jump < 220 px).
5. Hand review using `local/check_all/all.jpg` — drop pre-throw frames and any landed-on-person frames.

Result: **20 high-precision ball labels** spanning the full trajectory (frames 28 → 182), out of 183 total frames. The 163 empty label files are intentional — those are frames where the ball is fully occluded or off-frame.
