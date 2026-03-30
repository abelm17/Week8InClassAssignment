"""
Classroom Furniture 2D Mapper via Stereo (Two-Photo) Vision
============================================================
Uses two photos taken from different angles with a single phone camera
to reconstruct 2D floor positions of tables and chairs.

NO MEASUREMENT TOOLS REQUIRED — the baseline is estimated automatically
from the scene using one of three methods (tried in priority order):

  Method A — Floor homography + assumed camera height
             Finds the floor plane across both images via feature
             homography decomposition, then uses your phone's height
             above the floor (~1.5 m) to set the metric scale.

  Method B — Bounding-box height + known furniture dimensions
             Uses the apparent pixel height of detected chairs/tables
             together with their known real-world heights to compute
             per-object depth, which anchors the scale.

  Method C — Scene depth heuristic (fallback)
             Assumes the median scene depth is ~3 m (typical classroom
             photo distance) and scales accordingly.

Pipeline:
  1. Load two images
  2. SIFT feature matching
  3. Essential matrix → unit camera pose (R, t)
  4. Furniture detection via YOLOv8 (falls back to mock data)
  5. Auto-estimate metric scale (Methods A → B → C)
  6. Triangulate furniture centroids → 3D world points
  7. Project to floor plane → 2D X-Y plot

Requirements:
    pip install opencv-python opencv-contrib-python numpy matplotlib
    pip install ultralytics   # YOLO (optional but strongly recommended)
"""

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Try importing YOLOv8 ─────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — mock detections will be used.")
    print("       pip install ultralytics")

# ── COCO class IDs ───────────────────────────────────────────────────────────
CHAIR_IDS     = {56}
TABLE_IDS     = {60}
FURNITURE_IDS = CHAIR_IDS | TABLE_IDS

# ── Real-world height priors (metres) ─────────────────────────────────────────
# Standard furniture heights used by Method B
REAL_HEIGHT = {
    "chair": 0.90,   # seat + back ≈ 90 cm
    "table": 0.75,   # desk/table  ≈ 75 cm
}

# Default assumed phone camera height above floor (metres)
# ~chest height; pass --cam-height to override
CAMERA_HEIGHT_M = 1.50


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Camera intrinsics
# ═════════════════════════════════════════════════════════════════════════════

def estimate_intrinsics(image_shape, fov_deg: float = 70.0) -> np.ndarray:
    """
    Build approximate K from image dimensions + assumed horizontal FOV.
    Typical modern phone cameras: 60–75°.
    """
    h, w = image_shape[:2]
    fx   = (w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx,  0, cx],
                     [ 0, fx, cy],
                     [ 0,  0,  1]], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Feature matching + pose recovery
# ═════════════════════════════════════════════════════════════════════════════

def match_features(img1: np.ndarray, img2: np.ndarray, max_matches: int = 500):
    """SIFT + Lowe ratio test → matched pixel coords in both images."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, d1 = sift.detectAndCompute(g1, None)
    kp2, d2 = sift.detectAndCompute(g2, None)

    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Too few SIFT features — use more textured images.")

    raw  = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    print(f"  SIFT: {len(kp1)} / {len(kp2)} kpts  →  {len(good)} good matches")

    if len(good) < 8:
        raise RuntimeError(f"Only {len(good)} good matches. Increase photo overlap.")

    good = good[:max_matches]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def recover_pose(pts1, pts2, K):
    """Essential matrix (RANSAC) → R, unit-t, inlier mask."""
    E, mask = cv2.findEssentialMat(pts1, pts2, K,
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")
    inliers = mask.ravel().astype(bool)
    print(f"  Pose inliers: {inliers.sum()} / {len(inliers)}")
    _, R, t, _ = cv2.recoverPose(E, pts1[inliers], pts2[inliers], K)
    return R, t, inliers   # t is a UNIT vector


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Auto baseline / scale estimators
# ═════════════════════════════════════════════════════════════════════════════

def _triangulate_unit(pts_l, pts_r, K, R, t_unit):
    """Triangulate with the raw unit translation (up to unknown scale)."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t_unit])
    h4 = cv2.triangulatePoints(P1, P2,
                                pts_l.T.astype(np.float64),
                                pts_r.T.astype(np.float64))
    return (h4[:3] / h4[3]).T          # (N, 3)


def _find_correspondence(pt_l, pts2_in, pts1_in, radius=60.0):
    """Nearest-neighbour lookup: find right-image match for a left-image point."""
    dists = np.linalg.norm(pts1_in - pt_l, axis=1)
    idx   = np.argmin(dists)
    return pts2_in[idx] if dists[idx] < radius else pt_l + np.array([10., 0.])


# ── Method A ─────────────────────────────────────────────────────────────────
def _scale_floor_homography(pts1_in, pts2_in, K, img_h):
    """
    Fit homography on floor-region matches, decompose to get plane distance,
    then set  scale = CAMERA_HEIGHT_M / d_plane.
    """
    floor_mask = (pts1_in[:, 1] > 0.6 * img_h) & \
                 (pts2_in[:, 1] > 0.6 * img_h)
    fp1 = pts1_in[floor_mask]
    fp2 = pts2_in[floor_mask]

    if len(fp1) < 10:
        print("  [A] Too few floor-region matches.")
        return None

    H, _ = cv2.findHomography(fp1, fp2, cv2.RANSAC, 3.0)
    if H is None:
        return None

    num, _, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    if num == 0:
        return None

    # Pick decomposition where the normal is roughly vertical
    best_d = None
    for i in range(num):
        n = Ns[i].ravel()
        if abs(n[1]) > 0.5:
            d = float(np.linalg.norm(Ts[i]))
            if d > 1e-6:
                best_d = d
                break
    if best_d is None:
        best_d = float(np.linalg.norm(Ts[0])) if len(Ts) else None
    if not best_d or best_d < 1e-6:
        return None

    scale = CAMERA_HEIGHT_M / best_d
    print(f"  [A] Floor homography d={best_d:.4f} → scale={scale:.4f} m")
    return scale if 0.05 < scale < 20.0 else None


# ── Method B ─────────────────────────────────────────────────────────────────
def _scale_bbox_height(boxes_l, pts1_in, pts2_in, K, R, t_unit):
    """
    For each detected object compute:
        Z_real  = f_y * H_real / H_pixels      (perspective formula)
        Z_unit  = triangulated depth with unit-t
        scale   = Z_real / Z_unit
    Return median scale across all objects.
    """
    fy = K[1, 1]
    scales = []
    for cx, cy, label, h_px in boxes_l:
        if h_px < 10:
            continue
        H_real = REAL_HEIGHT.get(label, 0.85)
        Z_real = (fy * H_real) / h_px

        pt_l = np.array([[cx, cy]], dtype=np.float32)
        pt_r_vec = _find_correspondence(np.array([cx, cy]), pts2_in, pts1_in)
        pt_r = np.array([[pt_r_vec[0], pt_r_vec[1]]], dtype=np.float32)

        pts3d = _triangulate_unit(pt_l, pt_r, K, R, t_unit)
        Z_unit = pts3d[0, 2]
        if Z_unit > 1e-3:
            s = Z_real / Z_unit
            scales.append(s)
            print(f"  [B] {label:6s} h_px={h_px:.0f} Z_real={Z_real:.2f}m "
                  f"Z_unit={Z_unit:.4f} → s={s:.4f}")

    if not scales:
        return None
    scale = float(np.median(scales))
    print(f"  [B] Bounding-box scale = {scale:.4f} m  (median of {len(scales)})")
    return scale if 0.05 < scale < 20.0 else None


# ── Method C ─────────────────────────────────────────────────────────────────
def _scale_heuristic(pts1_in, pts2_in, K, R, t_unit,
                     assumed_depth_m: float = 3.0):
    """Assume the median feature depth equals assumed_depth_m."""
    pts3d = _triangulate_unit(pts1_in, pts2_in, K, R, t_unit)
    valid = pts3d[(pts3d[:, 2] > 0.01) & (pts3d[:, 2] < 1e4)]
    if len(valid) == 0:
        return 1.0
    med = float(np.median(valid[:, 2]))
    scale = assumed_depth_m / med
    print(f"  [C] Heuristic: median unit-depth={med:.4f} → scale={scale:.4f} m")
    return scale


# ── Dispatcher ────────────────────────────────────────────────────────────────
def auto_estimate_scale(pts1_in, pts2_in, K, R, t_unit,
                         img_l, boxes_l):
    """Try A → B → C, return (scale, method_name)."""
    print("\n  ── Auto-estimating metric scale ──")

    s = _scale_floor_homography(pts1_in, pts2_in, K, img_l.shape[0])
    if s is not None:
        return s, "floor homography + camera height"

    s = _scale_bbox_height(boxes_l, pts1_in, pts2_in, K, R, t_unit)
    if s is not None:
        return s, "bounding-box height + furniture priors"

    s = _scale_heuristic(pts1_in, pts2_in, K, R, t_unit)
    return s, "scene depth heuristic (3 m assumed)"


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Object detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_furniture_yolo(model, img, conf=0.30):
    """Returns (detections, boxes_full).
    detections  : [(cx, cy, label), ...]
    boxes_full  : [(cx, cy, label, h_pixels), ...]   ← used by Method B
    """
    results = model(img, verbose=False)[0]
    detections, boxes_full = [], []
    for box in results.boxes:
        cid = int(box.cls[0])
        if cid not in FURNITURE_IDS or float(box.conf[0]) < conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx    = (x1 + x2) / 2.0
        cy    = y1 + (y2 - y1) * 0.75   # foot-point
        h_px  = y2 - y1
        label = "chair" if cid in CHAIR_IDS else "table"
        detections.append((cx, cy, label))
        boxes_full.append((cx, cy, label, h_px))
        print(f"    {label:6s}  conf={float(box.conf[0]):.2f}  "
              f"px=({cx:.0f},{cy:.0f})  h={h_px:.0f}px")
    return detections, boxes_full


def mock_detections(img_shape, seed=0):
    rng = np.random.default_rng(seed)
    h, w = img_shape[:2]
    dets, boxes = [], []
    for _ in range(4):
        cx = rng.uniform(0.15*w, 0.85*w)
        cy = rng.uniform(0.30*h, 0.80*h)
        hp = rng.uniform(80, 140)
        dets.append((cx, cy, "table"));  boxes.append((cx, cy, "table", hp))
    for _ in range(8):
        cx = rng.uniform(0.15*w, 0.85*w)
        cy = rng.uniform(0.30*h, 0.90*h)
        hp = rng.uniform(60, 100)
        dets.append((cx, cy, "chair"));  boxes.append((cx, cy, "chair", hp))
    return dets, boxes


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Triangulation
# ═════════════════════════════════════════════════════════════════════════════

def build_proj(K, R, t):
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    return P1, P2


def triangulate(pts_l, pts_r, P1, P2):
    h4 = cv2.triangulatePoints(P1, P2,
                                pts_l.T.astype(np.float64),
                                pts_r.T.astype(np.float64))
    return (h4[:3] / h4[3]).T


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_floor_map(floor_pts, labels, scale_method="", output="classroom_map.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    style = {
        "table": dict(c="#e74c3c", marker="s", s=180),
        "chair": dict(c="#3498db", marker="o", s=80),
    }

    for lbl in ("table", "chair"):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        if not idx:
            continue
        xs = [floor_pts[i, 0] for i in idx]
        zs = [floor_pts[i, 1] for i in idx]
        ax.scatter(xs, zs, **style[lbl],
                   alpha=0.90, edgecolors="white", linewidths=0.6, zorder=3)
        for k, (x, z) in enumerate(zip(xs, zs)):
            ax.text(x, z, f" {lbl[0].upper()}{k+1}",
                    color="white", fontsize=7, va="center", zorder=4)

    ax.grid(color="#2c3e6e", ls="--", lw=0.5, alpha=0.6)
    ax.set_xlabel("X  (metres, approx.)", color="#aab4d4", fontsize=11)
    ax.set_ylabel("Z  (metres, approx.)", color="#aab4d4", fontsize=11)
    title = "Classroom Floor Map  —  Stereo Reconstruction"
    if scale_method:
        title += f"\nScale: {scale_method}"
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#aab4d4")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2c3e6e")

    handles = [
        mpatches.Patch(color="#e74c3c",
                       label=f"Tables ({sum(1 for l in labels if l=='table')})"),
        mpatches.Patch(color="#3498db",
                       label=f"Chairs ({sum(1 for l in labels if l=='chair')})"),
    ]
    ax.legend(handles=handles, facecolor="#0f3460", edgecolor="#aab4d4",
              labelcolor="white", fontsize=10)
    ax.annotate("↑  Camera forward direction",
                xy=(0.02, 0.97), xycoords="axes fraction",
                color="#aab4d4", fontsize=8, va="top")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Map saved → {output}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Main pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(left_path, right_path,
                 fov_deg=70.0, baseline_m=None,
                 output="classroom_map.png", use_mock=False):

    print("\n══════════  Classroom Stereo Mapper  ══════════")
    auto_mode = baseline_m is None
    print(f"  Scale mode: {'AUTO (no measuring needed)' if auto_mode else f'{baseline_m} m (user-provided)'}")

    # 1. Load ─────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading images …")
    img_l = cv2.imread(left_path)
    img_r = cv2.imread(right_path)
    if img_l is None: raise FileNotFoundError(f"Cannot open: {left_path}")
    if img_r is None: raise FileNotFoundError(f"Cannot open: {right_path}")

    if img_l.shape != img_r.shape:
        h = min(img_l.shape[0], img_r.shape[0])
        w = min(img_l.shape[1], img_r.shape[1])
        img_l = cv2.resize(img_l, (w, h))
        img_r = cv2.resize(img_r, (w, h))
    print(f"  Images: {img_l.shape[1]}×{img_l.shape[0]}")

    K = estimate_intrinsics(img_l.shape, fov_deg)
    print(f"  K: fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # 2. Features ─────────────────────────────────────────────────────────────
    print("\n[2/6] Matching features …")
    pts1, pts2 = match_features(img_l, img_r)

    # 3. Pose ─────────────────────────────────────────────────────────────────
    print("\n[3/6] Recovering camera pose …")
    R, t_unit, inliers = recover_pose(pts1, pts2, K)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    rot_deg = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    print(f"  Rotation ≈ {rot_deg:.1f}°   "
          f"Unit-t = {t_unit.ravel().round(3)}")

    # 4. Detect furniture ─────────────────────────────────────────────────────
    print("\n[4/6] Detecting furniture …")
    if not use_mock and YOLO_AVAILABLE:
        model  = YOLO("yolov8n.pt")
        dets_l, boxes_l = detect_furniture_yolo(model, img_l)
        dets_r, _       = detect_furniture_yolo(model, img_r)
    else:
        print("  Using mock detections.")
        dets_l, boxes_l = mock_detections(img_l.shape, seed=42)
        dets_r, _       = mock_detections(img_r.shape, seed=43)

    if not dets_l:
        print("  [WARN] No detections — falling back to mock")
        dets_l, boxes_l = mock_detections(img_l.shape, seed=42)

    # 5. Scale ────────────────────────────────────────────────────────────────
    print("\n[5/6] Estimating metric scale …")
    if auto_mode:
        scale, scale_method = auto_estimate_scale(
            pts1_in, pts2_in, K, R, t_unit, img_l, boxes_l)
        print(f"\n  ✓ Scale = {scale:.4f}  (method: {scale_method})")
        print(f"    Inferred baseline ≈ {scale * float(np.linalg.norm(t_unit)):.3f} m")
    else:
        scale        = baseline_m / max(float(np.linalg.norm(t_unit)), 1e-9)
        scale_method = "user-provided baseline"

    t_scaled = t_unit * scale
    P1, P2   = build_proj(K, R, t_scaled)

    # 6. Triangulate + plot ───────────────────────────────────────────────────
    print("\n[6/6] Triangulating furniture positions …")
    floor_pts, labels = [], []

    for cx_l, cy_l, label in dets_l:
        pt_l = np.array([cx_l, cy_l])
        pt_r = _find_correspondence(pt_l, pts2_in, pts1_in)
        pts3d = triangulate(np.array([pt_l]), np.array([pt_r]), P1, P2)
        fp    = pts3d[0, [0, 2]]   # X, Z  (drop Y)
        floor_pts.append(fp)
        labels.append(label)
        print(f"  {label:6s}  ({cx_l:6.1f},{cy_l:6.1f})px  "
              f"→  X={fp[0]:+.2f} m  Z={fp[1]:+.2f} m")

    floor_pts = np.array(floor_pts)
    print(f"\n  ✓ {sum(1 for l in labels if l=='table')} table(s)  "
          f"{sum(1 for l in labels if l=='chair')} chair(s)")

    plot_floor_map(floor_pts, labels, scale_method=scale_method, output=output)
    return floor_pts, labels


# ═════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Classroom furniture mapper — baseline auto-estimated, no measuring needed")
    p.add_argument("left",  nargs="?", default=None, help="First photo path")
    p.add_argument("right", nargs="?", default=None, help="Second photo path")
    p.add_argument("--fov",        type=float, default=70.0,
                   help="Phone camera horizontal FOV in degrees (default: 70)")
    p.add_argument("--baseline",   type=float, default=None,
                   help="Known baseline in metres — OMIT to auto-estimate")
    p.add_argument("--cam-height", type=float, default=CAMERA_HEIGHT_M,
                   help=f"Camera height above floor in metres (default: {CAMERA_HEIGHT_M})")
    p.add_argument("--output",     default="classroom_map.png")
    p.add_argument("--mock",       action="store_true",
                   help="Force mock detections (skip YOLO)")
    args = p.parse_args()

    CAMERA_HEIGHT_M = args.cam_height   # apply override

    if args.left is None or args.right is None:
        print("\n[DEMO MODE] — no images provided, running self-test.\n")
        print("Real usage:")
        print("  python classroom_mapper.py photo1.jpg photo2.jpg")
        print("  python classroom_mapper.py photo1.jpg photo2.jpg --fov 65")
        print("  python classroom_mapper.py photo1.jpg photo2.jpg --baseline 0.4\n")

        H, W = 720, 1280
        def _synth(shift=0, seed=0):
            rng = np.random.default_rng(seed)
            img = np.full((H, W, 3), (40,30,20), dtype=np.uint8)
            for r in range(0, H, 60):
                for c in range(0, W, 80):
                    col = (80,70,60) if (r//60+c//80)%2==0 else (55,45,38)
                    img[r:r+60, c:c+80] = col
            for tx, ty in [(200,300),(500,300),(200,500),(500,500)]:
                cv2.rectangle(img,(tx+shift-60,ty-40),(tx+shift+60,ty+40),(100,60,30),-1)
                cv2.rectangle(img,(tx+shift-60,ty-40),(tx+shift+60,ty+40),(140,90,50), 2)
            for cx2, cy2 in [(160,250),(240,250),(460,250),(540,250),
                             (160,560),(240,560),(460,560),(540,560)]:
                cv2.rectangle(img,(cx2+shift-20,cy2-20),(cx2+shift+20,cy2+20),(180,150,100),-1)
            noise = rng.integers(-15,15,img.shape,dtype=np.int16)
            return np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

        cv2.imwrite("/tmp/demo_L.jpg", _synth(shift=0,  seed=0))
        cv2.imwrite("/tmp/demo_R.jpg", _synth(shift=30, seed=1))
        run_pipeline("/tmp/demo_L.jpg", "/tmp/demo_R.jpg",
                     fov_deg=70.0, baseline_m=None,
                     output=args.output, use_mock=True)
    else:
        run_pipeline(args.left, args.right,
                     fov_deg=args.fov, baseline_m=args.baseline,
                     output=args.output, use_mock=args.mock)