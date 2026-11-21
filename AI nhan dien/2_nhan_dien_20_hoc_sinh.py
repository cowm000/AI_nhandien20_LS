import cv2
import numpy as np
import pandas as pd
import os
import time
import joblib
from collections import deque
from rtmlib import Body, draw_skeleton

# ==============================================================================
# CẤU HÌNH HỆ THỐNG GIÁM SÁT 20 HỌC SINH (5 HÀNG x 4 CỘT) - MÔ HÌNH KÉP
# ==============================================================================
POSE_MODEL = "study_model_3p.pkl"
OBJ_MODEL = "object_model_1p.pkl"
COCO_UPPER_BODY = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

STATE_NAMES = ["TAP TRUNG", "NGU GAT", "LO DANG", "LAM VIEC RIENG"]
STATE_COLORS = [(0, 210, 0), (0, 0, 255), (0, 180, 255), (255, 0, 255)]
OBJ_NAMES = ["But/Sach", "D.Thoai", "Casio/Coc", "TayKhong"]

NUM_ROWS = 5
NUM_COLS = 4
MAX_STUDENTS = NUM_ROWS * NUM_COLS  # 20 Học sinh (5 Hàng x 4 Cột)

body_estimator = Body(to_openpose=False, mode='lightweight', backend='onnxruntime', device='cpu')
AI_EVERY_N_FRAMES = 2

pose_clf = joblib.load(POSE_MODEL) if os.path.exists(POSE_MODEL) else None
obj_clf = joblib.load(OBJ_MODEL) if os.path.exists(OBJ_MODEL) else None

def calc_biomech_from_22(feat_22):
    pts = np.array(feat_22, dtype=float).reshape(11, 2)
    nose = pts[0]
    l_sh, r_sh = pts[3], pts[4]
    l_el, r_el = pts[5], pts[6]
    l_wr, r_wr = pts[7], pts[8]

    def angle_3pt(a, b, c):
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.arccos(np.clip(cos_a, -1.0, 1.0)))

    return [
        round(angle_3pt(l_sh, l_el, l_wr), 5),
        round(angle_3pt(r_sh, r_el, r_wr), 5),
        round(float(np.linalg.norm(l_wr - nose)), 5),
        round(float(np.linalg.norm(r_wr - nose)), 5)
    ]

def get_patch_visual_22d(bgr_patch):
    if bgr_patch is None or bgr_patch.size == 0:
        return [0.0] * 22
    patch = cv2.resize(bgr_patch, (48, 48))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    grid_mag = cv2.resize(mag, (4, 2), interpolation=cv2.INTER_AREA).flatten()
    grid_mag = grid_mag / (np.max(grid_mag) + 1e-5)

    ang_half = np.mod(ang, 180.0)
    hist_hog, _ = np.histogram(ang_half, bins=8, range=(0, 180), weights=mag)
    hist_hog = hist_hog / (np.sum(hist_hog) + 1e-5)

    center_hsv = hsv[12:36, 12:36]
    mean_s = float(np.mean(center_hsv[:, :, 1]) / 255.0)
    mean_v = float(np.mean(center_hsv[:, :, 2]) / 255.0)
    std_v = float(np.std(center_hsv[:, :, 2]) / 255.0)

    bright_paper = float(np.mean((hsv[:, :, 2] > 185) & (hsv[:, :, 1] < 55)))
    dark_screen = float(np.mean(hsv[:, :, 2] < 65))
    edge_density = float(np.mean(mag > 0.25))

    feats = list(grid_mag) + list(hist_hog) + [mean_s, mean_v, std_v, bright_paper, dark_screen, edge_density]
    return [round(float(x), 5) for x in feats]

def extract_student_dual_features(clean_frame, kpts_17, scores_17):
    """Trích xuất đồng thời Vector Tư thế 26D (có bù phối cảnh) và Vector Đồ vật 48D cho từng học sinh."""
    h, w, _ = clean_frame.shape
    l_sh, r_sh = kpts_17[5], kpts_17[6]
    sh_width = np.linalg.norm(l_sh - r_sh)
    if sh_width < 11.0:
        return None
    mid_sh = (l_sh + r_sh) / 2.0

    # 1. Vector Tư thế 26 chiều (bù góc chúc camera cho 5 hàng ghế)
    depth_ratio = np.clip(mid_sh[1] / float(h), 0.0, 1.0)
    pitch_rad = np.radians(8.0 + 18.0 * depth_ratio)
    y_comp = 1.0 / np.cos(pitch_rad)

    feat_22 = []
    for idx in COCO_UPPER_BODY:
        dx = (kpts_17[idx][0] - mid_sh[0]) / sh_width
        dy = ((kpts_17[idx][1] - mid_sh[1]) * y_comp) / sh_width
        feat_22.extend([round(float(dx), 5), round(float(dy), 5)])
    feat_26 = feat_22 + calc_biomech_from_22(feat_22)
    w_t = float(np.clip(np.mean(scores_17[COCO_UPPER_BODY]), 0.05, 1.0))

    # 2. Vector Đồ vật 48 chiều quanh bàn tay và mặt bàn của học sinh đó
    l_el, r_el = kpts_17[7], kpts_17[8]
    l_wr, r_wr = kpts_17[9], kpts_17[10]
    c_hand_l = l_wr + 0.35 * (l_wr - l_el)
    c_hand_r = r_wr + 0.35 * (r_wr - r_el)

    score_l = -c_hand_l[1] - 0.5 * abs(c_hand_l[0] - mid_sh[0])
    score_r = -c_hand_r[1] - 0.5 * abs(c_hand_r[0] - mid_sh[0])
    active_c = c_hand_l if score_l >= score_r else c_hand_r

    box_r = int(max(14, sh_width * 0.36))
    hx1 = max(0, min(w - 10, int(active_c[0] - box_r)))
    hy1 = max(0, min(h - 10, int(active_c[1] - box_r)))
    hx2 = max(hx1 + 10, min(w, int(active_c[0] + box_r)))
    hy2 = max(hy1 + 10, min(h, int(active_c[1] + box_r)))
    hand_patch = clean_frame[hy1:hy2, hx1:hx2]

    dx1 = max(0, min(w - 16, int(mid_sh[0] - sh_width * 0.60)))
    dx2 = max(dx1 + 16, min(w, int(mid_sh[0] + sh_width * 0.60)))
    dy1 = max(0, min(h - 16, int(mid_sh[1] + sh_width * 0.35)))
    dy2 = max(dy1 + 16, min(h, int(mid_sh[1] + sh_width * 1.10)))
    desk_patch = clean_frame[dy1:dy2, dx1:dx2]

    f_hand = get_patch_visual_22d(hand_patch)
    f_desk = get_patch_visual_22d(desk_patch)
    norm_act = (active_c - mid_sh) / sh_width
    hands_dist = float(np.linalg.norm(c_hand_l - c_hand_r) / sh_width)
    min_hy = float(min(c_hand_l[1] - mid_sh[1], c_hand_r[1] - mid_sh[1]) / sh_width)
    feat_48 = f_hand + f_desk + [
        round(float(norm_act[0]), 5), round(float(norm_act[1]), 5), round(hands_dist, 5), round(min_hy, 5)
    ]

    return {
        "feat_26": feat_26,
        "feat_48": feat_48,
        "mid_sh": mid_sh,
        "sh_w": sh_width,
        "w_t": w_t,
        "kpts": kpts_17,
        "hand_box": (hx1, hy1, hx2, hy2)
    }

class ConfidenceWeightedBuffer:
    def __init__(self, maxlen=10):
        self.probs_q = deque(maxlen=maxlen)
        self.weights_q = deque(maxlen=maxlen)

    def append(self, probs, weight=1.0):
        self.probs_q.append(probs)
        self.weights_q.append(weight)

    def get_smoothed_probs(self):
        if not self.probs_q:
            return None
        P = np.array(self.probs_q)
        w = np.array(self.weights_q)
        return np.sum(P * w[:, None], axis=0) / (np.sum(w) + 1e-6)

class ClassroomGrid5x4Tracker:
    def __init__(self, rows=5, cols=4):
        self.rows = rows
        self.cols = cols
        self.max_seats = rows * cols
        self.anchors = {}
        self.seat_rc = {}
        self.pose_buffers = {}
        self.obj_buffers = {}
        self.focus_sec = {}
        self.bad_streak = {}

    def calibrate_5x4_grid(self, detections):
        if not detections:
            return
        pts = sorted(detections[:self.max_seats], key=lambda d: d["mid_sh"][1])
        grouped_rows = []
        curr_row = [pts[0]]
        row_thr = max(42.0, np.mean([d["sh_w"] for d in pts]) * 0.82)

        for d in pts[1:]:
            if abs(d["mid_sh"][1] - np.mean([x["mid_sh"][1] for x in curr_row])) <= row_thr and len(curr_row) < self.cols:
                curr_row.append(d)
            else:
                grouped_rows.append(sorted(curr_row, key=lambda item: item["mid_sh"][0]))
                curr_row = [d]
        grouped_rows.append(sorted(curr_row, key=lambda item: item["mid_sh"][0]))

        self.anchors.clear()
        self.seat_rc.clear()
        self.pose_buffers.clear()
        self.obj_buffers.clear()
        self.focus_sec.clear()
        self.bad_streak.clear()

        seat_id = 1
        for r_idx, r_list in enumerate(grouped_rows[:self.rows], start=1):
            for c_idx, d in enumerate(r_list[:self.cols], start=1):
                self.anchors[seat_id] = d["mid_sh"].copy()
                self.seat_rc[seat_id] = (r_idx, c_idx)
                self.pose_buffers[seat_id] = ConfidenceWeightedBuffer(maxlen=10)
                self.obj_buffers[seat_id] = ConfidenceWeightedBuffer(maxlen=10)
                self.focus_sec[seat_id] = 0.0
                self.bad_streak[seat_id] = 0.0
                seat_id += 1

    def match_and_update(self, detections, img_w, img_h):
        if not self.anchors and len(detections) > 0:
            self.calibrate_5x4_grid(detections)

        assigned = {}
        used_seats = set()

        for d in detections:
            mid = d["mid_sh"]
            best_sid = None
            best_dist = 1e9
            max_allow_dist = max(65.0, d["sh_w"] * 1.8)

            for sid, anchor in self.anchors.items():
                if sid in used_seats:
                    continue
                dist = np.linalg.norm(mid - anchor)
                if dist < best_dist and dist < max_allow_dist:
                    best_dist = dist
                    best_sid = sid

            if best_sid is None and len(self.anchors) < self.max_seats:
                best_sid = len(self.anchors) + 1
                est_r = min(self.rows, max(1, int((mid[1] - 50) / ((img_h - 60) / self.rows)) + 1))
                est_c = min(self.cols, max(1, int(mid[0] / (img_w / self.cols)) + 1))
                self.anchors[best_sid] = mid.copy()
                self.seat_rc[best_sid] = (est_r, est_c)
                self.pose_buffers[best_sid] = ConfidenceWeightedBuffer(maxlen=10)
                self.obj_buffers[best_sid] = ConfidenceWeightedBuffer(maxlen=10)
                self.focus_sec[best_sid] = 0.0
                self.bad_streak[best_sid] = 0.0

            if best_sid is not None:
                used_seats.add(best_sid)
                self.anchors[best_sid] = 0.92 * self.anchors[best_sid] + 0.08 * mid
                assigned[best_sid] = d

        return assigned

def fuse_pose_and_object(pred_pose, conf_pose, pred_obj, conf_obj):
    if pred_obj == 1 and conf_obj >= 50.0:
        return 3, max(conf_pose, conf_obj)
    if pred_pose in (1, 2, 3):
        return pred_pose, conf_pose
    return 0, conf_pose

def draw_student_overhead_badge(frame, d, sid, rc_tuple, final_cls, final_conf, obj_txt, is_alert):
    """Hiển thị cả Trạng thái + Đồ vật phát hiện được ngay trên đầu từng học sinh."""
    h, w, _ = frame.shape
    kpts = d["kpts"]
    sh_w = d["sh_w"]
    color = STATE_COLORS[final_cls]

    # Vẽ khung vuông nhỏ quanh bàn tay đang cầm đồ vật của học sinh
    hx1, hy1, hx2, hy2 = d["hand_box"]
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 215, 255), 1)

    head_x = int(kpts[0][0])
    top_y = min(kpts[0][1], kpts[3][1], kpts[4][1])
    head_y = int(top_y - max(20.0, sh_w * 0.34))

    box_w, box_h = 162, 42
    x1 = max(6, min(w - box_w - 6, head_x - box_w // 2))
    y1 = max(52, min(h - box_h - 12, head_y - box_h))
    x2, y2 = x1 + box_w, y1 + box_h

    tri = np.array([[head_x - 6, y2], [head_x + 6, y2], [head_x, y2 + 8]], np.int32)
    cv2.fillPoly(frame, [tri], color)

    bg_color = (0, 0, 145) if is_alert else (20, 20, 20)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    r_idx, c_idx = rc_tuple
    line_top = f"HS{sid:02d}(H{r_idx}C{c_idx})|Vat:{obj_txt}"
    line_bot = f"{STATE_NAMES[final_cls]} ({final_conf:.0f}%)"

    cv2.putText(frame, line_top, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1)
    cv2.putText(frame, line_bot, (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)

tracker = ClassroomGrid5x4Tracker(rows=NUM_ROWS, cols=NUM_COLS)
show_grid_lines = True

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_time = time.time()
frame_count = 0
cached_kpts = []
cached_scores = []
fps_display = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = max(1e-4, now - last_time)
    last_time = now
    fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    clean_frame = frame.copy()
    h, w, _ = frame.shape
    frame_count += 1

    if show_grid_lines:
        col_w = w / float(NUM_COLS)
        row_h = (h - 50) / float(NUM_ROWS)
        for c in range(1, NUM_COLS):
            cx = int(c * col_w)
            cv2.line(frame, (cx, 50), (cx, h), (90, 90, 90), 1, cv2.LINE_AA)
        for r in range(1, NUM_ROWS):
            ry = int(50 + r * row_h)
            cv2.line(frame, (0, ry), (w, ry), (90, 90, 90), 1, cv2.LINE_AA)

    is_ai_frame = (frame_count % AI_EVERY_N_FRAMES == 0)
    if is_ai_frame:
        scale = 800.0 / w
        small_frame = cv2.resize(clean_frame, (800, int(h * scale)))
        kpts_s, cached_scores = body_estimator(small_frame)
        cached_kpts = kpts_s / scale if len(kpts_s) > 0 else []

    keypoints, scores = cached_kpts, cached_scores
    detections = []
    for kpts, sc in zip(keypoints, scores):
        item = extract_student_dual_features(clean_frame, kpts, sc)
        if item is not None:
            detections.append(item)

    if len(keypoints) > 0:
        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.30)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        tracker.calibrate_5x4_grid(detections)
    elif key == ord('g'):
        show_grid_lines = not show_grid_lines
    elif key == ord('r'):
        for sid in tracker.focus_sec:
            tracker.focus_sec[sid] = 0.0
            tracker.bad_streak[sid] = 0.0
    elif key == ord('q'):
        break

    assigned = tracker.match_and_update(detections, w, h)
    active_sids = sorted(assigned.keys())

    # Suy luận lô Kép (Dual Batch Inference: N x 26 cho Tư thế & N x 48 cho Đồ vật)
    if is_ai_frame and len(active_sids) > 0:
        if pose_clf is not None:
            X_pose_batch = np.array([assigned[sid]["feat_26"] for sid in active_sids])
            p_batch_probs = pose_clf.predict_proba(X_pose_batch)
            for sid, probs in zip(active_sids, p_batch_probs):
                tracker.pose_buffers[sid].append(probs, assigned[sid]["w_t"])

        if obj_clf is not None:
            X_obj_batch = np.array([assigned[sid]["feat_48"] for sid in active_sids])
            o_batch_probs = obj_clf.predict_proba(X_obj_batch)
            for sid, probs in zip(active_sids, o_batch_probs):
                tracker.obj_buffers[sid].append(probs, 1.0)

    state_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    alerts = []

    for sid in active_sids:
        d = assigned[sid]
        p_buf = tracker.pose_buffers[sid]
        o_buf = tracker.obj_buffers[sid]

        pred_pose, conf_pose = 0, 0.0
        pred_obj, conf_obj = 3, 0.0

        if pose_clf is not None and len(p_buf.probs_q) > 0:
            avg_p = p_buf.get_smoothed_probs()
            pred_pose = int(np.argmax(avg_p))
            conf_pose = avg_p[pred_pose] * 100

        if obj_clf is not None and len(o_buf.probs_q) > 0:
            avg_o = o_buf.get_smoothed_probs()
            pred_obj = int(np.argmax(avg_o))
            conf_obj = avg_o[pred_obj] * 100

        final_cls, final_conf = fuse_pose_and_object(pred_pose, conf_pose, pred_obj, conf_obj)
        state_counts[final_cls] += 1
        obj_txt = OBJ_NAMES[pred_obj] if obj_clf is not None else "--"

        if final_cls == 0:
            tracker.focus_sec[sid] += dt
            tracker.bad_streak[sid] = max(0.0, tracker.bad_streak[sid] - dt * 2)
        else:
            tracker.bad_streak[sid] += dt

        is_alert = tracker.bad_streak[sid] >= 3.0
        r_i, c_i = tracker.seat_rc.get(sid, (1, 1))
        if is_alert:
            alerts.append(f"HS{sid:02d}(H{r_i}C{c_i}):{STATE_NAMES[final_cls]}[{obj_txt}]")

        draw_student_overhead_badge(frame, d, sid, (r_i, c_i), final_cls, final_conf, obj_txt, is_alert)

    present = len(active_sids)
    header = (f"CODE 2: 20 HS (5x4) + DO VAT | Si so:{present}/20 | Tap trung:{state_counts[0]} | "
              f"Ngu gat:{state_counts[1]} | Lo dang:{state_counts[2]} | Viec rieng:{state_counts[3]} | "
              f"FPS:{fps_display:.0f} | 'm':Khoa ghe | 'g':Luoi")
    cv2.rectangle(frame, (0, 0), (w, 48), (15, 15, 15), -1)
    cv2.putText(frame, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

    if alerts:
        alert_bar = "CANH BAO (>3s): " + " | ".join(alerts[:4])
        cv2.rectangle(frame, (0, h - 38), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, alert_bar, (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    cv2.imshow("Code 2: Nhan Dien 20 Hoc Sinh (5 Hang x 4 Cot) - Tu The 26D + Do Vat 48D", frame)

cap.release()
cv2.destroyAllWindows()