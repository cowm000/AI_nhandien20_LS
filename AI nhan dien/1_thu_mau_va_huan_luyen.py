import cv2
import numpy as np
import pandas as pd
import csv
import os
import time
import joblib
from collections import deque
from rtmlib import Body, draw_skeleton
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# ==============================================================================
# 1. CẤU HÌNH TÀI NGUYÊN MÔ HÌNH KÉP (TƯ THẾ 26D + ĐỒ VẬT HỌC SINH 48D)
# ==============================================================================
POSE_CSV = "study_dataset_3p.csv"
POSE_MODEL = "study_model_3p.pkl"

OBJ_CSV = "object_dataset_1p.csv"
OBJ_MODEL = "object_model_1p.pkl"

COCO_UPPER_BODY = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
BIOMECH_COLS = ["ang_el_l", "ang_el_r", "dist_wr_nose_l", "dist_wr_nose_r"]

# 4 Trạng thái hành vi chính (Phím 0, 1, 2, 3)
STATE_NAMES = ["TAP TRUNG", "NGU GAT", "LO DANG", "LAM VIEC RIENG"]
STATE_COLORS = [(0, 210, 0), (0, 0, 255), (0, 180, 255), (255, 0, 255)]
POSE_KEYS = {ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3}

# 4 Nhóm đồ vật của học sinh trên tay / mặt bàn (Phím 4, 5, 6, 7)
OBJ_NAMES = ["But/Sach", "D.Thoai", "Casio/Coc", "TayKhong"]
OBJ_FULL_NAMES = ["BUT VIET & SACH VO", "DIEN THOAI", "MAY TINH CASIO / COC", "TAY KHONG / BAN TRONG"]
OBJ_KEYS = {ord('4'): 0, ord('5'): 1, ord('6'): 2, ord('7'): 3}

body_estimator = Body(to_openpose=False, mode='lightweight', backend='onnxruntime', device='cpu')
AI_EVERY_N_FRAMES = 2

def init_pose_csv():
    coord_cols = [f"{c}_{idx}" for idx in COCO_UPPER_BODY for c in ("x", "y")]
    with open(POSE_CSV, mode="w", newline="") as f:
        csv.writer(f).writerow(["label"] + coord_cols + BIOMECH_COLS)

def init_obj_csv():
    obj_cols = [f"v_{i}" for i in range(48)]
    with open(OBJ_CSV, mode="w", newline="") as f:
        csv.writer(f).writerow(["label"] + obj_cols)

if not os.path.exists(OBJ_CSV):
    init_obj_csv()

# ==============================================================================
# 2. TRÍCH XUẤT VECTOR TƯ THẾ 26 CHIỀU & VECTOR ĐỒ VẬT 48 CHIỀU
# ==============================================================================
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

def extract_pose_26d(kpts_17, scores_17):
    l_sh, r_sh = kpts_17[5], kpts_17[6]
    sh_width = np.linalg.norm(l_sh - r_sh)
    if sh_width < 12.0:
        return None, 0.0
    mid_sh = (l_sh + r_sh) / 2.0

    feat_22 = []
    for idx in COCO_UPPER_BODY:
        norm_pt = (kpts_17[idx] - mid_sh) / sh_width
        feat_22.extend([round(float(norm_pt[0]), 5), round(float(norm_pt[1]), 5)])

    feat_26 = feat_22 + calc_biomech_from_22(feat_22)
    w_t = float(np.clip(np.mean(scores_17[COCO_UPPER_BODY]), 0.05, 1.0))
    return feat_26, w_t

def get_patch_visual_22d(bgr_patch):
    """Trích xuất 22 đặc trưng hình thái (Sobel Grid + Mini-HOG 8 hướng + HSV) từ ô ảnh."""
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

def extract_object_48d(clean_frame, kpts_17):
    """Định vị vùng Bàn tay cầm đồ vật & vùng Mặt bàn theo tỷ lệ vai (tương thích mọi khoảng cách)."""
    h, w, _ = clean_frame.shape
    l_sh, r_sh = kpts_17[5], kpts_17[6]
    l_el, r_el = kpts_17[7], kpts_17[8]
    l_wr, r_wr = kpts_17[9], kpts_17[10]

    sh_width = np.linalg.norm(l_sh - r_sh)
    if sh_width < 12.0:
        return None, None
    mid_sh = (l_sh + r_sh) / 2.0

    # Ngoại suy tâm lòng bàn tay từ Khuỷu tay -> Cổ tay
    c_hand_l = l_wr + 0.35 * (l_wr - l_el)
    c_hand_r = r_wr + 0.35 * (r_wr - r_el)

    score_l = -c_hand_l[1] - 0.5 * abs(c_hand_l[0] - mid_sh[0])
    score_r = -c_hand_r[1] - 0.5 * abs(c_hand_r[0] - mid_sh[0])
    active_c = c_hand_l if score_l >= score_r else c_hand_r

    box_r = int(max(16, sh_width * 0.36))
    hx1 = max(0, min(w - 12, int(active_c[0] - box_r)))
    hy1 = max(0, min(h - 12, int(active_c[1] - box_r)))
    hx2 = max(hx1 + 12, min(w, int(active_c[0] + box_r)))
    hy2 = max(hy1 + 12, min(h, int(active_c[1] + box_r)))
    hand_patch = clean_frame[hy1:hy2, hx1:hx2]

    dx1 = max(0, min(w - 20, int(mid_sh[0] - sh_width * 0.60)))
    dx2 = max(dx1 + 20, min(w, int(mid_sh[0] + sh_width * 0.60)))
    dy1 = max(0, min(h - 20, int(mid_sh[1] + sh_width * 0.35)))
    dy2 = max(dy1 + 20, min(h, int(mid_sh[1] + sh_width * 1.10)))
    desk_patch = clean_frame[dy1:dy2, dx1:dx2]

    f_hand = get_patch_visual_22d(hand_patch)
    f_desk = get_patch_visual_22d(desk_patch)

    norm_act = (active_c - mid_sh) / sh_width
    hands_dist = float(np.linalg.norm(c_hand_l - c_hand_r) / sh_width)
    min_hy = float(min(c_hand_l[1] - mid_sh[1], c_hand_r[1] - mid_sh[1]) / sh_width)
    f_ctx = [round(float(norm_act[0]), 5), round(float(norm_act[1]), 5), round(hands_dist, 5), round(min_hy, 5)]

    return f_hand + f_desk + f_ctx, {"hand_box": (hx1, hy1, hx2, hy2), "desk_box": (dx1, dy1, dx2, dy2)}

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

# ==============================================================================
# 3. HUẤN LUYỆN ĐỒNG THỜI CẢ 2 MÔ HÌNH AI (TƯ THẾ + ĐỒ VẬT)
# ==============================================================================
pose_clf = None
obj_clf = None

def train_dual_models():
    """Huấn luyện cả Mô hình Tư thế (26D) và Mô hình Đồ vật (48D) khi nhấn phím 't'."""
    global pose_clf, obj_clf
    if os.path.exists(POSE_CSV):
        df_p = pd.read_csv(POSE_CSV)
        if len(df_p) >= 40 and df_p["label"].nunique() >= 2:
            X = df_p.drop("label", axis=1).values
            y = df_p["label"].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            rf = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=4, random_state=42)
            hgb = HistGradientBoostingClassifier(max_iter=150, max_depth=10, random_state=42)
            m_p = VotingClassifier(estimators=[('rf', rf), ('hgb', hgb)], voting='soft')
            m_p.fit(X_tr, y_tr)
            acc_p = accuracy_score(y_te, m_p.predict(X_te)) * 100
            joblib.dump(m_p, POSE_MODEL)
            pose_clf = m_p
            print(f"[MODEL 1 - TU THE 26D] Da luu '{POSE_MODEL}' | Do chinh xac: {acc_p:.2f}%")

    if os.path.exists(OBJ_CSV):
        df_o = pd.read_csv(OBJ_CSV)
        if len(df_o) >= 40 and df_o["label"].nunique() >= 2:
            X = df_o.drop("label", axis=1).values
            y = df_o["label"].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            rf_o = RandomForestClassifier(n_estimators=200, max_depth=14, min_samples_split=4, random_state=42)
            hgb_o = HistGradientBoostingClassifier(max_iter=150, max_depth=10, random_state=42)
            m_o = VotingClassifier(estimators=[('rf', rf_o), ('hgb', hgb_o)], voting='soft')
            m_o.fit(X_tr, y_tr)
            acc_o = accuracy_score(y_te, m_o.predict(X_te)) * 100
            joblib.dump(m_o, OBJ_MODEL)
            obj_clf = m_o
            print(f"[MODEL 2 - DO VAT 48D] Da luu '{OBJ_MODEL}' | Do chinh xac: {acc_o:.2f}%")
        else:
            print("[CHU Y] Model Do vat can >= 40 mau (Bam phim 4, 5, 6, 7 de thu mau do vat!).")

def startup_check():
    global pose_clf, obj_clf
    if not os.path.exists(POSE_CSV):
        init_pose_csv()
    else:
        df = pd.read_csv(POSE_CSV)
        if len(df) > 0 and df.shape[1] == 23:
            X_22 = df.drop("label", axis=1).values
            bio_df = pd.DataFrame([calc_biomech_from_22(r) for r in X_22], columns=BIOMECH_COLS)
            pd.concat([df, bio_df], axis=1).to_csv(POSE_CSV, index=False)
            train_dual_models()
    if os.path.exists(POSE_MODEL):
        m = joblib.load(POSE_MODEL)
        if getattr(m, "n_features_in_", 0) == 26:
            pose_clf = m
    if os.path.exists(OBJ_MODEL):
        m = joblib.load(OBJ_MODEL)
        if getattr(m, "n_features_in_", 0) == 48:
            obj_clf = m

startup_check()

def get_counts(csv_path):
    c = {0: 0, 1: 0, 2: 0, 3: 0}
    if os.path.exists(csv_path):
        try:
            df_c = pd.read_csv(csv_path)
            for k, v in df_c["label"].value_counts().items():
                c[int(k)] = int(v)
        except Exception:
            pass
    return c

def fuse_pose_and_object(pred_pose, conf_pose, pred_obj, conf_obj):
    """Hợp nhất kết quả Tư thế (26D) + Đồ vật (48D) thành 4 trạng thái chuẩn."""
    if pred_obj == 1 and conf_obj >= 50.0:
        # Phát hiện cầm Điện thoại -> Chuyển thẳng sang LÀM VIỆC RIÊNG kể cả khi ngồi thẳng
        return 3, max(conf_pose, conf_obj)
    if pred_pose in (1, 2, 3):
        return pred_pose, conf_pose
    return 0, conf_pose

def draw_head_badge(frame, kpts, sh_w, top_line, bottom_line, color):
    h, w, _ = frame.shape
    head_x = int(kpts[0][0])
    top_y = min(kpts[0][1], kpts[3][1], kpts[4][1])
    head_y = int(top_y - max(25.0, sh_w * 0.36))

    box_w, box_h = 205, 48
    x1 = max(10, min(w - box_w - 10, head_x - box_w // 2))
    y1 = max(62, min(h - box_h - 15, head_y - box_h))
    x2, y2 = x1 + box_w, y1 + box_h

    tri = np.array([[head_x - 8, y2], [head_x + 8, y2], [head_x, y2 + 10]], np.int32)
    cv2.fillPoly(frame, [tri], color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, top_line, (x1 + 8, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (0, 230, 255), 1)
    cv2.putText(frame, bottom_line, (x1 + 8, y1 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

pose_counts = get_counts(POSE_CSV)
obj_counts = get_counts(OBJ_CSV)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

pose_buf = ConfidenceWeightedBuffer(maxlen=10)
obj_buf = ConfidenceWeightedBuffer(maxlen=10)

record_type = None      # "POSE" (Phím 0-3) hoặc "OBJ" (Phím 4-7)
record_label = None
countdown_start = 0.0
samples_left = 0
SAMPLES_PER_BATCH = 100

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

    is_ai_frame = (frame_count % AI_EVERY_N_FRAMES == 0)
    if is_ai_frame:
        scale = 640.0 / w
        small_frame = cv2.resize(clean_frame, (640, int(h * scale)))
        kpts_s, cached_scores = body_estimator(small_frame)
        cached_kpts = kpts_s / scale if len(kpts_s) > 0 else []

    keypoints, scores = cached_kpts, cached_scores
    key = cv2.waitKey(1) & 0xFF

    if key in POSE_KEYS and record_type is None:
        record_type = "POSE"
        record_label = POSE_KEYS[key]
        countdown_start = now
        samples_left = SAMPLES_PER_BATCH
    elif key in OBJ_KEYS and record_type is None:
        record_type = "OBJ"
        record_label = OBJ_KEYS[key]
        countdown_start = now
        samples_left = SAMPLES_PER_BATCH

    if len(keypoints) > 0:
        main_idx = int(np.argmax([np.linalg.norm(k[5] - k[6]) for k in keypoints]))
        kpts = keypoints[main_idx]
        sc = scores[main_idx]
        sh_w = float(np.linalg.norm(kpts[5] - kpts[6]))

        feat_26, w_t = extract_pose_26d(kpts, sc)
        feat_48, roi_boxes = extract_object_48d(clean_frame, kpts)

        frame = draw_skeleton(frame, keypoints[main_idx:main_idx+1], scores[main_idx:main_idx+1], kpt_thr=0.35)

        if roi_boxes is not None:
            hx1, hy1, hx2, hy2 = roi_boxes["hand_box"]
            dx1, dy1, dx2, dy2 = roi_boxes["desk_box"]
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 215, 255), 2)
            cv2.putText(frame, "ROI BAN TAY", (hx1, max(70, hy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1)
            cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 190, 0), 1)

        # 1. Chế độ Thu mẫu đếm ngược 3 giây (Tư thế 0-3 hoặc Đồ vật 4-7)
        if record_type is not None and feat_26 is not None and feat_48 is not None:
            t_name = STATE_NAMES[record_label] if record_type == "POSE" else OBJ_FULL_NAMES[record_label]
            elapsed = now - countdown_start
            if elapsed < 3.0:
                sec_left = 3 - int(elapsed)
                cv2.rectangle(frame, (w // 2 - 320, h // 2 - 45), (w // 2 + 320, h // 2 + 35), (0, 140, 255), -1)
                cv2.putText(frame, f"CHUAN BI [{record_type}: {t_name}] TRONG: {sec_left}s...",
                            (w // 2 - 300, h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                if is_ai_frame:
                    if record_type == "POSE":
                        with open(POSE_CSV, mode="a", newline="") as f:
                            csv.writer(f).writerow([record_label] + feat_26)
                        pose_counts[record_label] += 1
                    else:
                        with open(OBJ_CSV, mode="a", newline="") as f:
                            csv.writer(f).writerow([record_label] + feat_48)
                        obj_counts[record_label] += 1
                    samples_left -= 1

                draw_head_badge(frame, kpts, sh_w, f"DANG THU {record_type} (Con {samples_left})", t_name[:18], (0, 215, 255))
                if samples_left <= 0:
                    record_type = None

        # 2. Chế độ Kiểm thử Mô hình Kép trực tiếp trên đầu học sinh
        elif feat_26 is not None and feat_48 is not None:
            pred_pose, conf_pose = 0, 0.0
            pred_obj, conf_obj = 3, 0.0

            if pose_clf is not None:
                if is_ai_frame:
                    p_probs = pose_clf.predict_proba(np.array(feat_26).reshape(1, -1))[0]
                    pose_buf.append(p_probs, w_t)
                avg_p = pose_buf.get_smoothed_probs()
                if avg_p is not None:
                    pred_pose = int(np.argmax(avg_p))
                    conf_pose = avg_p[pred_pose] * 100

            if obj_clf is not None:
                if is_ai_frame:
                    o_probs = obj_clf.predict_proba(np.array(feat_48).reshape(1, -1))[0]
                    obj_buf.append(o_probs, 1.0)
                avg_o = obj_buf.get_smoothed_probs()
                if avg_o is not None:
                    pred_obj = int(np.argmax(avg_o))
                    conf_obj = avg_o[pred_obj] * 100

            final_cls, final_conf = fuse_pose_and_object(pred_pose, conf_pose, pred_obj, conf_obj)
            obj_label_str = OBJ_NAMES[pred_obj] if obj_clf is not None else "ChuaTrain"

            draw_head_badge(
                frame, kpts, sh_w,
                f"Vat: {obj_label_str} | w_t:{w_t:.2f}",
                f"{STATE_NAMES[final_cls]} ({final_conf:.0f}%)",
                STATE_COLORS[final_cls]
            )

    if key == ord('t'):
        train_dual_models()
    elif key == ord('c'):
        init_obj_csv()
        obj_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        print("[DA XOA] Da lam sach file CSV Do vat (object_dataset_1p.csv)!")
    elif key == ord('q'):
        break

    line1 = (f"CODE 1 (THU MAU KEP) | FPS:{fps_display:.0f} || Phim 0-3 (Tu the 26D): "
             f"0(TapTrung):{pose_counts[0]} 1(NguGat):{pose_counts[1]} 2(LoDang):{pose_counts[2]} 3(ViecRieng):{pose_counts[3]}")
    line2 = (f"Phim 4-7 (Do vat 48D): 4(But/Sach):{obj_counts[0]} 5(D.Thoai):{obj_counts[1]} "
             f"6(Casio/Coc):{obj_counts[2]} 7(TayKhong):{obj_counts[3]} || 't':Train ca 2 | 'c':Xoa Do vat")
    cv2.rectangle(frame, (0, 0), (w, 56), (15, 15, 15), -1)
    cv2.putText(frame, line1, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, line2, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    cv2.imshow("Code 1: Thu Mau & Huan Luyen Mo Hinh Kep (Tu The 26D + Do Vat 48D)", frame)

cap.release()
cv2.destroyAllWindows()