# -*- coding: utf-8 -*-
"""
ComfyUI-FaceBeauty —— 综合美颜精修节点（支持多人脸）
- 磨皮/美白/锐化：只作用于皮肤遮罩（FaceOval ∩ 肤色 ∩ 非眼非嘴），不影响头发/背景
- 瘦脸/大眼/微笑/牙齿美白：mediapipe 478 关键点局部形变（cv2.remap 高斯衰减）
- 自动摆正头部：双眼连线水平矫正（带方向自检）
- 头部左右转/上下仰：人脸区域透视形变（2D 近似，局部混合不伤背景）
- 眼球左右/上下：眼周局部位移
"""

import os
import math
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "models", "face_landmarker.task")

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=5,
            running_mode=vision.RunningMode.IMAGE)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def get_landmarks(bgr):
    """返回 [ (N,2) float32, ... ] 所有检出人脸（最多5张）；无人脸返回 []"""
    if bgr is None or bgr.ndim != 3:
        return []
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return []
    h, w = bgr.shape[:2]
    return [np.array([(p.x * w, p.y * h) for p in face], dtype=np.float32)
            for face in result.face_landmarks]


# ---------------- 关键点索引 ----------------
NOSE_TIP = 1
LEFT_CHEEK, RIGHT_CHEEK = 234, 454
LEFT_MOUTH, RIGHT_MOUTH = 61, 291
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
             365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE_POLY = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
                 159, 160, 161, 246]
RIGHT_EYE_POLY = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
                  386, 385, 384, 398]
MOUTH_POLY = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
              318, 402, 317, 14, 87, 178, 88, 95]
INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
             308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


# ---------------- 局部形变核心 ----------------
def _warp(bgr, center, radius, shift_fn):
    """局部形变：map(p) = p - v*falloff，v 为视觉位移向量"""
    h, w = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = float(max(radius, 1.0))
    ys, xs = np.indices((h, w), dtype=np.float32)
    dx, dy = xs - cx, ys - cy
    d = np.sqrt(dx * dx + dy * dy)
    falloff = np.exp(-(d / r) ** 2 * 2.5)
    vx, vy = shift_fn(dx, dy, d)
    vx = np.broadcast_to(np.asarray(vx, np.float32), d.shape)
    vy = np.broadcast_to(np.asarray(vy, np.float32), d.shape)
    return cv2.remap(bgr, xs - vx * falloff, ys - vy * falloff,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _eye_centers(pts):
    return ((pts[33] + pts[133]) / 2.0, (pts[362] + pts[263]) / 2.0)


# ---------------- 各项精修（faces 为多张脸的关键点列表） ----------------
def slim_face(bgr, faces, amount):
    if amount <= 0:
        return bgr
    out = bgr
    for pts in faces:
        lc, rc, nose = pts[LEFT_CHEEK], pts[RIGHT_CHEEK], pts[NOSE_TIP]
        face_w = float(np.linalg.norm(lc - rc))
        if face_w <= 1:
            continue
        radius = 0.40 * face_w
        shift = amount / 100.0 * 0.055 * face_w
        sign_l = 1.0 if nose[0] >= lc[0] else -1.0
        out = _warp(out, lc, radius, lambda dx, dy, d: (np.full_like(d, sign_l * shift), np.zeros_like(d)))
        out = _warp(out, rc, radius, lambda dx, dy, d: (np.full_like(d, -sign_l * shift), np.zeros_like(d)))
    return out


def enlarge_eyes(bgr, faces, amount):
    if amount <= 0:
        return bgr
    out = bgr
    for pts in faces:
        lc, rc = _eye_centers(pts)
        lw = float(np.linalg.norm(pts[33] - pts[133]))
        rw = float(np.linalg.norm(pts[362] - pts[263]))
        if lw <= 1 or rw <= 1:
            continue
        k = amount / 100.0 * 0.07
        out = _warp(out, lc, 1.7 * lw, lambda dx, dy, d: (dx * k, dy * k))
        out = _warp(out, rc, 1.7 * rw, lambda dx, dy, d: (dx * k, dy * k))
    return out


def smile_morph(bgr, faces, amount):
    if amount <= 0:
        return bgr
    out = bgr
    for pts in faces:
        l, r = pts[LEFT_MOUTH], pts[RIGHT_MOUTH]
        mw = float(np.linalg.norm(l - r))
        if mw <= 1:
            continue
        radius = 0.9 * mw
        mag = amount / 100.0 * 0.085 * mw
        inv = mag / math.sqrt(2.0)
        out = _warp(out, l, radius, lambda dx, dy, d: (np.full_like(d, -inv), np.full_like(d, -inv)))
        out = _warp(out, r, radius, lambda dx, dy, d: (np.full_like(d, inv), np.full_like(d, -inv)))
    return out


def whiten_teeth(bgr, faces, amount):
    if amount <= 0:
        return bgr
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    ok = False
    for pts in faces:
        if len(pts) <= max(INNER_LIP):
            continue
        cv2.fillPoly(mask, [pts[INNER_LIP].astype(np.int32)], 255)
        ok = True
    if not ok:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, 105), (180, 90, 255))
    teeth = cv2.bitwise_and(mask, bright)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    teeth = cv2.GaussianBlur(teeth, (9, 9), 0)
    if teeth.max() == 0:
        return bgr
    h_c, s_c, v_c = cv2.split(hsv)
    v_new = cv2.add(v_c, int(amount * 0.45))
    s_new = cv2.subtract(s_c, int(amount * 0.45))
    adjusted = cv2.cvtColor(cv2.merge([h_c, s_new, v_new]), cv2.COLOR_HSV2BGR)
    alpha = (teeth.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(adjusted * alpha + bgr * (1 - alpha), 0, 255).astype(np.uint8)


def level_head(bgr, faces):
    """以第一张脸的双眼连线做水平矫正，带方向自检"""
    if not faces:
        return bgr
    pts = faces[0]
    if len(pts) < 363:
        return bgr
    l, r = _eye_centers(pts)
    dx, dy = float(r[0] - l[0]), float(r[1] - l[1])
    if abs(dx) < 1e-3:
        return bgr
    angle = float(np.clip(np.degrees(np.arctan2(dy, dx)), -15.0, 15.0))
    if abs(angle) < 0.3:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    faces2 = get_landmarks(out)  # 方向自检
    if faces2 and len(faces2[0]) >= 363:
        l2, r2 = _eye_centers(faces2[0])
        if abs(np.degrees(np.arctan2(r2[1] - l2[1], r2[0] - l2[0]))) > abs(angle):
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
            out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
    return out


def head_turn(bgr, faces, yaw, pitch):
    """头部左右转/上下仰：每张脸各自透视形变 + 椭圆羽化混合（2D 近似）"""
    if yaw == 0 and pitch == 0:
        return bgr
    h, w = bgr.shape[:2]
    out = bgr.copy()
    for pts in faces:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        fw, fh = float(x_max - x_min), float(y_max - y_min)
        if fw < 8 or fh < 8:
            continue

        mx, my = fw * 0.35, fh * 0.30
        x0 = int(max(0, x_min - mx)); x1 = int(min(w, x_max + mx))
        y0 = int(max(0, y_min - my)); y1 = int(min(h, y_max + my))
        roi = out[y0:y1, x0:x1].copy()
        rh, rw = roi.shape[:2]

        yaw_f, pitch_f = yaw / 50.0, pitch / 50.0
        sx = rw * 0.16 * yaw_f
        sy = rh * 0.14 * pitch_f

        src = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
        dst = np.float32([
            [sx if yaw_f > 0 else -sx * 0.4, sy if pitch_f < 0 else -sy * 0.4],
            [rw + (sx if yaw_f < 0 else sx * 0.4), sy if pitch_f < 0 else -sy * 0.4],
            [rw + (sx if yaw_f < 0 else sx * 0.4), rh - (sy if pitch_f > 0 else sy * 0.4)],
            [sx if yaw_f > 0 else -sx * 0.4, rh - (sy if pitch_f > 0 else sy * 0.4)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(roi, M, (rw, rh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)

        mask = np.zeros((rh, rw), np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2),
                    (int(rw * 0.46), int(rh * 0.48)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        blended = warped.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)
        out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def move_eyes(bgr, faces, eyes_x, eyes_y):
    """眼球左右/上下：眼周内容整体位移（含眼睑，视觉上像转动视线）"""
    if eyes_x == 0 and eyes_y == 0:
        return bgr
    out = bgr
    vx = eyes_x / 50.0
    vy = eyes_y / 50.0
    for pts in faces:
        lc, rc = _eye_centers(pts)
        ew = (float(np.linalg.norm(pts[33] - pts[133])) +
              float(np.linalg.norm(pts[362] - pts[263]))) / 2.0
        if ew <= 1:
            continue
        radius = 1.25 * ew
        dx_s = vx * 0.10 * ew
        dy_s = vy * 0.08 * ew
        out = _warp(out, lc, radius, lambda dx, dy, d: (np.full_like(d, dx_s), np.full_like(d, dy_s)))
        out = _warp(out, rc, radius, lambda dx, dy, d: (np.full_like(d, dx_s), np.full_like(d, dy_s)))
    return out


# ---------------- 美颜（只作用于皮肤） ----------------
def get_skin_mask(bgr, faces):
    """皮肤遮罩（float32 0~1）。
    = 脸部核心区(FaceOval ∩ 肤色, 全权重) ⊕ 颈部/肩颈延伸区(肤色, 0.75 权重)，
    排除眼/嘴，边缘羽化。延伸区保证美白/磨皮覆盖脖子，避免脸部与颈部断层。"""
    h, w = bgr.shape[:2]
    core = np.zeros((h, w), np.uint8)
    ext = np.zeros((h, w), np.uint8)
    for pts in faces:
        oval_pts = pts[FACE_OVAL]
        cv2.fillPoly(core, [oval_pts.astype(np.int32)], 255)
        for poly in (LEFT_EYE_POLY, RIGHT_EYE_POLY, MOUTH_POLY):
            cv2.fillPoly(core, [pts[poly].astype(np.int32)], 0)
        # 延伸区：面部下方的椭圆，覆盖脖子/肩颈肤色
        fw = float(np.linalg.norm(pts[LEFT_CHEEK] - pts[RIGHT_CHEEK]))
        fh = float(oval_pts[:, 1].max() - oval_pts[:, 1].min())
        if fw > 4 and fh > 4:
            center = (int(oval_pts[:, 0].mean()),
                      int(oval_pts[:, 1].mean() + fh * 0.9))
            cv2.ellipse(ext, center, (int(fw * 1.15), int(fh * 1.7)),
                        0, 0, 360, 255, -1)
    if core.max() == 0 and ext.max() == 0:
        return np.zeros((h, w), np.float32)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    core = cv2.bitwise_and(core, skin)
    ext = cv2.bitwise_and(ext, skin)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ext = cv2.morphologyEx(ext, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    kb = max(3, int(max(h, w) / 1000.0 * 15)) | 1
    core_f = cv2.GaussianBlur(core, (kb, kb), 0).astype(np.float32) / 255.0
    ext_f = cv2.GaussianBlur(ext, (kb, kb), 0).astype(np.float32) / 255.0
    return np.clip(np.maximum(core_f, 0.75 * ext_f), 0.0, 1.0)


def apply_beauty(bgr, faces, smooth, whiten):
    if (smooth <= 0 and whiten <= 0) or not faces:
        return bgr
    mask = get_skin_mask(bgr, faces)
    if mask.max() <= 0:
        return bgr
    out = bgr.astype(np.float32)

    if smooth > 0:
        d = int(smooth / 10) + 5
        sigma = max(smooth, 1)
        smoothed = cv2.bilateralFilter(bgr, d, sigma, sigma)
        a = (mask * (smooth / 100.0))[:, :, None]
        out = smoothed.astype(np.float32) * a + out * (1 - a)

    if whiten > 0:
        # 与原版一致的配方：YCrCb 亮度通道提亮 (Y + amount*0.6)，
        # 再按 amount/100 强度在皮肤遮罩内混合，色度不变、不发灰
        base = np.clip(out, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(base, cv2.COLOR_BGR2YCR_CB)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        y_new = cv2.convertScaleAbs(y_ch, alpha=1.0, beta=whiten * 0.6)
        whitened = cv2.cvtColor(cv2.merge([y_new, cr_ch, cb_ch]),
                                cv2.COLOR_YCR_CB2BGR)
        a = (mask * (whiten / 100.0))[:, :, None]
        out = whitened.astype(np.float32) * a + out * (1 - a)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_sharpen(bgr, amount):
    if amount <= 0:
        return bgr
    a = amount / 50.0
    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    return cv2.addWeighted(bgr, 1.0 + a, blur, -a, 0)


# ---------------- ComfyUI 节点 ----------------
class FaceBeautyNode:
    """综合美颜精修（控件名已汉化，界面语言为英文时同样显示中文）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "磨皮": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "双边滤波磨皮，只作用于皮肤区域"}),
                "美白": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "YCrCb 亮度提亮（色度不变），覆盖脸部与颈部皮肤"}),
                "锐化": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "USM 锐化（全图，让五官更清晰）"}),
                "自动摆正头部": ("BOOLEAN", {"default": False,
                                            "tooltip": "按第一张脸的双眼连线自动水平摆正"}),
                "瘦脸": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "两侧脸颊向脸中轴收窄"}),
                "大眼": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "以眼心为圆心径向放大"}),
                "微笑": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                "display": "slider", "tooltip": "两侧嘴角向上向外轻推"}),
                "牙齿美白": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                    "display": "slider", "tooltip": "仅嘴内亮而低饱和的像素提亮降饱和"}),
                "头部左右转": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1,
                                      "display": "slider", "tooltip": "正值向观众视角右侧转（2D 透视近似）"}),
                "头部上下仰": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1,
                                      "display": "slider", "tooltip": "正值抬头、负值低头（2D 透视近似）"}),
                "眼球左右": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1,
                                    "display": "slider", "tooltip": "眼球左右看"}),
                "眼球上下": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1,
                                    "display": "slider", "tooltip": "眼球上下看"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "process"
    CATEGORY = "美颜精修"
    DESCRIPTION = "皮肤级美颜（磨皮/美白/锐化）+ 面部精修（瘦脸/大眼/微笑/牙齿美白）+ 头部与眼球姿态调整，支持多人脸。"

    def process(self, **kwargs):
        """ComfyUI 以关键字参数调用，键名与 INPUT_TYPES 中的中文键一致。"""
        return (self._process_batch(
            kwargs["图像"],
            kwargs["磨皮"], kwargs["美白"], kwargs["锐化"],
            kwargs["自动摆正头部"], kwargs["瘦脸"], kwargs["大眼"],
            kwargs["微笑"], kwargs["牙齿美白"],
            kwargs["头部左右转"], kwargs["头部上下仰"],
            kwargs["眼球左右"], kwargs["眼球上下"],
        ),)

    def _process_batch(self, image, smooth_skin, whiten_skin, sharpen, auto_align,
                       slim_face, big_eyes, smile, teeth_whiten,
                       head_yaw, head_pitch, eyes_x, eyes_y):
        batch = []
        for i in range(image.shape[0]):
            arr = (image[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            try:
                out = self._process_one(bgr, smooth_skin, whiten_skin, sharpen,
                                        auto_align, slim_face, big_eyes, smile,
                                        teeth_whiten, head_yaw, head_pitch,
                                        eyes_x, eyes_y)
            except Exception:
                out = bgr  # 任何异常都原图返回，不炸队列
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            batch.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
        return torch.stack(batch, dim=0)

    def _process_one(self, bgr, smooth, whiten, sharpen, auto_align,
                     slim, big, smile_amt, teeth, yaw, pitch, ex, ey):
        faces = get_landmarks(bgr)
        if not faces:
            return bgr  # 无人脸：原图返回

        # 1) 摆正（几何变化后重新取关键点）
        if auto_align:
            bgr = level_head(bgr, faces)
            faces = get_landmarks(bgr)
            if not faces:
                return bgr

        # 2) 头部转动（先做大形变，再刷新关键点做局部精修）
        bgr = head_turn(bgr, faces, yaw, pitch)
        if yaw != 0 or pitch != 0:
            faces = get_landmarks(bgr)
            if not faces:
                return bgr

        # 3) 局部形变精修
        bgr = slim_face(bgr, faces, slim)
        bgr = enlarge_eyes(bgr, faces, big)
        bgr = smile_morph(bgr, faces, smile_amt)
        bgr = whiten_teeth(bgr, faces, teeth)
        bgr = move_eyes(bgr, faces, ex, ey)

        # 4) 美颜（皮肤遮罩）+ 锐化
        bgr = apply_beauty(bgr, faces, smooth, whiten)
        bgr = apply_sharpen(bgr, sharpen)
        return bgr


NODE_CLASS_MAPPINGS = {"FaceBeautyNode": FaceBeautyNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceBeautyNode": "✨ 综合美颜精修（皮肤级·面部形变）"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
