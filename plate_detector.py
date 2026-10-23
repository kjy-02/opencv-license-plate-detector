import cv2
import numpy as np
import math


class PlateParams:
    def __init__(self):
        self.resize_width = 960

        # ==============================
        # GUI 조절 기본 파라미터
        # ==============================
        self.adaptive_block_size = 19
        self.adaptive_c = 9

        self.morph_width = 11
        self.morph_height = 8

        self.min_area = 3651
        self.max_area = 65518

        self.min_aspect = 2.0
        self.max_aspect = 7.9

        # ==============================
        # 번호판 위치 제한
        # ==============================
        self.top_margin_ratio = 0.10
        self.bottom_margin_ratio = 0.98

        # ==============================
        # 문자 후보 필터
        # ==============================
        self.char_min_area = 20
        self.char_min_width = 2
        self.char_min_height = 8
        self.char_min_aspect = 0.15
        self.char_max_aspect = 1.25

        # ==============================
        # 문자 후보 그룹화 조건
        # ==============================
        self.min_char_count = 3
        self.max_angle_diff = 15.0
        self.max_distance_multiplier = 6.5
        self.max_area_diff = 0.75
        self.max_width_diff = 0.85
        self.max_height_diff = 0.60
        self.max_y_diff_multiplier = 0.70

        # ==============================
        # 최종 후보 조건
        # ==============================
        self.min_best_score = 0.25
        self.max_candidates_to_draw = 5

        # ==============================
        # 번호판 Crop 설정
        # ==============================
        self.plate_width_padding = 1.48
        self.plate_height_padding = 1.65
        self.crop_pad_x_ratio = 0.08
        self.crop_pad_y_ratio = 0.18


def read_image_unicode(filename):
    data = np.fromfile(filename, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def write_image_unicode(filename, image):
    ext = filename.split(".")[-1].lower()

    if ext == "png":
        success, encoded = cv2.imencode(".png", image)
    else:
        success, encoded = cv2.imencode(".jpg", image)

    if not success:
        return False

    encoded.tofile(filename)
    return True


def resize_image(image, target_width):
    height, width = image.shape[:2]

    if width == target_width:
        return image

    scale = target_width / width
    target_height = int(height * scale)

    return cv2.resize(image, (target_width, target_height))


def make_odd(value, minimum=3):
    value = max(minimum, int(value))

    if value % 2 == 0:
        value += 1

    return value


def enhance_plate_text(gray):
    structuring_element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        structuring_element
    )

    black_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        structuring_element
    )

    gray_plus_top_hat = cv2.add(gray, top_hat)
    enhanced_gray = cv2.subtract(gray_plus_top_hat, black_hat)

    return enhanced_gray, top_hat, black_hat


def preprocess_image(src, params):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    enhanced_gray, top_hat, black_hat = enhance_plate_text(gray)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray_clahe = clahe.apply(enhanced_gray)

    blur = cv2.GaussianBlur(
        gray_clahe,
        (5, 5),
        0
    )

    block_size = make_odd(params.adaptive_block_size, 3)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        params.adaptive_c
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (params.morph_width, params.morph_height)
    )

    morph = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        kernel
    )

    return {
        "gray": gray,
        "enhanced_gray": enhanced_gray,
        "top_hat": top_hat,
        "black_hat": black_hat,
        "gray_clahe": gray_clahe,
        "blur": blur,
        "adaptive": adaptive,
        "morph": morph
    }


def find_char_candidates(binary_img, params):
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    char_candidates = []
    idx = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h == 0:
            continue

        area = w * h
        aspect = w / h

        if area < params.char_min_area:
            continue

        if w < params.char_min_width:
            continue

        if h < params.char_min_height:
            continue

        if aspect < params.char_min_aspect:
            continue

        if aspect > params.char_max_aspect:
            continue

        candidate = {
            "idx": idx,
            "contour": contour,
            "rect": (x, y, w, h),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "cx": x + w / 2,
            "cy": y + h / 2,
            "area": area,
            "aspect": aspect
        }

        idx += 1
        char_candidates.append(candidate)

    char_candidates.sort(key=lambda item: item["x"])

    return char_candidates


def distance_between_chars(c1, c2):
    dx = c1["cx"] - c2["cx"]
    dy = c1["cy"] - c2["cy"]
    return math.sqrt(dx * dx + dy * dy)


def angle_between_chars(c1, c2):
    dx = abs(c1["cx"] - c2["cx"])
    dy = abs(c1["cy"] - c2["cy"])

    if dx <= 1e-6:
        return 90.0

    return math.degrees(math.atan(dy / dx))


def is_matching_char(c1, c2, params):
    # 두 문자 후보 사이의 거리 계산
    distance = distance_between_chars(c1, c2)
    # 두 문자 후보가 이루는 기울기 계산
    angle = angle_between_chars(c1, c2)
    # 기준 문자 후보의 대각선 길이 계산
    diagonal_length = math.sqrt(c1["w"] ** 2 + c1["h"] ** 2)
    # 두 문자 후보의 면적 차이 비율 계산
    area_diff = abs(c1["area"] - c2["area"]) / max(c1["area"], c2["area"])
    # 두 문자 후보의 폭 차이 비율 계산
    width_diff = abs(c1["w"] - c2["w"]) / max(c1["w"], c2["w"])
    # 두 문자 후보의 높이 차이 비율 계산
    height_diff = abs(c1["h"] - c2["h"]) / max(c1["h"], c2["h"])
    # 두 문자 후보의 세로 중심 좌표 차이 계산
    y_diff = abs(c1["cy"] - c2["cy"])
    # 두 문자 후보의 평균 높이 계산
    avg_height = (c1["h"] + c2["h"]) / 2

    # 문자 후보 사이가 너무 멀면 같은 번호판 문자로 보지 않음
    if distance > diagonal_length * params.max_distance_multiplier:
        return False
    # 기울기가 너무 크면 같은 줄의 문자로 보지 않음
    if angle > params.max_angle_diff:
        return False
    # 면적 차이가 너무 크면 제외
    if area_diff > params.max_area_diff:
        return False
    # 폭 차이가 너무 크면 제외
    if width_diff > params.max_width_diff:
        return False
    # 높이 차이가 너무 크면 제외
    if height_diff > params.max_height_diff:
        return False
    # 세로 위치 차이가 너무 크면 제외
    if y_diff > avg_height * params.max_y_diff_multiplier:
        return False
    # 모든 조건을 만족하면 같은 번호판 문자 후보로 판단
    return True


def compute_group_width(chars):
    if len(chars) == 0:
        return 0

    x1 = min(item["x"] for item in chars)
    x2 = max(item["x"] + item["w"] for item in chars)

    return x2 - x1


def group_matching_chars(char_candidates, params):
    groups = []

    for base_char in char_candidates:
        group = [base_char]

        for compare_char in char_candidates:
            if base_char is compare_char:
                continue

            if is_matching_char(base_char, compare_char, params):
                group.append(compare_char)

        if len(group) < params.min_char_count:
            continue

        group = sorted(group, key=lambda item: item["x"])

        group_ids = set(id(item) for item in group)

        duplicated = False

        for existing_group in groups:
            existing_ids = set(id(item) for item in existing_group)
            overlap = len(group_ids.intersection(existing_ids))

            if overlap >= min(len(group_ids), len(existing_ids)) * 0.6:
                duplicated = True
                break

        if not duplicated:
            groups.append(group)

    groups.sort(
        key=lambda group: (len(group), compute_group_width(group)),
        reverse=True
    )

    return groups


def compute_alignment_score(chars):
    if len(chars) < 2:
        return 0.0

    cy_values = np.array([item["cy"] for item in chars], dtype=np.float32)
    h_values = np.array([item["h"] for item in chars], dtype=np.float32)

    cy_std = np.std(cy_values)
    avg_h = np.mean(h_values)

    if avg_h <= 1e-6:
        return 0.0

    score = 1.0 - min(1.0, cy_std / avg_h)

    return float(score)


def compute_x_spread_score(chars):
    if len(chars) < 2:
        return 0.0

    chars = sorted(chars, key=lambda item: item["x"])

    distances = []

    for i in range(len(chars) - 1):
        d = chars[i + 1]["cx"] - chars[i]["cx"]
        distances.append(d)

    distances = np.array(distances, dtype=np.float32)

    if len(distances) == 0:
        return 0.0

    mean_d = np.mean(distances)

    if mean_d <= 1e-6:
        return 0.0

    std_d = np.std(distances)

    score = 1.0 - min(1.0, std_d / mean_d)

    return float(score)


def compute_aspect_score(aspect, min_aspect, max_aspect):
    if aspect < min_aspect or aspect > max_aspect:
        return 0.0

    ideal = 4.5
    diff = abs(aspect - ideal)

    score = 1.0 - min(1.0, diff / ideal)

    return float(score)


def compute_area_score(area, min_area, max_area):
    if area < min_area or area > max_area:
        return 0.0

    ideal = (min_area + max_area) / 2
    diff = abs(area - ideal)

    score = 1.0 - min(1.0, diff / ideal)

    return float(score)


def compute_roi_score(x, y, w, h, image_width, image_height):
    cx = x + w / 2
    cy = y + h / 2

    norm_x = abs(cx - image_width * 0.5) / image_width
    norm_y = abs(cy - image_height * 0.62) / image_height

    score = 1.0 - min(1.0, norm_x * 1.5 + norm_y * 1.0)

    return float(score)


def estimate_plate_from_char_group(src, char_group, params):
    height, width = src.shape[:2]

    chars = sorted(char_group, key=lambda item: item["cx"])

    first = chars[0]
    last = chars[-1]

    plate_cx = (first["cx"] + last["cx"]) / 2
    plate_cy = (first["cy"] + last["cy"]) / 2

    x_min = min(item["x"] for item in chars)
    x_max = max(item["x"] + item["w"] for item in chars)

    plate_width = (x_max - x_min) * params.plate_width_padding

    avg_char_height = np.mean([item["h"] for item in chars])
    plate_height = avg_char_height * params.plate_height_padding

    if plate_width <= 0 or plate_height <= 0:
        return None

    triangle_height = last["cy"] - first["cy"]
    triangle_width = last["cx"] - first["cx"]

    hypotenuse = math.sqrt(
        triangle_width * triangle_width +
        triangle_height * triangle_height
    )

    if hypotenuse <= 1e-6:
        angle = 0.0
    else:
        angle = math.degrees(math.asin(triangle_height / hypotenuse))

    rotated_rect = (
        (float(plate_cx), float(plate_cy)),
        (float(plate_width), float(plate_height)),
        float(angle)
    )

    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    bx, by, bw, bh = cv2.boundingRect(box)

    bx1 = max(0, bx)
    by1 = max(0, by)
    bx2 = min(width - 1, bx + bw)
    by2 = min(height - 1, by + bh)

    if bx2 <= bx1 or by2 <= by1:
        return None

    axis_w = bx2 - bx1
    axis_h = by2 - by1

    area = axis_w * axis_h
    aspect = plate_width / plate_height

    roi_score = compute_roi_score(
        bx1,
        by1,
        axis_w,
        axis_h,
        width,
        height
    )

    aspect_score = compute_aspect_score(
        aspect,
        params.min_aspect,
        params.max_aspect
    )

    area_score = compute_area_score(
        area,
        params.min_area,
        params.max_area
    )

    char_count_score = min(1.0, len(chars) / 6)
    alignment_score = compute_alignment_score(chars)
    x_spread_score = compute_x_spread_score(chars)
    angle_score = 1.0 - min(1.0, abs(angle) / 25.0)

    score = (
        aspect_score * 0.22 +
        char_count_score * 0.24 +
        alignment_score * 0.20 +
        x_spread_score * 0.12 +
        roi_score * 0.10 +
        angle_score * 0.07 +
        area_score * 0.05
    )

    candidate = {
        "rect": (bx1, by1, axis_w, axis_h),
        "rotated_rect": rotated_rect,
        "box": box,
        "center": (plate_cx, plate_cy),
        "plate_width": plate_width,
        "plate_height": plate_height,
        "angle": angle,
        "area": area,
        "aspect": aspect,
        "char_count": len(chars),
        "chars": chars,
        "score": score,
        "aspect_score": aspect_score,
        "area_score": area_score,
        "center_score": roi_score,
        "char_count_score": char_count_score,
        "alignment_score": alignment_score,
        "x_spread_score": x_spread_score,
        "angle_score": angle_score
    }

    return candidate


def build_plate_candidates_from_char_groups(src, char_groups, params):
    candidates = []

    height, width = src.shape[:2]

    y_min_limit = int(height * params.top_margin_ratio)
    y_max_limit = int(height * params.bottom_margin_ratio)

    for group in char_groups:
        candidate = estimate_plate_from_char_group(src, group, params)

        if candidate is None:
            continue

        x, y, w, h = candidate["rect"]

        if y < y_min_limit:
            continue

        if y + h > y_max_limit:
            continue

        if candidate["area"] < params.min_area:
            continue

        if candidate["area"] > params.max_area:
            continue

        if candidate["aspect"] < params.min_aspect:
            continue

        if candidate["aspect"] > params.max_aspect:
            continue

        candidates.append(candidate)

    candidates.sort(key=lambda item: item["score"], reverse=True)

    return candidates


def get_valid_candidates(candidates, params):
    valid_candidates = []

    for candidate in candidates:
        if candidate["score"] >= params.min_best_score:
            valid_candidates.append(candidate)

    return valid_candidates


def crop_rotated_plate(src, candidate):
    if candidate is None:
        return None

    height, width = src.shape[:2]

    plate_cx, plate_cy = candidate["center"]
    plate_width = candidate["plate_width"]
    plate_height = candidate["plate_height"]
    angle = candidate["angle"]

    pad_w = plate_width * candidate.get("extra_pad_w", 0.0)
    pad_h = plate_height * candidate.get("extra_pad_h", 0.0)

    crop_width = int(max(1, plate_width + pad_w))
    crop_height = int(max(1, plate_height + pad_h))

    rotation_matrix = cv2.getRotationMatrix2D(
        center=(plate_cx, plate_cy),
        angle=angle,
        scale=1.0
    )

    rotated = cv2.warpAffine(
        src,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    cropped = cv2.getRectSubPix(
        rotated,
        patchSize=(crop_width, crop_height),
        center=(plate_cx, plate_cy)
    )

    return cropped


def crop_best_plate(src, valid_candidates, params):
    if len(valid_candidates) == 0:
        return None

    best = dict(valid_candidates[0])
    best["extra_pad_w"] = params.crop_pad_x_ratio
    best["extra_pad_h"] = params.crop_pad_y_ratio

    crop = crop_rotated_plate(src, best)

    return crop


def make_empty_plate_crop(message="No Plate Detected"):
    img = np.zeros((120, 380, 3), dtype=np.uint8)

    cv2.putText(
        img,
        message,
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    return img


def draw_char_candidates(src, char_candidates):
    result = src.copy()

    for char in char_candidates:
        x, y, w, h = char["rect"]

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            1
        )

    cv2.putText(
        result,
        f"Char Candidates: {len(char_candidates)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    return result


def draw_all_candidates(src, candidates, params):
    result = src.copy()

    draw_candidates = candidates[:params.max_candidates_to_draw]

    for idx, candidate in enumerate(draw_candidates):
        box = candidate["box"]

        if idx == 0 and candidate["score"] >= params.min_best_score:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 200, 255)
            thickness = 2

        cv2.polylines(
            result,
            [box],
            True,
            color,
            thickness
        )

        x, y, w, h = candidate["rect"]

        text = f"{idx + 1}: {candidate['score']:.2f}, chars={candidate['char_count']}"

        cv2.putText(
            result,
            text,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        for char in candidate["chars"]:
            cx, cy, cw, ch = char["rect"]
            cv2.rectangle(
                result,
                (cx, cy),
                (cx + cw, cy + ch),
                (255, 0, 255),
                1
            )

    cv2.putText(
        result,
        f"Plate Candidates: {len(candidates)} | Showing Top {len(draw_candidates)}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return result


def draw_best_plate(src, valid_candidates, raw_candidates, params):
    result = src.copy()

    if len(valid_candidates) == 0:
        cv2.putText(
            result,
            "Plate Not Found",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        if len(raw_candidates) > 0:
            best = raw_candidates[0]
            cv2.putText(
                result,
                f"Best candidate score too low: {best['score']:.2f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
        else:
            cv2.putText(
                result,
                "No valid character group found",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        return result

    best = valid_candidates[0]

    cv2.polylines(
        result,
        [best["box"]],
        True,
        (0, 255, 0),
        4
    )

    for char in best["chars"]:
        cx, cy, cw, ch = char["rect"]

        cv2.rectangle(
            result,
            (cx, cy),
            (cx + cw, cy + ch),
            (255, 0, 255),
            1
        )

    cv2.putText(
        result,
        f"Best Plate | Score: {best['score']:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"Chars: {best['char_count']} | Aspect: {best['aspect']:.2f} | Angle: {best['angle']:.1f}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return result


def make_candidate_score_panel(src, raw_candidates, valid_candidates, char_candidates, char_groups, params):
    panel = np.zeros_like(src)

    y = 40

    cv2.putText(
        panel,
        "Candidate Score Details",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    y += 40

    cv2.putText(
        panel,
        f"Char candidates: {len(char_candidates)} | Char groups: {len(char_groups)}",
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y += 40

    if len(raw_candidates) == 0:
        cv2.putText(
            panel,
            "No plate candidate generated from character groups",
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2
        )
        return panel

    top_candidates = raw_candidates[:min(5, len(raw_candidates))]

    for idx, cand in enumerate(top_candidates):
        is_valid = cand["score"] >= params.min_best_score
        color = (0, 255, 0) if is_valid else (0, 0, 255)

        text1 = (
            f"{idx + 1}. score={cand['score']:.2f}, "
            f"chars={cand['char_count']}, aspect={cand['aspect']:.2f}, angle={cand['angle']:.1f}"
        )

        text2 = (
            f"align={cand['alignment_score']:.2f}, "
            f"spread={cand['x_spread_score']:.2f}, "
            f"center={cand['center_score']:.2f}, "
            f"angleScore={cand['angle_score']:.2f}"
        )

        cv2.putText(
            panel,
            text1,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        y += 30

        cv2.putText(
            panel,
            text2,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            2
        )

        y += 42

    cv2.putText(
        panel,
        f"Valid threshold: score >= {params.min_best_score:.2f}",
        (30, panel.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return panel


def make_debug_view(src, prep, char_img, candidates_img, best_img, plate_crop, score_panel):
    h, w = src.shape[:2]

    gray_bgr = cv2.cvtColor(prep["gray"], cv2.COLOR_GRAY2BGR)
    enhanced_bgr = cv2.cvtColor(prep["enhanced_gray"], cv2.COLOR_GRAY2BGR)
    top_hat_bgr = cv2.cvtColor(prep["top_hat"], cv2.COLOR_GRAY2BGR)
    black_hat_bgr = cv2.cvtColor(prep["black_hat"], cv2.COLOR_GRAY2BGR)
    adaptive_bgr = cv2.cvtColor(prep["adaptive"], cv2.COLOR_GRAY2BGR)
    morph_bgr = cv2.cvtColor(prep["morph"], cv2.COLOR_GRAY2BGR)

    if plate_crop is None:
        crop_view = np.zeros_like(src)

        cv2.putText(
            crop_view,
            "No Plate Crop",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
    else:
        crop_view = np.zeros_like(src)

        crop_h, crop_w = plate_crop.shape[:2]

        scale = min(w / max(crop_w, 1), h / max(crop_h, 1))
        new_w = max(1, int(crop_w * scale * 0.8))
        new_h = max(1, int(crop_h * scale * 0.8))

        crop_resized = cv2.resize(plate_crop, (new_w, new_h))

        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2

        crop_view[start_y:start_y + new_h, start_x:start_x + new_w] = crop_resized

    debug_items = [
        ("Original", src),
        ("Gray", gray_bgr),
        ("TopHat", top_hat_bgr),
        ("BlackHat", black_hat_bgr),
        ("Enhanced Gray", enhanced_bgr),
        ("Adaptive Threshold", adaptive_bgr),
        ("Morphology", morph_bgr),
        ("Character Candidates", char_img),
        ("Plate Candidates", candidates_img),
        ("Best Plate", best_img),
        ("Rotated Plate Crop", crop_view),
        ("Score Details", score_panel)
    ]

    small_w = w // 2
    small_h = h // 2

    small_imgs = []

    for title, img in debug_items:
        temp = img.copy()

        cv2.putText(
            temp,
            title,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2
        )

        small = cv2.resize(temp, (small_w, small_h))
        small_imgs.append(small)

    row1 = np.concatenate([small_imgs[0], small_imgs[1], small_imgs[2]], axis=1)
    row2 = np.concatenate([small_imgs[3], small_imgs[4], small_imgs[5]], axis=1)
    row3 = np.concatenate([small_imgs[6], small_imgs[7], small_imgs[8]], axis=1)
    row4 = np.concatenate([small_imgs[9], small_imgs[10], small_imgs[11]], axis=1)

    debug = np.concatenate([row1, row2, row3, row4], axis=0)

    return debug


def process_image(filename, params=None):
    if params is None:
        params = PlateParams()

    src = read_image_unicode(filename)

    if src is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {filename}")

    src = resize_image(src, params.resize_width)

    prep = preprocess_image(src, params)

    char_candidates = find_char_candidates(prep["adaptive"], params)
    char_groups = group_matching_chars(char_candidates, params)

    raw_candidates = build_plate_candidates_from_char_groups(
        src,
        char_groups,
        params
    )

    valid_candidates = get_valid_candidates(
        raw_candidates,
        params
    )

    char_img = draw_char_candidates(src, char_candidates)
    candidates_img = draw_all_candidates(src, raw_candidates, params)
    best_img = draw_best_plate(src, valid_candidates, raw_candidates, params)

    plate_crop = crop_best_plate(src, valid_candidates, params)

    if plate_crop is None:
        plate_crop_view = make_empty_plate_crop()
    else:
        plate_crop_view = plate_crop.copy()

    score_panel = make_candidate_score_panel(
        src,
        raw_candidates,
        valid_candidates,
        char_candidates,
        char_groups,
        params
    )

    debug_view = make_debug_view(
        src,
        prep,
        char_img,
        candidates_img,
        best_img,
        plate_crop,
        score_panel
    )

    best_score = 0.0
    best_aspect = 0.0
    best_area = 0
    best_char_count = 0
    best_angle = 0.0

    if len(raw_candidates) > 0:
        best_score = raw_candidates[0]["score"]
        best_aspect = raw_candidates[0]["aspect"]
        best_area = raw_candidates[0]["area"]
        best_char_count = raw_candidates[0]["char_count"]
        best_angle = raw_candidates[0]["angle"]

    return {
        "src": src,
        "gray": prep["gray"],
        "gray_clahe": prep["gray_clahe"],
        "enhanced_gray": prep["enhanced_gray"],
        "top_hat": prep["top_hat"],
        "black_hat": prep["black_hat"],
        "adaptive": prep["adaptive"],
        "morph": prep["morph"],
        "char_candidates_result": char_img,
        "candidates_result": candidates_img,
        "best_result": best_img,
        "plate_crop": plate_crop_view,
        "debug": debug_view,
        "score_panel": score_panel,
        "candidate_count": len(raw_candidates),
        "valid_candidate_count": len(valid_candidates),
        "char_candidate_count": len(char_candidates),
        "char_group_count": len(char_groups),
        "plate_found": len(valid_candidates) > 0,
        "best_score": best_score,
        "best_aspect": best_aspect,
        "best_area": best_area,
        "best_char_count": best_char_count,
        "best_angle": best_angle
    }