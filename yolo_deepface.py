# yolo_deepface.py
# Usage example:
# python yolo_deepface.py --image_path images/image17.jpg --yolo_model yolo11x.pt --deepface_backend retinaface


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
import cv2
import csv
import datetime
from ultralytics import YOLO
from deepface import DeepFace

CSV_LOG = "daycare_result_pipeline.csv"

# --- Important COCO class indices ---
RISKY_CLASSES = {42, 43, 76, 40}  # fork, knife, scissors, wine glass

IMPORTANT_CLASSES = {
    24, 26, 32, 39, 41, 45, 46, 47, 49, 56, 57, 58, 59, 60, 73, 77
}
# Subset: Most daycare-specific items
DAYCARE_CLASSES = {
    24,    # backpack
    26,    # handbag
    32,    # sports ball
    39,    # bottle
    41,    # cup
    45,    # bowl
    46,    # banana
    47,    # apple
    49,    # orange
    56,    # chair
    57,    # couch
    58,    # potted plant
    59,    # bed
    60,    # dining table
    73,    # book
    77     # teddy bear
}

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

def to_python_type(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif hasattr(obj, "item") and callable(obj.item):
        return obj.item()
    else:
        return obj

def resize_with_aspect_ratio(image, max_dim=720):
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def log_csv(row, csv_path=CSV_LOG):
    fieldnames = list(row.keys())
    file_exists = os.path.isfile(csv_path)
    write_header = not file_exists or os.stat(csv_path).st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def draw_emotion_label(img, fx1, fy1, fx2, fy2, emotion_text):
    margin = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
    label_x = fx1
    label_y = fy1 - text_h - 8 if fy1 - text_h - 8 > 0 else fy1 + text_h + 8
    cv2.rectangle(
        img,
        (label_x - margin, label_y - text_h - margin),
        (label_x + text_w + margin, label_y + margin),
        (30, 30, 30),
        thickness=-1,
    )
    cv2.putText(
        img,
        emotion_text,
        (label_x, label_y),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA,
    )

def analyze_image(image_path, yolo_model_path, deepface_backend, output_dir, max_dim=720):
    os.makedirs(output_dir, exist_ok=True)
    img_basename = os.path.basename(image_path)
    img_name, img_ext = os.path.splitext(img_basename)
    output_image_path = os.path.join(output_dir, f"{img_name}_output{img_ext}")
    output_json_path = os.path.join(output_dir, f"{img_name}_yolodeepface.json")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = resize_with_aspect_ratio(img, max_dim=max_dim)
    h, w = img.shape[:2]

    # --- YOLO Detection (all objects) ---
    yolo_model = YOLO(yolo_model_path)
    results = yolo_model(img)

    person_boxes = []
    risky_items = []
    important_items = []
    daycare_items = []
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        class_id = int(cls)
        label = COCO_NAMES[class_id]
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        box_tuple = (x1, y1, x2, y2)

        if label == 'person':
            if (x2 - x1) > 32 and (y2 - y1) > 32:
                person_boxes.append(box_tuple)
        elif class_id in RISKY_CLASSES:
            risky_items.append({
                "class_id": class_id,
                "class_name": label,
                "box": box_tuple
            })
        elif class_id in IMPORTANT_CLASSES:
            important_items.append({
                "class_id": class_id,
                "class_name": label,
                "box": box_tuple
            })
        if class_id in DAYCARE_CLASSES:
            daycare_items.append({
                "class_id": class_id,
                "class_name": label,
                "box": box_tuple
            })

    person_summary = f"Total persons: {len(person_boxes)}; Boxes: {person_boxes}"
    print(f"[INFO] YOLO Person Detection Done: {person_summary}")
    if risky_items:
        print(f"[INFO] Risky Items Detected: {len(risky_items)}")
    if important_items:
        print(f"[INFO] Important Items Detected: {len(important_items)}")
    if daycare_items:
        print(f"[INFO] Daycare Items Detected: {len(daycare_items)}")

    # --- DeepFace Emotion Analysis ---
    emotion_summary = []
    for pidx, (px1, py1, px2, py2) in enumerate(person_boxes):
        person_crop = img[py1:py2, px1:px2].copy()
        try:
            results = DeepFace.analyze(
                img_path=person_crop,
                actions=['emotion'],
                detector_backend=deepface_backend,
                align=True,
                enforce_detection=True
            )
            if not isinstance(results, list):
                results = [results]
            for fidx, result in enumerate(results):
                all_emotions = to_python_type(result['emotion'])
                region = result.get('region', None)
                summary_line = {
                    "PersonIndex": pidx + 1,
                    "FaceIndex": fidx + 1,
                    "Emotions": all_emotions,
                    "Region": to_python_type(region)
                }
                emotion_summary.append(summary_line)
                # Draw face box and top-3 emotions
                if region:
                    fx1 = px1 + region['x']
                    fy1 = py1 + region['y']
                    fx2 = fx1 + region['w']
                    fy2 = fy1 + region['h']
                    cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                    # Get top-3 emotions
                    sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                    emotion_text = " | ".join([f"{em} {score:.1f}%" for em, score in sorted_emotions])
                    draw_emotion_label(img, fx1, fy1, fx2, fy2, emotion_text)
        except Exception as e:
            print(f"[WARN] Could not analyze Person#{pidx + 1} with DeepFace: {e}")

    print(f"[INFO] DeepFace Emotion Analysis Done: {len(emotion_summary)} faces found.")

    # Draw green boxes for all persons
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output_image_path, img)
    print(f"[INFO] Annotated result image saved as: {output_image_path}")

    # --- Prepare Result JSON ---
    result = {
        "YOLO_Person_Summary": person_summary,
        "YOLO_Person_Boxes": person_boxes,
        "Detected_Risky_Items": risky_items,
        "Detected_Important_Items": important_items,
        "Detected_Daycare_Items": daycare_items,
        "DeepFace_Emotion_Summary": emotion_summary,
        "Output_Image": output_image_path
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results JSON saved as: {output_json_path}")

    # --- Print JSON to Terminal at End ---
    print("\n===== Final YOLO + DeepFace Output =====\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # --- Logging to CSV ---
    log_row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "image_path": image_path,
        "yolo_model": yolo_model_path,
        "deepface_backend": deepface_backend,
        "yolo_output": person_summary,
        "deepface_output": json.dumps(emotion_summary, ensure_ascii=False),
        "risky_items": json.dumps(risky_items, ensure_ascii=False),
        "important_items": json.dumps(important_items, ensure_ascii=False),
        "daycare_items": json.dumps(daycare_items, ensure_ascii=False)
    }
    log_csv(log_row)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--yolo_model", type=str, required=True, help="Path to YOLOv8 person detection model")
    parser.add_argument(
        "--deepface_backend",
        type=str,
        default="retinaface",
        help=(
            "DeepFace detector backend (default: retinaface). "
            "Other options: 'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn', 'retinaface', 'mediapipe', 'yolov8', 'yolov11s', 'yolov11n', 'yolov11m', 'yunet', 'centerface',"
        )
    )
    parser.add_argument("--output_dir", type=str, default="yolodeepface_output", help="Output directory for results")
    args = parser.parse_args()
    analyze_image(
        args.image_path,
        args.yolo_model,
        args.deepface_backend,
        args.output_dir
    )

