# pip install ollama requests Pillow
# ollamaqwen.py
# python ollamaqwen.py --image_path images/image17.jpg --model qwen2.5vl:7b

import argparse
import requests
import os
import json
import csv
import base64
from collections import OrderedDict

CSV_LOG = "daycare_result_pipeline.csv"

DAYCARE_PROMPT = """
You are an evidence-based, objective, and honest AI assistant specializing in visual analysis of daycare environments for research, safety, behavioral science, and developmental psychology.

Your task: Analyze the image strictly as a daycare scene and provide an **exhaustive, structured, sectioned report** using only visible, verifiable evidence. Do NOT speculate beyond what is seen. Your analysis must be quantitative and qualitative, and should clearly differentiate between what is certain and what is uncertain or invisible.

**1. Children**
- Total number of children visible.
- For each child (numbered):
    - Apparent gender (or 'uncertain')
    - Best-estimate age (or age range)
    - Mood/emotion (happy, sad, anxious, engaged, bored, neutral, playful, etc.—describe only if visually evident; else state 'uncertain')
    - Activity: What is the child doing (e.g., playing, eating, drawing, napping, interacting)?
    - Is this activity individual or part of a group? If group, specify group size and groupings.
    - Face visibility (clearly visible, partially visible, occluded, turned away, blurred—describe reason for invisibility)
    - Physical interactions (with peers, adults, objects—be specific)
    - Apparent engagement (active, passive, withdrawn, disruptive, etc.)

**2. Adults / Supervisors**
- Total number of adults (supervisors, teachers, caregivers) visible.
- For each adult (numbered):
    - Apparent gender (or 'uncertain')
    - Apparent role (if clear: supervisor, teacher, other staff)
    - Mood/emotion (smiling, attentive, distracted, neutral, stressed, caring—based on facial/body cues)
    - Activity (actively supervising, assisting, interacting, passive, on phone, not engaged, etc.)
    - Supervisor-child association: Are they interacting with children? If so, how (group/individual)?
    - Face visibility (same categories as above)

**3. Activities & Group Dynamics**
- List every distinct activity observed (e.g., play, study, eating, cleaning, conflict, creative/art, resting).
- For each activity:
    - Number and identity (by position/appearance) of participating children/adults.
    - Is activity individual or group? Describe any group structure (size, composition, collaboration).
    - Note any leader/initiator if visible.
    - Are supervisors facilitating or observing?
    - Apparent mood/tone of each activity (joyful, tense, quiet, chaotic, focused, etc.)
- Describe overall spatial layout and interaction patterns.

**4. Supervision & Association**
- For each child: Is direct supervision present? (Describe adult proximity and attentiveness)
- Are supervisors circulating, stationary, or engaged with specific children/groups?
- Are any children unsupervised? If so, describe their situation.

**5. Safety, Risk, Hygiene, Accessibility**
- Clearly identify any visible hazards, risks, or unsafe conditions (sharp objects, clutter, open doors/windows, climbing, unsupervised risky activity, etc.)
- Note visible hygiene factors (cleanliness of floors/tables/toys, visible sanitation supplies)
- Is space overcrowded or spacious? Describe density and child-to-supervisor ratio.
- Note accessibility (ramps, clear walkways, child-appropriate furniture).
- If no risks, hazards, or hygiene/accessibility issues are visible, explicitly state so.

**6. Facility Environment**
- Is the setting indoors or outdoors? State visible clues (windows, walls, floor, light, greenery).
- Quality and type of furniture, toys, learning materials present (if any).
- General atmosphere: Safe, inviting, orderly, chaotic, dull, vibrant, etc. (based only on visual cues)

**7. Summary & Quantitative Insights**
- Exact counts for children, adults, groups, activities.
- Proportions: Gender, age (if inferable), child:supervisor ratio.
- Group and individual activities, engagement, and supervision quality.
- Overall mood, group dynamics, and quality of environment.
- Clearly mention any aspects that cannot be determined or are not visible (“Age of child 2 unclear due to occlusion”, “Cannot confirm if all children are supervised”, etc.)

**Important Instructions:**
- Base all answers strictly on visible evidence; never speculate.
- Clearly differentiate certain from uncertain/invisible findings.
- Use numbered or bulleted lists under each section.
- Strive for precision, clarity, and research-grade structure.
"""  

def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        print(f"Downloading image from: {image_path_or_url}")
        resp = requests.get(image_path_or_url)
        resp.raise_for_status()
        return resp.content
    else:
        print(f"Loading image from: {image_path_or_url}")
        with open(image_path_or_url, 'rb') as f:
            return f.read()

def call_ollama_vision_fast(prompt, image_path_or_url, model='qwen2.5vl:7b'):
    image_bytes = load_image(image_path_or_url)
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }],
        "stream": False,
        #Unload the model and free up memory us'keep_alive': 0 to keep model loaded for speed.
        # Optionally set 'keep_alive': '10m'
        "keep_alive": "0"
    }
    try:
        r = requests.post("http://localhost:11434/api/chat", json=payload)
        r.raise_for_status()
        response = r.json()
        if 'message' in response and 'content' in response['message']:
            output = response['message']['content']
        elif 'response' in response:
            output = response['response']
        else:
            output = None

        metrics_fields = [
            "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration",
            "eval_count", "eval_duration"
        ]
        metrics = {}
        for k in metrics_fields:
            metrics[k] = response.get(k, "")

        try:
            eval_count_num = float(metrics.get("eval_count", 0))
            eval_duration_num = float(metrics.get("eval_duration", 0))
            tokens_per_second = (eval_count_num / eval_duration_num) * 1e9 if eval_duration_num > 0 else ""
            if tokens_per_second != "":
                tokens_per_second = round(tokens_per_second, 2)
        except Exception:
            tokens_per_second = ""

        return output, {
            "vlm_total_duration": metrics.get("total_duration", ""),
            "vlm_load_duration": metrics.get("load_duration", ""),
            "vlm_prompt_eval_count": metrics.get("prompt_eval_count", ""),
            "vlm_prompt_eval_duration": metrics.get("prompt_eval_duration", ""),
            "vlm_eval_count": metrics.get("eval_count", ""),
            "vlm_eval_duration": metrics.get("eval_duration", ""),
            "vlm_tokens_per_second": tokens_per_second
        }

    except Exception as e:
        print(f"[ERROR] Ollama HTTP API failed: {e}")
        return None, {}

def update_last_row_with_vlm_metrics(vlm_metrics_dict, csv_path=CSV_LOG):
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} does not exist; cannot update last row.")
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [OrderedDict(row) for row in reader]
        existing_fieldnames = reader.fieldnames if reader.fieldnames else []

    if not rows:
        print(f"[WARN] No rows in {csv_path}; cannot update last row.")
        return

    last_row = rows[-1]
    for k, v in vlm_metrics_dict.items():
        last_row[k] = v

    new_fieldnames = list(existing_fieldnames)
    for k in vlm_metrics_dict:
        if k not in new_fieldnames:
            new_fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path or URL to the image")
    parser.add_argument("--model", type=str, required=True, help="Ollama Vision-Language model name (e.g. 'qwen2.5vl:7b')")
    parser.add_argument("--output_dir", type=str, default="vlm_output", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output, vlm_metrics = call_ollama_vision_fast(DAYCARE_PROMPT, args.image_path, args.model)
    if output:
        img_base = os.path.splitext(os.path.basename(args.image_path))[0]
        out_file = os.path.join(args.output_dir, f"{img_base}_vlm.json")
        result_obj = {
            "model": args.model,
            "image": args.image_path,
            "output": output,
            **vlm_metrics
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result_obj, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Qwen output saved to {out_file}")

        vlm_log_dict = {
            "vlm_model": args.model,
            "vlm_response": output,
            **vlm_metrics
        }
        update_last_row_with_vlm_metrics(vlm_log_dict)

    else:
        print("[ERROR] No output received.")

