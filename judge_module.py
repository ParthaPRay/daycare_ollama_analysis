#judge_module.py

# python judge_module.py --yolo_output_json yolodeepface_output/image17_yolodeepface.json --vlm_output_json vlm_output/image17_vlm.json --judge_model deepseek-r1:1.5b

import json
import argparse
import requests
import os
import csv
from collections import OrderedDict

CSV_LOG = "daycare_result_pipeline.csv"

def judge_inference(
    yolo_output_json: str,
    vlm_output_json: str,
    judge_model: str = 'deepseek-r1:1.5b',
    keep_alive: str = "10m"
) -> tuple[str, dict]:
    # Load YOLO+DeepFace outputs
    with open(yolo_output_json, 'r', encoding='utf-8') as f:
        yolo_data = json.load(f)
    yolo_summary = yolo_data.get('YOLO_Person_Summary', '')
    deepface_summary = yolo_data.get('DeepFace_Emotion_Summary', '')

    # Load VLM output JSON
    with open(vlm_output_json, 'r', encoding='utf-8') as f:
        vlm_data = json.load(f)
    vlm_output = vlm_data.get('output', '')

    # Construct judge prompt
    judge_prompt = f"""
You are a highly reliable, evidence-driven AI judge for daycare scene analysis pipelines. Your primary data is the output of an advanced Vision-Language Model (VLM) that comprehensively analyzes the image for all relevant aspects. You also receive basic person/face counts and box-level emotions from a YOLO+DeepFace pipeline.

**Inputs:**
- **YOLO+DeepFace JSON:** {yolo_summary}\n{deepface_summary}
- **VLM JSON:** {vlm_output}

**Instructions:**
1. **Center your analysis on the VLM output**, as it contains vision-aware, detail-rich, and evidence-based information. Use YOLO+DeepFace strictly for cross-verification of basic, countable facts (e.g., number of persons, crude mood, rough age/gender cues).
2. **For each key aspect**—children's mood, activity, number, gender, age; supervisor presence, number, mood, gender, activity; environment (indoor/outdoor); risk/hazard indicators:
   - **Extract and summarize the quantitative and qualitative details** from the VLM output.
   - **Cross-check the most basic facts** (like person/face counts and any high-confidence mood tags) from YOLO+DeepFace for consistency. If there are discrepancies, clearly state which source you trust and why, and which aspects remain uncertain.
   - **Do NOT simply compare outputs.** Always explain your reasoning for trusting VLM as the authoritative source unless clear evidence from YOLO+DeepFace indicates a basic error or miss in the VLM output.
3. **Risk and Hazard Assessment:** Carefully extract any indicators of danger, lack of supervision, hazardous objects, or visible distress from the VLM output. If YOLO+DeepFace highlights specific objects or risks missed by the VLM, mention them explicitly.
4. **Produce a structured final judgment** including:
    - **Quantitative Summary:** Key numbers (children, adults, visible gender/age breakdown, emotional distribution, count of risky objects, etc.)
    - **Qualitative Assessment:** Overall quality of care, engagement, mood, supervision, activity variety, environment, and any concerning findings.
    - **Discrepancy Analysis:** Explicitly list points where YOLO+DeepFace and VLM disagree on any critical fact, and explain your final stance.
    - **Risk/Hazard Assessment:** Highlight specific risks or safety concerns with supporting evidence.
    - **Overall Judgment & Recommendations:** Clear, actionable verdict on daycare safety, supervision, child well-being, and any recommended actions.
5. **Provide the following confidence metrics at the end (all 0–100%)**:
    - **Analysis confidence:** (your overall confidence in the judgment)
    - **Reliability:** (your view of the data's trustworthiness)
    - **Visual clarity:** (how clear and interpretable the visual data was)
    - **Reasoning confidence:** (how certain you are in your analysis logic)
6. **Be strictly evidence-based, do not speculate beyond the image and outputs.**
7. **Conclude with a clear summary paragraph** describing the overall state and safety of the daycare scenario, including key points and your metrics.

Remember, **the VLM output is authoritative**; YOLO+DeepFace is used only for simple verification of countable, visible facts.
"""
    # Call Ollama HTTP API for metrics
    payload = {
        "model": judge_model,
        "messages": [{
            "role": "user",
            "content": judge_prompt
        }],
        "stream": False,
        "keep_alive": keep_alive
    }
    r = requests.post("http://localhost:11434/api/chat", json=payload)
    r.raise_for_status()
    response = r.json()

    if 'message' in response and 'content' in response['message']:
        judge_response = response['message']['content']
    elif 'response' in response:
        judge_response = response['response']
    else:
        judge_response = ""

    # Gather reasoning_ metrics from Ollama API root
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

    reasoning_metrics = {
        "reasoning_total_duration": metrics.get("total_duration", ""),
        "reasoning_load_duration": metrics.get("load_duration", ""),
        "reasoning_prompt_eval_count": metrics.get("prompt_eval_count", ""),
        "reasoning_prompt_eval_duration": metrics.get("prompt_eval_duration", ""),
        "reasoning_eval_count": metrics.get("eval_count", ""),
        "reasoning_eval_duration": metrics.get("eval_duration", ""),
        "reasoning_tokens_per_second": tokens_per_second
    }

    return judge_response, reasoning_metrics

def update_last_row_with_reasoning_metrics(reasoning_metrics_dict, csv_path=CSV_LOG):
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
    for k, v in reasoning_metrics_dict.items():
        last_row[k] = v

    new_fieldnames = list(existing_fieldnames)
    for k in reasoning_metrics_dict:
        if k not in new_fieldnames:
            new_fieldnames.append(k)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Deepseek judge on YOLO+DeepFace and Qwen outputs')
    parser.add_argument('--yolo_output_json', required=True, help='Path to YOLO+DeepFace JSON')
    parser.add_argument('--vlm_output_json', required=True, help='Path to Qwen VLM JSON output')
    parser.add_argument('--judge_model', default='deepseek-r1:8b', help='Judge model name')
    parser.add_argument('--output_dir', type=str, default='judge_output', help='Directory to save judge results')
    parser.add_argument('--keep_alive', type=str, default='10m', help='Ollama keep_alive string')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Output filename based on YOLO file base name
    yolo_base = os.path.splitext(os.path.basename(args.yolo_output_json))[0]
    output_json_path = os.path.join(args.output_dir, f"{yolo_base}_judge.json")

    result_text, reasoning_metrics = judge_inference(
        args.yolo_output_json,
        args.vlm_output_json,
        args.judge_model,
        args.keep_alive
    )

    # Save judge result as JSON
    result_obj = {
        "reasoning_model": args.judge_model,
        "yolo_json": args.yolo_output_json,
        "vlm_json": args.vlm_output_json,
        "reasoning_model_response": result_text,
        **reasoning_metrics
    }
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Judge output saved to {output_json_path}\n")
    print(result_text)

    # Update last row of pipeline CSV
    update_last_row_with_reasoning_metrics(result_obj, CSV_LOG)

