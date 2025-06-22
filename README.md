# Daycare Multimodal Reasoning Pipeline (Local Edge – Ollama, YOLO, DeepFace)

This repository provides a research-grade, fully local pipeline for in-depth, evidence-based analysis of daycare images, running on resource-constrained edge devices or local servers. It combines **[YOLO object detection](https://docs.ultralytics.com/models/yolo11/)**, **[DeepFace emotion analysis](https://github.com/serengil/deepface)**, a local **Vision-Language Model (VLM)** via [Ollama](https://ollama.com), and a reasoning **Judge Model** (LLM) for robust, explainable scene understanding and automated judgment.

---

## Key Features

* **YOLO + DeepFace**: Detects people, relevant objects, and assesses facial emotions in daycare scenes.
* **VLM (Qwen/Gemma/Moondream, etc.) via Ollama**: Provides detailed, structured, and evidence-based image analysis.
* **Judge/Reasoner Model**: Cross-verifies outputs and delivers final quantitative, qualitative, and risk/safety assessment—authoritative, explainable, and research-focused.
* **Fully Local**: No cloud calls; runs on your local network or edge device.
* **Extensible**: Swap models (YOLO weights, VLM, Judge) as needed.

---

## Pipeline Overview

1. **YOLO + DeepFace** (`yolo_deepface.py`):
   Detects all persons, objects, risky items, and faces in the image. Analyzes emotions for each face. Outputs:

   * Annotated image with emotion labels and detection boxes.
   * A JSON file summarizing people, objects, and detected emotions.

2. **VLM Analysis** (`vlm.py`):
   Sends the image and a detailed prompt to a local VLM via Ollama’s HTTP API (e.g., Qwen2.5VL, Gemma). Produces:

   * A comprehensive, structured textual analysis (JSON).

3. **Reasoning Judge** (`judge_module.py`):
   Invokes a local LLM (e.g., DeepSeek, Qwen3) to:

   * Integrate VLM and YOLO+DeepFace outputs.
   * Deliver a final judgment with quantitative/qualitative summary, risk assessment, discrepancies, and actionable recommendations (JSON).

4. **Orchestrator** (`daycare_pipeline.py`):
   Runs all stages in sequence, passing outputs as inputs, handling files, and ensuring reproducibility.

---

## Example Workflow

```bash
python daycare_pipeline.py
```

* Edit `daycare_pipeline.py` to set your input image, model names, and backends as needed.
* Outputs are saved in respective folders:

  * `yolodeepface_output/`
  * `vlm_output/`
  * `judge_output/`

**Each output includes a `.json` result for downstream analysis or auditability.**

---

## Directory Structure

```
├── daycare_pipeline.py       # Orchestrates the full workflow
├── yolo_deepface.py         # YOLO + DeepFace detection and emotion pipeline
├── vlm.py                   # Vision-Language Model (Ollama) analysis
├── judge_module.py          # Final reasoning & judgment (Ollama LLM)
├── requirements.txt
├── yolodeepface_output/     # Outputs from YOLO+DeepFace
├── vlm_output/              # Outputs from VLM
├── judge_output/            # Final judgments
├── daycare_result_pipeline.csv # Final logged output
├── images/                  # Place input images here
```

---

## Requirements

* Python 3.11+
* [Ollama](https://ollama.com) installed and running (for VLM and judge models)
* All dependencies in `requirements.txt`:

  ```text
  ultralytics
  deepface
  tf-keras
  opencv-python
  requests
  numpy
  pillow
  # For Gradio interface (optional): gradio
  ```
* Download or place your YOLO weights (e.g., `yolo11x.pt`) in the working directory.
* Place input images in the `images/` folder.

**For Gradio web UI (optional), see `daycare_pipeline_gradio.py` (not included here but supported by the requirements).**

---

## How to Run

### 1. **Start Ollama**

Make sure [Ollama](https://ollama.com) is running locally and you have pulled required models, e.g.:

```bash
ollama pull qwen2.5vl:7b
ollama pull deepseek-r1:1.5b
```

### 2. **Place Image**

Put your daycare image in the `images/` folder.

### 3. **Edit Pipeline Variables**

Open `daycare_pipeline.py` and edit:

```python
image_path = "images/your_image.jpg"
yolo_model = "yolo11x.pt"
deepface_backend = "retinaface"  # or your preferred backend
vlm_model = "qwen2.5vl:7b"
judge_model = "deepseek-r1:1.5b"
```

### 4. **Run the Pipeline**

```bash
python daycare_pipeline.py
```

Outputs will be saved in their respective folders, and you’ll see logs in the console.

---

## Output Details

* **YOLO+DeepFace JSON**: Counts, boxes, detected items, face emotion summaries.
* **VLM JSON**: Structured, sectioned report as per research-grade prompt (see `vlm.py`).
* **Judge JSON**: Final, evidence-based judgment, including discrepancies, safety assessment, and confidence metrics.

---

## Notes

* All steps are strictly **evidence-based**; no speculation is permitted by design.
* You may swap VLM and judge models (Gemma, Moondream, Qwen3, etc.) as long as they are supported by Ollama.
* Runs fully offline (except model downloads).

---

## License

Creative Commons Attribution-NonCommercial 4.0 International Public License

By using, copying, or modifying this code, you agree to the following terms:

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).  
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

**The licensor cannot revoke these freedoms as long as you follow the license terms.**

## Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.

## For commercial use:
- **A separate commercial license is required.**  
  Please contact the author, Partha Pratim Ray (parthapratimray1986@gmail.com), to obtain a commercial license and discuss terms.

## Notices:
- You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
- No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

---

To obtain a commercial license, please contact: parthapratimray1986@gmail.com


---
## 📖 Citation

If you use this repository or pipeline in your research, please cite:

**Partha Pratim Ray**  
*daycare_ollama_analysis: Local Multimodal Pipeline for Daycare Image Reasoning (YOLO, DeepFace, Ollama VLM, Judge LLM).*  
GitHub, 2025.  
[https://github.com/ParthaPRay/daycare_ollama_analysis](https://github.com/ParthaPRay/daycare_ollama_analysis)



```bibtex
@software{Ray2025DaycareOllamaAnalysis,
  author = {Partha Pratim Ray},
  title = {daycare_ollama_analysis: Local Multimodal Pipeline for Daycare Image Reasoning (YOLO, DeepFace, Ollama VLM, Judge LLM)},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/ParthaPRay/daycare_ollama_analysis}},
  note = {Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)},
  version = {v1.0},
  accessdate = {2025-06-22}
}

