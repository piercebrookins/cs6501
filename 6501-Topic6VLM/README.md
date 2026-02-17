# Topic6VLM

Done. Yes, all the stuff. 🐶

This folder contains complete implementations for the tasks in `vlm.html`:

## Table of Contents

1. [Setup](#setup)
2. [Exercise 1: Vision-Language LangGraph Chat Agent](#exercise-1-vision-language-langgraph-chat-agent)
3. [Exercise 2: Video-Surveillance Agent](#exercise-2-video-surveillance-agent)
4. [Optional: Realtime Webcam Monitor](#optional-realtime-webcam-monitor)
5. [Project Structure](#project-structure)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llava
```

## Exercise 1: Vision-Language LangGraph Chat Agent

Multi-turn image Q&A app using:
- **LangGraph** for clean state flow
- **Ollama + LLaVA** for vision-language inference
- **Gradio** for UI

Run:

```bash
python exercise1_langgraph_vlm_chat.py
```

Features:
- Upload an image once and ask multiple follow-up questions
- Context-aware replies based on recent turns
- Automatic image downscaling for better performance

## Exercise 2: Video-Surveillance Agent

Analyzes a video by:
1. Extracting frames every N seconds (default: 2)
2. Asking LLaVA whether a person is visible
3. Reporting entry and exit timestamps

Run:

```bash
python exercise2_video_surveillance.py --video ./video.mp4 --interval 2 --frames-dir ./frames_out
```

Output includes:
- Frame-by-frame detection log
- Summary with entry/exit times in `MM:SS`

## Optional: Realtime Webcam Monitor

Bonus script for live webcam checks:

```bash
python exercise2_webcam_optional.py --camera 0 --interval 8.5
```

If a person appears, it prints `INTRUDER ALERT`.

## Project Structure

```text
.
├── exercise1_langgraph_vlm_chat.py
├── exercise2_video_surveillance.py
├── exercise2_webcam_optional.py
├── requirements.txt
├── README.md
├── exercise1_terminal_output.txt
├── exercise2_terminal_output.txt
└── topic6vlm/
    ├── __init__.py
    ├── config.py
    ├── image_utils.py
    ├── llava_client.py
    └── video_utils.py
```

