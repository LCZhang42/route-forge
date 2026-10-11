"""Legacy Gradio client for the Modal-hosted API.

This interface never loads or downloads model weights locally. The React app is
the intended product UI; Gradio remains a small debugging client.
"""

from __future__ import annotations

import json
import os
from urllib import error, request

import gradio as gr


API_URL = os.environ.get("CLIMB_API_URL", "").rstrip("/")
GRADES = ["6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+"]


def generate_remote(
    grade: str,
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    temperature: float,
) -> str:
    if not API_URL:
        return "Set `CLIMB_API_URL` to the deployed Modal API URL."

    payload = json.dumps(
        {
            "grade": grade,
            "start_hold": [start_x, start_y],
            "end_hold": [end_x, end_y],
            "candidate_count": 3,
            "temperature": temperature,
        }
    ).encode("utf-8")
    http_request = request.Request(
        f"{API_URL}/api/v1/routes/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        return f"Modal API returned HTTP {exc.code}."
    except error.URLError:
        return "Could not reach the Modal API. Check `CLIMB_API_URL`."

    lines = [f"Model: {result['model_version']}"]
    for index, candidate in enumerate(result["candidates"], start=1):
        labels = [f"{chr(ord('A') + x)}{y}" for x, y in candidate["path"]]
        warning_text = "; ".join(candidate["warnings"]) or "none"
        lines.extend(
            [
                "",
                f"Candidate {index}: {' -> '.join(labels)}",
                f"Warnings: {warning_text}",
            ]
        )
    return "\n".join(lines)


def create_interface() -> gr.Blocks:
    with gr.Blocks(title="MoonBoard Route Debug Client") as interface:
        gr.Markdown(
            "# MoonBoard Route Debug Client\n"
            "Model training and inference run on Modal; no weights are stored locally."
        )
        with gr.Row():
            grade = gr.Dropdown(GRADES, value="7A", label="Grade")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
        with gr.Row():
            start_x = gr.Number(value=5, precision=0, label="Start X")
            start_y = gr.Number(value=4, precision=0, label="Start Y")
            end_x = gr.Number(value=5, precision=0, label="End X")
            end_y = gr.Number(value=17, precision=0, label="End Y")
        generate = gr.Button("Generate on Modal", variant="primary")
        output = gr.Textbox(label="Candidates", lines=12)
        generate.click(
            generate_remote,
            inputs=[grade, start_x, start_y, end_x, end_y, temperature],
            outputs=output,
        )
    return interface


if __name__ == "__main__":
    create_interface().launch(server_name="127.0.0.1", server_port=7861, share=False)
