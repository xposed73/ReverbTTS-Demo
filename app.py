"""Local Gradio playground for Software Mansion's Reverb-TTS."""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import logging
import secrets
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import soundfile as sf
from reverb_tts import TTS

ROOT = Path(__file__).resolve().parent
LANGUAGES = [("English", "en"), ("German", "de"), ("French", "fr"),
             ("Spanish", "es"), ("Italian", "it"), ("Polish", "pl"), ("Portuguese", "pt")]
PRESETS = ["de_female_standard", "de_male_standard", "en_female_narrator",
           "en_female_whisper", "en_male_angry", "en_male_narrator_1",
           "en_male_narrator_2", "es_male_standard_1", "es_male_standard_2",
           "fr_female_standard", "fr_male_standard", "it_male_expressive",
           "it_male_standard", "pl_male_standard", "pt_female_standard", "pt_male_standard"]


@lru_cache(maxsize=1)
def get_model():
    return TTS.from_pretrained("software-mansion/Reverb-TTS", variant="int8",
                               device=os.environ.get("REVERB_DEVICE") or None)


def generate(text, language, mode, preset, reference, transcript, reference_language,
             speed, steps, seed, progress=gr.Progress()):
    if not text or not text.strip():
        raise gr.Error("Enter some text to speak.")
    if len(text) > 5000:
        raise gr.Error("Please keep each generation under 5,000 characters.")
    if mode == "Clone a voice" and (not reference or not transcript.strip()):
        raise gr.Error("Add a reference recording and its exact transcript.")
    seed = int(seed)
    if seed < -1 or seed > 2**31 - 1:
        raise gr.Error("Use a seed between zero and 2147483647, or -1 for a random take.")
    actual_seed = secrets.randbelow(2**31) if seed == -1 else seed
    started = time.perf_counter()
    try:
        progress(0.05, desc="Loading model (first run downloads weights and audio codec)…")
        tts = get_model()
        progress(0.25, desc="Preparing voice…")
        if mode == "Clone a voice":
            voice = tts.voice(reference, transcript.strip(), language=reference_language)
        else:
            voice = tts.preset_voice(preset)
        progress(0.4, desc=f"Generating on {tts.device}…")
        audio = tts.generate(text.strip(), voice, language=language, speed=float(speed),
                             steps=int(steps), seed=actual_seed)
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)
        path = output_dir / f"reverb-{time.time_ns()}.wav"
        sf.write(path, audio, tts.sample_rate)
        elapsed = time.perf_counter() - started
        duration = len(audio) / tts.sample_rate
        return str(path), (f"{duration:.1f}s audio · {elapsed:.1f}s elapsed · "
                           f"device: {tts.device} · seed: {actual_seed}")
    except Exception as exc:
        logging.exception("Speech generation failed")
        raise gr.Error(f"Generation failed: {exc}") from exc


def build_ui():
    with gr.Blocks(title="Reverb TTS") as demo:
        gr.Markdown("# Reverb TTS\nTest preset voices or clone a voice from a short recording. "
                    "Runs locally; the first generation downloads the model and audio codec.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text to speak", lines=5,
                                  value="Hello! This is Reverb, running locally on my Mac. Let's try a new voice.")
                language = gr.Dropdown(LANGUAGES, value="en", label="Output language")
                mode = gr.Radio(["Preset voice", "Clone a voice"], value="Preset voice", label="Voice source")
                preset = gr.Dropdown(PRESETS, value="en_male_narrator_1", label="Preset voice")
                with gr.Group(visible=False) as clone_controls:
                    gr.Markdown("Use 3–10 seconds of clean speech and a matching transcript. "
                                "Only use voices you have permission to clone.")
                    reference = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference audio")
                    transcript = gr.Textbox(label="Exact reference transcript", lines=3)
                    reference_language = gr.Dropdown(LANGUAGES, value="en", label="Reference language")
                with gr.Accordion("Generation settings", open=False):
                    speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                    steps = gr.Slider(4, 32, value=8, step=1, label="Sampling steps")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                button = gr.Button("Generate speech", variant="primary")
            with gr.Column():
                output = gr.Audio(label="Generated speech", type="filepath", format="wav")
                status = gr.Textbox(label="Generation details", interactive=False)
                gr.Markdown("Download the WAV from the audio player. Recordings are saved in `outputs/`. "
                            "Model weights are licensed CC BY-NC 4.0 for non-commercial use.")
        mode.change(lambda value: (gr.update(visible=value == "Preset voice"),
                                    gr.update(visible=value == "Clone a voice")),
                    mode, [preset, clone_controls])
        button.click(generate, [text, language, mode, preset, reference, transcript,
                                reference_language, speed, steps, seed],
                     [output, status], api_name="generate", concurrency_limit=1)
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().queue(max_size=8).launch(server_name="127.0.0.1", server_port=args.port,
                                       theme=gr.themes.Soft())
