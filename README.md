# Reverb TTS · Gradio Playground

A local web UI for [Software Mansion’s Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS). Generate speech with a preset voice or clone a voice from a short recording—then listen to and download the result in your browser.

Built with Gradio and the official `reverb-tts` Python package. Speech generation runs on the computer hosting the app.

## Features

- **16 preset voices**, including narration, whispering, and expressive voices.
- **Voice cloning** from an uploaded file or microphone recording.
- **Seven languages:** English, German, French, Spanish, Italian, Polish, and Portuguese.
- **Cross-language generation:** the reference recording and output can use different supported languages.
- Controls for speaking speed, sampling steps, and random seed.
- Downloadable **24 kHz WAV files**, also saved locally in `outputs/`.
- Automatic model downloads and hardware selection: NVIDIA CUDA → Apple MPS → CPU.

## Quick start

### 1. Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package and environment manager.

This project pins Python **3.11** in `.python-version`. With its default settings, uv can install that Python version if it is missing.

### 2. Install dependencies

Download or clone this project, open a terminal in its folder, and run:

```bash
uv sync --locked
```

This creates a local `.venv` and installs the versions recorded in `uv.lock`.

### 3. Start the app

```bash
uv run app.py
```

Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

Choose a preset voice, enter some text, and click **Generate speech**. To stop the server, press `Ctrl+C` in its terminal.

> **New users do not need to download weights manually.** The first time you click Generate speech, the app downloads the model and audio codec automatically. Keep the server running while it finishes.

## What happens on the first run?

| Stage | What happens |
| --- | --- |
| `uv sync --locked` | Installs Python dependencies into `.venv`. |
| `uv run app.py` | Starts the Gradio UI; the speech model is not loaded yet. |
| First generation | Downloads Reverb’s int8 weights, tokenizer, audio codec, and the selected preset if needed; loads the model; generates audio. |
| Later generations | Reuses the model already loaded in memory. Additional presets may need a small download. |
| After restarting | Reuses downloaded files from disk and loads the model into memory on the next generation. |

Allow several gigabytes of free disk space for dependencies, model files, and generated audio. The first generation needs internet access and can take several minutes, depending on your connection and hardware.

Model files are stored in the Hugging Face cache, normally `~/.cache/huggingface/hub/`. You can set `HF_HOME` before starting the app to choose a different cache location.

If the app is hosted on another computer, downloads and inference happen on **that computer**, not in each visitor’s browser. This project starts as a localhost-only app by default.

## Using the UI

### Generate with a preset

1. Enter the text you want spoken.
2. Choose its **Output language**.
3. Leave **Voice source** set to **Preset voice** and choose a voice.
4. Click **Generate speech**.
5. Listen in the audio player or use its download control to save the WAV.

Preset names start with their reference language, such as `en_male_narrator_1` or `fr_female_standard`. You can use any preset with any supported output language.

### Clone a voice

1. Set **Voice source** to **Clone a voice**.
2. Upload a reference recording or record one with your microphone.
3. Enter the **Exact reference transcript**, matching the spoken words.
4. Set **Reference language** to the recording’s language.
5. Enter your new text, choose its output language, and generate.

For best results, use **3–10 seconds of clean speech from one speaker**. Avoid background music, strong echo, and overlapping speakers. The reference influences the generated voice’s delivery and recording characteristics.

Only clone voices you have permission to use.

### Generation settings

| Setting | Default | Description |
| --- | --- | --- |
| Speed | `1.0` | Range `0.5–2.0`. Values above one produce faster speech; values below one produce slower speech. |
| Sampling steps | `8` | Range `4–32`. Controls the number of sampling passes. More steps add compute time and do not guarantee better results. |
| Seed | `-1` | Draws a random seed. Use a fixed value from `0` to `2147483647` to help reproduce a take with the same inputs and settings. |

Each request accepts up to **5,000 characters**. The underlying package handles text normalization and splits long text automatically. Generation details show the audio duration, elapsed time, device, and actual seed used. First-run elapsed time includes downloading and loading.

## Hardware and configuration

The app selects the first available device in this order:

1. NVIDIA GPU with CUDA available to PyTorch.
2. Apple GPU through Metal Performance Shaders (MPS).
3. CPU.

Preset generation and the reference-audio path have been smoke-tested on an Apple Silicon Mac using MPS. CUDA and CPU performance will depend on your system; CPU inference can be slower.

The app downloads the `int8` model variant. This describes the stored weights, not a guarantee of int8 execution or a particular runtime memory footprint.

### Use a different port

```bash
uv run app.py --port 7861
```

Then open [http://127.0.0.1:7861](http://127.0.0.1:7861).

### Force CPU execution

On macOS or Linux:

```bash
REVERB_DEVICE=cpu uv run app.py
```

In Windows PowerShell:

```powershell
$env:REVERB_DEVICE = "cpu"
uv run app.py
```

You can also set `REVERB_DEVICE` to `mps` or `cuda` when that device is available. Restart the server after changing environment variables.

## Files and storage

```text
.
├── app.py             # Gradio interface and inference callbacks
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Resolved dependency versions
├── .python-version    # Python version used by uv
├── .gitignore
├── README.md
└── outputs/           # Generated WAV files; created on first success
```

Generated audio is saved as `outputs/reverb-<timestamp>.wav`. Files are not automatically deleted; remove old recordings when you no longer need them. Gradio may also create temporary copies for uploads and playback.

The server binds to `127.0.0.1` and does not create a public Gradio share link. Inference runs locally, while model and preset downloads contact Hugging Face. Requests are processed one at a time to limit simultaneous model memory use.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| First generation takes a long time | Allow the model and codec downloads to finish. Check the terminal for progress or network errors. Later generations reuse the cached files. |
| Download fails | Check your internet connection and access to Hugging Face, then retry. The app defaults to HTTP downloads with `HF_HUB_DISABLE_XET=1`. |
| Port 7860 is already in use | Start with `uv run app.py --port 7861`. |
| GPU error or insufficient memory | Close other GPU-heavy apps, shorten the input, or restart with `REVERB_DEVICE=cpu`. |
| Microphone recording does not work | Allow microphone access in your browser, or upload an audio file instead. |
| Cloned speech sounds noisy or inaccurate | Use a cleaner, shorter reference and check that its transcript matches word for word. |
| Speech in an unsupported language sounds wrong | Choose one of the seven supported languages and provide text in that language. |
| A generation fails | Read the message in the UI and the traceback in the server terminal. |

## Model and licensing

This is a Gradio wrapper around Reverb-TTS, not the upstream model repository.

- [Model card and weights](https://huggingface.co/software-mansion/Reverb-TTS)
- [Upstream source code](https://github.com/software-mansion-labs/Reverb-TTS)
- [Upstream usage guide](https://github.com/software-mansion-labs/Reverb-TTS/blob/main/docs/usage.md)

Reverb’s **model weights are licensed under CC BY-NC 4.0**, for non-commercial use with attribution. The upstream Reverb-TTS code is Apache 2.0. Refer to the upstream licenses for their full terms.
