---
title: Image Captioning with BLIP
emoji: 📷
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Generate captions for uploaded images
---

# Image Captioning with BLIP

Generate captions for uploaded images or images embedded in a web page using Hugging Face's `Salesforce/blip-image-captioning-base` model.

## Setup

In PowerShell, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run downloads the BLIP model files from Hugging Face, so it requires an internet connection and can take a few minutes.

## Run the web app

```powershell
python image_captioning_app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

## Container deployment

Build and test the container locally with Docker:

```powershell
docker build -t image-captioning .
docker run --rm -p 7860:7860 image-captioning
```

For IBM Code Engine, build this directory through the Code Engine CLI, then deploy the resulting image with port `7860`. The application accepts Code Engine's `PORT` environment variable automatically.

## Caption images from a URL

```powershell
python automate_url_captioner.py https://en.wikipedia.org/wiki/IBM --limit 10
```

Captions are written to `captions.txt`. Use only web pages and images that you are permitted to download and process.
