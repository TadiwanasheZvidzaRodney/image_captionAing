"""Gradio interface for generating captions with the BLIP base model."""

import os

import gradio as gr
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"


def load_captioning_model():
    """Load the processor and model once when the app starts."""
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


processor, model = load_captioning_model()


def caption_image(input_image: Image.Image | None) -> str:
    """Return a concise caption for an uploaded image."""
    if input_image is None:
        return "Upload an image to generate a caption."

    image = input_image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.Textbox(label="Caption"),
    title="Image Captioning",
    description="Upload an image to generate a BLIP caption.",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))