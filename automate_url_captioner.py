"""Generate BLIP captions for usable images found on a web page."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"
HEADERS = {"User-Agent": "Mozilla/5.0 (image-captioning-demo)"}


def get_image_url(element, page_url: str) -> str | None:
    """Return an absolute image URL from a page image element."""
    source = element.get("src") or element.get("data-src")
    if not source and element.get("srcset"):
        source = element["srcset"].split(",")[0].strip().split()[0]
    if not source or source.lower().split("?")[0].endswith(".svg"):
        return None
    return urljoin(page_url, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Web page whose images should be captioned")
    parser.add_argument("--output", default="captions.txt", help="Output text file")
    parser.add_argument("--limit", type=int, default=20, help="Maximum images to caption")
    arguments = parser.parse_args()

    page = requests.get(arguments.url, headers=HEADERS, timeout=20)
    page.raise_for_status()
    images = BeautifulSoup(page.text, "html.parser").find_all("img")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()

    saved = 0
    with Path(arguments.output).open("w", encoding="utf-8") as output_file:
        for element in images:
            if saved >= arguments.limit:
                break
            image_url = get_image_url(element, arguments.url)
            if not image_url:
                continue
            try:
                response = requests.get(image_url, headers=HEADERS, timeout=20)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                if image.width * image.height < 200:
                    continue
                inputs = processor(images=image, return_tensors="pt")
                generated = model.generate(**inputs, max_new_tokens=50)
                caption = processor.decode(generated[0], skip_special_tokens=True)
            except (requests.RequestException, UnidentifiedImageError, OSError) as error:
                print(f"Skipped {image_url}: {error}")
                continue

            output_file.write(f"{image_url}: {caption}\n")
            saved += 1
            print(f"[{saved}/{arguments.limit}] Caption saved")

    print(f"Wrote {saved} captions to {arguments.output}")


if __name__ == "__main__":
    main()