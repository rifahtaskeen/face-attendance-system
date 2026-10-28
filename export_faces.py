
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_lfw_people


def main():
    # Save images in the local project Images folder
    project_path = Path(__file__).resolve().parent
    output_dir = project_path / "Images"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download/load the LFW face dataset
    dataset = fetch_lfw_people(
        color=True,
        download_if_missing=True
    )

    number_of_images = min(100, len(dataset.images))

    for i in range(number_of_images):
        face = (dataset.images[i] * 255).clip(0, 255).astype(np.uint8)

        image = Image.fromarray(face)
        image = image.resize((216, 216), Image.Resampling.LANCZOS)

        image.save(output_dir / f"{i + 100}.png")

    print(f"Saved {number_of_images} images locally in: {output_dir}")


if __name__ == "__main__":
    main()
