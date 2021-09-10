import os
from pathlib import Path

from ska.sdc1.utils.image_utils import crop_to_training_area

# Challenge frequency bands
#
FREQS = [560, 1400, 9200]

full_image_dir = os.path.join("data", "images")
sample_image_dir = os.path.join("data", "sample_images")


def full_image_path(freq):
    return os.path.join(full_image_dir, "{}mhz_1000h.fits".format(freq))


def sample_image_path(freq):
    return os.path.join(sample_image_dir, "{}mhz_1000h_sample.fits".format(freq))


if __name__ == "__main__":
    """
    Helper script to generate small sample images from the full images, for testing.

    These are 1.5 times the size (2.25 times the area) of the training area.
    """

    for freq in FREQS:
        try:
            Path(sample_image_dir).mkdir(parents=True, exist_ok=True)
            crop_to_training_area(
                full_image_path(freq), sample_image_path(freq), freq, 1.5
            )
        except FileNotFoundError:
            print(
                "Could not find image {}; run download_data.sh first".format(
                    full_image_path(freq)
                )
            )
