from ska.sdc1.utils.image_utils import crop_to_training_area

# Challenge frequency bands
#
FREQS = [560, 1400, 9200]


def full_image_path(freq):
    return "data/images/{}mhz_1000h.fits".format(freq)


def sample_image_path(freq):
    return "data/sample_images/{}mhz_1000h_sample.fits".format(freq)


if __name__ == "__main__":
    """
    Helper script to generate small sample images from the full images, for testing.

    These are 1.5 times the size (2.25 times the area) of the training area.
    """

    for freq in FREQS:
        try:
            crop_to_training_area(
                full_image_path(freq), sample_image_path(freq), freq, 1.5
            )
        except FileNotFoundError:
            print(
                "Could not find image {}; run download_data.sh first".format(
                    full_image_path(freq)
                )
            )
