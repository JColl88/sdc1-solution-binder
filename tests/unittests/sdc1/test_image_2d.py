import os

from ska.sdc1.models.image_2d import Image2d


class TestImage2d:
    def test_preprocess(self, images_dir, test_image_name, pb_image_name):
        train_file_expected = test_image_name[:-5] + "_train.fits"
        test_image_path = os.path.join(images_dir, test_image_name)
        pb_image_path = os.path.join(images_dir, pb_image_name)

        # Before running preprocess, the segment and train files shouldn't exist
        assert os.path.isfile(os.path.join(images_dir, train_file_expected)) is False
        image_2d = Image2d(560, test_image_path, pb_image_path, prep=False)
        image_2d.preprocess(overwrite=True)

        # Check files have been created
        assert os.path.isfile(os.path.join(images_dir, train_file_expected))

        # Delete them again
        image_2d._delete_train()

        # Verify
        assert os.path.isfile(os.path.join(images_dir, train_file_expected)) is False
