import os

from ska.sdc1.models.image_2d import Image2d


class TestImage2d:
    def test_preprocess(self, images_dir, test_image_name, pb_image_name):
        train_file_expected = test_image_name[:-5] + "_train.fits"
        pbcor_file_expected = test_image_name[:-5] + "_pbcor.fits"
        test_image_path = os.path.join(images_dir, test_image_name)
        pb_image_path = os.path.join(images_dir, pb_image_name)

        # Before running preprocess, the segment and train files shouldn't exist
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False
        image_2d = Image2d(560, test_image_path, pb_image_path)
        image_2d.preprocess()

        # Check files have been created
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file))

        # Delete them again
        image_2d._delete_train()
        image_2d._delete_pb_corr()

        # Verify
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False
