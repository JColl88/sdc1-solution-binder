import os

from ska.sdc1.models.sdc1_image import Sdc1Image


class TestSdc1Image:
    def test_preprocess_simple_pb(
        self,
        images_dir,
        test_sdc1_image_image_small_name,
        test_sdc1_image_pb_image_name,
    ):
        """
        Test preprocess with a small test image, with a simple PB correction
        """
        train_file_expected = test_sdc1_image_image_small_name[:-5] + "_train.fits"
        pbcor_file_expected = test_sdc1_image_image_small_name[:-5] + "_pbcor.fits"
        test_image_path = os.path.join(images_dir, test_sdc1_image_image_small_name)
        pb_image_path = os.path.join(images_dir, test_sdc1_image_pb_image_name)

        # Before running preprocess, the segment and train files shouldn't exist
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False
        sdc1_image = Sdc1Image(560, test_image_path, pb_image_path)
        sdc1_image.preprocess()

        # Check files have been created
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file))

        # Delete them again
        sdc1_image._delete_train()
        sdc1_image._delete_pb_corr()

        # Verify
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False

    def test_preprocess_full_pb(
        self,
        images_dir,
        test_sdc1_image_image_large_name,
        test_sdc1_image_pb_image_name,
    ):
        """
        Test preprocess with a larger test image, employing a full PB correction
        """
        train_file_expected = test_sdc1_image_image_large_name[:-5] + "_train.fits"
        pbcor_file_expected = test_sdc1_image_image_large_name[:-5] + "_pbcor.fits"
        test_image_path = os.path.join(images_dir, test_sdc1_image_image_large_name)
        pb_image_path = os.path.join(images_dir, test_sdc1_image_pb_image_name)

        # Before running preprocess, the segment and train files shouldn't exist
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False
        sdc1_image = Sdc1Image(560, test_image_path, pb_image_path)
        sdc1_image.preprocess()

        # Check files have been created
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file))

        # Delete them again
        sdc1_image._delete_train()
        sdc1_image._delete_pb_corr()

        # Verify
        for expected_file in [train_file_expected, pbcor_file_expected]:
            assert os.path.isfile(os.path.join(images_dir, expected_file)) is False
