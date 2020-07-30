import os

from ska.sdc1.models.image_2d import Image2d


class TestImage2d:
    def test_preprocess_simple_pb(self, images_dir, test_image_2d_image_small_name, 
    test_image_2d_pb_image_name):
        """
        Test preprocess with a small test image, with a simple PB correction
        """
        train_file_expected = test_image_2d_image_small_name[:-5] + "_train.fits"
        pbcor_file_expected = test_image_2d_image_small_name[:-5] + "_pbcor.fits"
        test_image_path = os.path.join(images_dir, test_image_2d_image_small_name)
        pb_image_path = os.path.join(images_dir, test_image_2d_pb_image_name)

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

    def test_preprocess_full_pb(self, images_dir, test_image_2d_image_large_name, 
    test_image_2d_pb_image_name):
        """
        Test preprocess with a larger test image, employing a full PB correction
        """
        train_file_expected = test_image_2d_image_large_name[:-5] + "_train.fits"
        pbcor_file_expected = test_image_2d_image_large_name[:-5] + "_pbcor.fits"
        test_image_path = os.path.join(images_dir, test_image_2d_image_large_name)
        pb_image_path = os.path.join(images_dir, test_image_2d_pb_image_name)

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
