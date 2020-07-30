import pytest


@pytest.fixture
def images_dir():
    return "tests/testdata/images"


@pytest.fixture
def test_image_small():
    return "B1_1000h_0.05tr.fits"


@pytest.fixture
def test_image_large():
    return "B1_1000h_0.3tr.fits"


@pytest.fixture
def pb_image_name():
    return "PrimaryBeam_B1.fits"
