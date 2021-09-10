import pytest


@pytest.fixture
def cat_dir():
    return "tests/testdata/cat"


@pytest.fixture
def gaul_dir():
    return "tests/testdata/gaul"


@pytest.fixture
def images_dir():
    return "tests/testdata/images"


@pytest.fixture
def srl_dir():
    return "tests/testdata/srl"


@pytest.fixture
def test_classification_test_gaul_name():
    return "B1_1000h_0.05tr.pybdsm.gaul"


@pytest.fixture
def test_classification_test_srl_name():
    return "B1_1000h_0.05tr.pybdsm.srl"


@pytest.fixture
def test_classification_train_gaul_name():
    return "B1_1000h_0.3tr.pybdsm.gaul"


@pytest.fixture
def test_classification_train_truth_cat_name():
    return "truth_B1_0.3tr.cat"


@pytest.fixture
def test_classification_train_srl_name():
    return "B1_1000h_0.3tr.pybdsm.srl"


@pytest.fixture
def test_sdc1_image_image_large_name():
    return "B1_1000h_0.3tr.fits"


@pytest.fixture
def test_sdc1_image_image_small_name():
    return "B1_1000h_0.05tr.fits"


@pytest.fixture
def test_sdc1_image_pb_image_name():
    return "PrimaryBeam_B1.fits"


@pytest.fixture
def test_source_finder_image_name():
    return "B1_1000h_0.3tr.fits"
