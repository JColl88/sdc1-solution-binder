import os
import pytest

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

from ska.sdc1.utils.classification import SKLRegression


class TestClassification:
    def test_SKLRegression(self, cat_dir, gaul_dir, srl_dir, 
    test_classification_train_gaul_name, test_classification_train_truth_cat_name, 
    test_classification_train_srl_name, test_classification_test_gaul_name, 
    test_classification_test_srl_name):
        train_srl_path = os.path.join(srl_dir, test_classification_train_srl_name)
        train_truth_cat_path = os.path.join(
            cat_dir, test_classification_train_truth_cat_name)
        train_gaul_path = os.path.join(gaul_dir, test_classification_train_gaul_name)
        test_srl_path = os.path.join(srl_dir, test_classification_test_srl_name)
        test_gaul_path = os.path.join(gaul_dir, test_classification_test_gaul_name)

        regressor = SKLRegression(algorithm=RandomForestRegressor, 
            regressor_args=[], regressor_kwargs={'random_state': 0})

        srl_df = regressor.train(train_srl_path, train_truth_cat_path, train_gaul_path, 
            regressand_col='b_maj_t')

        # Assertion for validation score.
        #
        assert regressor.validate(srl_df, regressand_col='b_maj_t') == \
            pytest.approx(0.25679, 1E-5)

        # Assertion for testing score.
        #
        test_y = regressor.test(test_srl_path, test_gaul_path)
        assert np.mean(test_y) == pytest.approx(1.12799, 1E-5)
        assert np.min(test_y) == pytest.approx(0.40677, 1E-5)
        assert np.max(test_y) == pytest.approx(4.69862, 1E-5)
        