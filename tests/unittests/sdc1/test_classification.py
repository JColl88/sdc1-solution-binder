import os
import pytest

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

from ska.sdc1.utils.classification import SKLRegression, SKLClassification


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


        # Regression.
        #
        regressor = SKLRegression(algorithm=RandomForestRegressor, 
            regressor_args=[], regressor_kwargs={'random_state': 0})

        srl_df = regressor.train(train_srl_path, train_truth_cat_path, train_gaul_path, 
            regressand_col='b_maj_t')

        ## Assertion for xmatch score.
        assert regressor.last_xmatch_score == pytest.approx(20.05238, 1E-5)

        ## Assertion for validation score.
        assert regressor.validate(srl_df, regressand_col='b_maj_t', 
            validation_metric=mean_squared_error) == pytest.approx(0.065941, 1E-5)

        ## Assertion for testing score.
        test_y = regressor.test(test_srl_path, test_gaul_path)
        assert np.mean(test_y) == pytest.approx(1.12799, 1E-5)
        assert np.min(test_y) == pytest.approx(0.40677, 1E-5)
        assert np.max(test_y) == pytest.approx(4.69862, 1E-5)


        # Classification.
        #
        classifier = SKLClassification(algorithm=RandomForestClassifier, 
            classifier_args=[], classifier_kwargs={'random_state': 0})

        srl_df = classifier.train(train_srl_path, train_truth_cat_path, train_gaul_path, 
            regressand_col='class')

        ## Assertion for xmatch score.
        assert classifier.last_xmatch_score == pytest.approx(20.05238, 1E-5)

        ## Assertion for validation score.
        assert classifier.validate(srl_df, regressand_col='class',
            validation_metric=accuracy_score) == 1

        ## Assertion for testing score.
        test_y = classifier.test(test_srl_path, test_gaul_path)
        assert all(test_y)
        