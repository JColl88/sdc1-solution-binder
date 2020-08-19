import os

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from ska.sdc1.utils.bdsf_utils import load_truth_df, srl_gaul_df
from ska.sdc1.utils.classification import SKLClassification, SKLRegression


class TestClassification:
    def test_SKLRegression(
        self,
        cat_dir,
        gaul_dir,
        srl_dir,
        test_classification_train_gaul_name,
        test_classification_train_truth_cat_name,
        test_classification_train_srl_name,
        test_classification_test_gaul_name,
        test_classification_test_srl_name,
    ):
        train_srl_path = os.path.join(srl_dir, test_classification_train_srl_name)
        train_truth_cat_path = os.path.join(
            cat_dir, test_classification_train_truth_cat_name
        )
        train_gaul_path = os.path.join(gaul_dir, test_classification_train_gaul_name)
        test_srl_path = os.path.join(srl_dir, test_classification_test_srl_name)
        test_gaul_path = os.path.join(gaul_dir, test_classification_test_gaul_name)

        train_srl_df = srl_gaul_df(train_gaul_path, train_srl_path)
        train_truth_cat_df = load_truth_df(train_truth_cat_path)
        test_srl_df = srl_gaul_df(test_gaul_path, test_srl_path)

        # Regression.
        #
        regressor = SKLRegression(
            algorithm=RandomForestRegressor,
            regressor_args=[],
            regressor_kwargs={"random_state": 0},
        )

        srl_df = regressor.train(
            train_srl_df, train_truth_cat_df, regressand_col="b_maj_t",
        )

        # Assertion for xmatch score.
        assert regressor.last_xmatch_score == pytest.approx(36.63223, 1e-5)

        # Assertion for validation score.
        assert regressor.validate(
            srl_df, regressand_col="b_maj_t", validation_metric=mean_squared_error
        ) == pytest.approx(0.079704, 1e-5)

        # Assertion for testing score.
        test_y = regressor.test(test_srl_df)
        assert np.mean(test_y) == pytest.approx(1.57434, 1e-5)
        assert np.min(test_y) == pytest.approx(0.49190, 1e-5)
        assert np.max(test_y) == pytest.approx(7.61947, 1e-5)

        # Classification.
        #
        classifier = SKLClassification(
            algorithm=RandomForestClassifier,
            classifier_args=[],
            classifier_kwargs={"random_state": 0},
        )

        srl_df = classifier.train(
            train_srl_df, train_truth_cat_df, regressand_col="class",
        )

        # Assertion for xmatch score.
        assert classifier.last_xmatch_score == pytest.approx(36.63223, 1e-5)

        # Assertion for validation score.
        assert (
            classifier.validate(
                srl_df, regressand_col="class", validation_metric=accuracy_score
            )
            == 1
        )

        # Assertion for testing score.
        test_y = classifier.test(test_srl_df)
        proba_y = classifier.predict_proba(test_srl_df).ravel()
        assert all(test_y)
        assert all(proba_y)
