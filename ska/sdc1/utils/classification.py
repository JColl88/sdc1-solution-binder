import os

from ska_sdc import Sdc1Scorer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from ska.sdc1.utils.bdsf_utils import cat_df_from_srl, load_truth_df, srl_gaul_df
from ska.sdc1.utils.columns import SRL_CAT_COLS, SRL_COLS_TO_DROP, SRL_NUM_COLS


class SKLRegression():
    def __init__(self, algorithm=RandomForestRegressor, regressor_args=[], 
    regressor_kwargs={}):
        """
        Sci-kit learn regression.
        
        Args:
            algorithm (`class`): SKL regressor class.
            regressor_args (`list`): regressor args.
            regressor_kwargs (`dict`): regressor kwargs.
        """
        self.regressor = algorithm(*regressor_args, **regressor_kwargs)


    def _preprocess_srl_df(self, srl_df, srl_cat_cols, srl_num_cols, srl_drop_cols):
        """
        Preprocess the source list DataFrame ready for model generation and 
        prediction.

        Args:
            srl_df (:obj:`pandas.DataFrame`): Source list.
            srl_cat_cols: (`list`) Categorical columns in source list.
            srl_num_cols: (`list`) Numerical columns in source list.
            srl_drop_cols: (`list`) Columns to exclude in source list.
        Returns:
            (:obj:`pandas.DataFrame`): Processed source list.
        """
        # Drop NaNs.
        #
        srl_df = srl_df.dropna()

        # Drop obsolete columns.
        #
        srl_df = srl_df.drop(srl_drop_cols, axis=1)

        # Encode categorical columns.
        #
        for col in srl_cat_cols:
            lbl = LabelEncoder()
            lbl.fit(list(srl_df[col].values.astype("str")))
            srl_df[col] = lbl.transform(list(srl_df[col].values.astype("str")))

        # Cast numerical columns as floats.
        #
        for col in srl_num_cols:
            srl_df[col] = srl_df[col].astype(float)

        return srl_df


    def _SKLfit(self, X, y):
        """
        Wrapper for SKL fit(). 
        
        Fits training input samples, X[X_columns], against target values, 
        y[Y_column].

        Args:
            X (:obj:`pandas.DataFrame`): Training input samples.
            y (:obj:`numpy.array`): Target values.
        Returns:
            None
        """              
        self.regressor.fit(X, y)


    def _SKLpredict(self, X):
        """
        Wrapper for SKL predict().

        Predicts values, y, from input samples X[X_columns].

        Args:
            X (:obj:`pandas.DataFrame`): Input samples.
        Returns:
            (:obj:`numpy.ndarray`): Predicted values.
        """
        return self.regressor.predict(X)


    def _SKLScore(self, X, y, metric):
        """
        Score the validation using the metric, <metric>.

        Args:
            X (:obj:`numpy.array`): Input samples.
            y (:obj:`numpy.array`): True values for X.
        Returns:
            (:obj:`float`): Score.
        """
        return np.sqrt(metric(X, y))


    def _xmatch_using_scorer(self, srl_path, truth_cat_path, freq):
        """
        Crossmatch source list against a truth catalogue using the SDC1 scorer. 

        Args:
            srl_path (`str`): Path to source list (.srl file).
            truth_cat_path (`str`): Path to truth catalogue.
            freq: (`int`): Frequency band (MHz).
        """
        sub_cat_df = cat_df_from_srl(srl_path)
        truth_cat_df = load_truth_df(truth_cat_path)

        truth_cat_df = truth_cat_df.dropna()
        
        scorer = Sdc1Scorer(sub_cat_df, truth_cat_df, freq)
        score = scorer.run(train=True, detail=True, mode=1)

        return score


    def test(self, srl_path, gaul_path, srl_cat_cols=SRL_CAT_COLS, 
    srl_num_cols=SRL_NUM_COLS, srl_drop_cols=SRL_COLS_TO_DROP, sl=np.s_[::]):
        """
        Predict the <regressand_column> for the test set source list using the 
        regressor.

        Args:
            srl_path (`str`): Path to source list (.srl file).
            gaul_path (`str`): Path to Gaussian list (.srl file).
            srl_cat_cols: (`list`) Categorical columns in source list.
            srl_num_cols: (`list`) Numerical columns in source list.
            srl_drop_cols: (`list`) Columns to exclude in source list.
            sl: (`slice`) Slice of source list to use for testing.
        Returns:
            (:obj:`numpy.ndarray`): Predicted values.
        """
        # Append the number of Gaussians to the source list DataFrame and take slice.
        #
        srl_df = srl_gaul_df(gaul_path, srl_path)

        # Preprocess source list, take slice, and construct test dataset.
        # 
        srl_df = self._preprocess_srl_df(srl_df, srl_cat_cols, srl_num_cols, 
            srl_drop_cols).iloc[sl, :]
        test_x = srl_df[srl_cat_cols+srl_num_cols]
        test_y = self._SKLpredict(test_x)

        return test_y


    def train(self, srl_path, truth_cat_path, gaul_path, regressand_col, freq=1400, 
        srl_cat_cols=SRL_CAT_COLS, srl_num_cols=SRL_NUM_COLS,
        srl_drop_cols=SRL_COLS_TO_DROP, sl=np.s_[::2]):
        """
        Train the regressor on <regressand_col> using a crossmatched PyBDSF 
        source list.

        Args:
            srl_path (`str`): Path to source list (.srl file).
            truth_cat_path (`str`): Path to truth catalogue.
            gaul_path (`str`): Path to Gaussian list (.srl file).
            regressand_col: (`str`): Regressand column name.
            freq: (`int`): Frequency band (MHz).
            srl_cat_cols: (`list`) Categorical columns in source list.
            srl_num_cols: (`list`) Numerical columns in source list.
            srl_drop_cols: (`list`) Columns to exclude in source list.
            sl: (`slice`) Slice of source list to use for training.
        Returns:
            srl_df (`str`): Crossmatched source list DataFrame used for training.
        """
        # Get crossmatched DataFrame using the SDC1 scorer.
        #
        xmatch = self._xmatch_using_scorer(srl_path, truth_cat_path, freq)
        xmatch_df = self._xmatch_using_scorer(srl_path, truth_cat_path, freq).match_df

        # Append the number of Gaussians to the source list DataFrame.
        #
        srl_df = srl_gaul_df(gaul_path, srl_path)

        # Reindex both source list and matched dataframes and add matched regressand 
        # column values to source list DataFrame.
        #
        # This leaves NaN values for unmatched sources in <srl_df>.
        #
        srl_df = srl_df.set_index("Source_id")
        xmatch_df = xmatch_df.set_index("id")
        srl_df[regressand_col] = xmatch_df[regressand_col]

        # Preprocess source list, take slice, and construct training dataset.
        # 
        srl_df = self._preprocess_srl_df(srl_df, srl_cat_cols, srl_num_cols, 
            srl_drop_cols).iloc[sl, :]
        train_x = srl_df[srl_cat_cols+srl_num_cols]
        train_y = srl_df[regressand_col].values

        self._SKLfit(train_x, train_y)

        return srl_df


    def validate(self, srl_df, regressand_col, srl_cat_cols=SRL_CAT_COLS, 
    srl_num_cols=SRL_NUM_COLS, srl_drop_cols=SRL_COLS_TO_DROP, sl=np.s_[1::2], 
    validation_metric=mean_squared_error):
        """
        Predict the <regressand_column> for the validation set source list using the 
        regressor.

        Args:
            srl_df (`str`): Path to source list (.srl file).
            regressand_col: (`str`): Regressand column name.
            srl_cat_cols: (`list`) Categorical columns in source list.
            srl_num_cols: (`list`) Numerical columns in source list.
            srl_drop_cols: (`list`) Columns to exclude in source list.
            sl: (`slice`) Slice of source list to use for validation.
            validation_metric: (`function`) SKL metric.
        Returns:
            (:obj:`float`): The validation score.
        """
        # Take slice and construct validation set.
        # 
        srl_df = srl_df.iloc[sl, :]
        validate_x = srl_df[srl_cat_cols+srl_num_cols]
        validate_y_true = srl_df[regressand_col].values

        validate_y = self._SKLpredict(validate_x)

        return self._SKLScore(validate_y, validate_y_true, metric=validation_metric)



