from ska_sdc import Sdc1Scorer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ska.sdc1.utils.bdsf_utils import cat_df_from_srl, load_truth_df, srl_gaul_df
from ska.sdc1.utils.columns from utils import SRL_CAT_COLS, SRL_COLS_TO_DROP, SRL_NUM_COLS


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
        self.regressor.fit(train_x, train_y)


    def _SKLpredict(self, X):
        """
        Wrapper for SKL predict().

        Predicts values, y, from input samples X[X_columns].

        Args:
            X (:obj:`pandas.DataFrame`): Input samples.
        Returns:
            (:obj:`numpy.ndarray`): Predicted values.
        """
        return regressor.predict(test_x)


    def _xmatch_using_scorer(self, srl_path, truth_cat_path):
        """
        Crossmatch source list against a truth catalogue using the SDC1 scorer. 

        Args:
            srl_path (`str`): Path to source list (.srl file).
            truth_cat_path (`str`): Path to truth catalogue.
            freq: (`int`): Frequency band (MHz).
        """
        sub_cat_df = cat_df_from_srl(srl_path)
        truth_cat_df = load_truth_df(truth_cat_path)

        scorer = Sdc1Scorer(sub_cat_df, truth_cat_df, freq)
        score = scorer.run(train=True, detail=True, mode=1)

        return score.match_df


    def train(self, srl_path, truth_cat_path, gaul_path, regressand_col, freq=1400, 
    srl_cat_cols=SRL_CAT_COLS, srl_num_cols=SRL_NUM_COLS,
    srl_drop_cols=SRL_COLS_TO_DROP):
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
        Returns:
            None
        """
        # Get crossmatched DataFrame using the SDC1 scorer.
        #
        xmatch_df = _xmatch_using_scorer(srl_path, truth_cat_path, freq)

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

        # Remove unmatched sources and obsolete columns, encode categorical data,
        # perform any necessary type casting and fit the training set.
        # 
        srl_df = self._preprocess_srl_df(srl_df, srl_cat_cols, srl_num_cols, 
            srl_drop_cols)
        train_x = srl_df[srl_cat_cols+srl_num_cols]
        train_y = srl_df[regressand_col].values

        self._SKLfit(train_x, train_y)


    def predict(self, srl_path, gaul_path, srl_cat_cols=SRL_CAT_COLS, 
    srl_num_cols=SRL_NUM_COLS, srl_drop_cols=SRL_COLS_TO_DROP):
        """
        Predict the <regressand_column> for a source list using the regressor.

        Args:
            srl_path (`str`): Path to source list (.srl file).
            gaul_path (`str`): Path to Gaussian list (.srl file).
            srl_cat_cols: (`list`) Categorical columns in source list.
            srl_num_cols: (`list`) Numerical columns in source list.
            srl_drop_cols: (`list`) Columns to exclude in source list.
        Returns:
            (:obj:`numpy.ndarray`): Predicted values.
        """
        # Append the number of Gaussians to the source list DataFrame and take a 
        # copy of the full DataFrame.
        #
        srl_df = srl_gaul_df(gaul_path, srl_path)

        # Remove unmatched sources and obsolete columns, encode categorical data,
        # perform any necessary type casting and fit the full dataset.
        # 
        srl_df = self._preprocess_srl_df(srl_df, srl_cat_cols, srl_num_cols, 
            srl_drop_cols)
        test_x = srl_df[srl_cat_cols+srl_num_cols]
        test_y = self._SKLpredict(test_x)

        return test_y
