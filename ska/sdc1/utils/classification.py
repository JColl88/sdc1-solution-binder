import time

from ska_sdc import Sdc1Scorer
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ska.sdc1.utils.bdsf_utils import cat_df_from_srl, load_truth_df, srl_gaul_df
from ska.sdc1.utils.columns from utils import SRL_CAT_COLS, SRL_COLS_TO_DROP, SRL_NUM_COLS


'''def get_xmatch_df(cat_df, truth_df, freq):
    """
    Get the matched sources DataFrame for the passed catalogue DataFrame. Uses the
    scorer to yield the match catalogue.

    Args:
        cat_df (:obj:`pandas.DataFrame`): Catalogue DataFrame to obtain match_df for
        truth_df (:obj:`pandas.DataFrame`): Truth catalogue DataFrame
        freq: (`int`): Frequency band
    """
    scorer = Sdc1Scorer(cat_df, truth_df, freq)
    score = scorer.run(train=True, detail=True, mode=1)

    return score.match_df'''


def SKLRegression(Model):
    """ 
        Base class for regression using scikit-learn. 
    """
    def __init__(self, srl_path, truth_cat_path, gaul_path, freq=1400):
        """
        Args:
            srl_path (`str`): Path to source list (.srl file).
            truth_cat_path (`str`): Path to truth catalogue.
            gaul_path (`str`): Path to gaussian list (.srl file).
            freq: (`int`): Frequency band (MHz).
        """
        self.srl_path = srl_path
        self.truth_cat_path = truth_cat_path
        self.gaul_path = gaul_path
        self.freq = freq
        self.regressor = None


    def _fit(self, X, y):
        """
        Fits training input samples, X, against target values, y.

        Args:
            X (:obj:`pandas.DataFrame`): Training input samples.
            y (:obj:`numpy.array`): Target values.
        Returns:
            None
        """
        self.regressor.fit(train_x, train_y)


    def _predict(self, X):
        """
        Predicts values, y, from input samples X.

        Args:
            X (:obj:`pandas.DataFrame`): Input samples.
        Returns:
            (:obj:`numpy.ndarray`): Predicted values.
        """
        return regressor.predict(test_x)


    def _preprocess_srl_df(self, srl_df):
        """
        Preprocess the source list DataFrame for model generation and prediction.

        Args:
            srl_df (:obj:`pandas.DataFrame`): Source list.
        Returns:
            (:obj:`pandas.DataFrame`): Processed source list.
        """

        # Drop NaNs.
        #
        srl_df = srl_df.dropna()

        # Drop obsolete columns.
        #
        srl_df = srl_df.drop(SRL_COLS_TO_DROP, axis=1)

        # Encode categorical columns.
        #
        for col in SRL_CAT_COLS:
            lbl = LabelEncoder()
            lbl.fit(list(srl_df[col].values.astype("str")))
            srl_df[col] = lbl.transform(list(srl_df[col].values.astype("str")))

        # Cast numerical columns as floats.
        #
        for col in SRL_NUM_COLS:
            srl_df[col] = srl_df[col].astype(float)

        return srl_df


    def run(column):
        """
        Given a PyBDSF source list, create an SDC1 catalogue and obtain the
        match catalogue from the score pipeline, then use the truth catalogue's size
        values together with the source list properties to build a predictive model.

        Args:
            column (`str`): b_maj_t or b_min_t
        """

        # Crossmatch the dataframe.
        #
        sub_cat_df = cat_df_from_srl(self.srl_path)
        truth_cat_df = load_truth_df(self.truth_path)
        xmatch_df = get_match_df(sub_cat_df, truth_cat_df, self.freq)

        # Append the number of gaussians to the source list dataframe, and set the 
        # [Source_id] and [id] columns as the source list and truth catalogue 
        # dataframe indexes respectively.
        #
        srl_df = srl_gaul_df(gaul_path, srl_path)

        srl_df_full = srl_df.copy()
        
        srl_df = srl_df.set_index("Source_id")
        match_df = match_df.set_index("id")

        # Set the true size ID for the source list & drop resulting NaN values 
        # from unmatched sources.
        #
        srl_df[size_col] = match_df[size_col]
        srl_df = self._preprocess_srl_df(srl_df)

        srl_df_full = self._preprocess_srl_df(srl_df_full)
 
        train_x = srl_df[SRL_CAT_COLS + SRL_NUM_COLS]
        train_y = srl_df[size_col].values
        self._fit(train_x, train_y)

        test_x = srl_df_full[SRL_CAT_COLS + SRL_NUM_COLS]
        test_y = self._predict(test_x)

        return test_y
    
    
def SKLRandomForestRegression(SKLRegression):
    def __init__(self, srl_path, truth_cat_path, gaul_path, freq=1400):
        """
        Args:
            srl_path (`str`): Path to source list (.srl file).
            truth_cat_path (`str`): Path to truth catalogue.
            gaul_path (`str`): Path to gaussian list (.srl file).
            freq: (`int`): Frequency band (MHz).
            algorithm: (`class`) sklearn algorithm class
        """
        super().__init__(srl_path, truth_cat_path, gaul_path, freq)
        self.regressor = RandomForestRegressor()









def train_classifier(freq, source_df, col_name, algorithm=RandomForestRegressor):
    '''
    Given a source dataframe, build a classifier which can predict 'col_name'
    in the truth catalogue (e.g. 'class', 'b_maj' or 'b_min')

    For development, will require a separate method which performs diagnostic
    '''


    # (See predict_size.py for example workflow)
    # Get match catalog (by running scorer, or approximately by
    # astropy.match_catalog_to_sky)

    # Use the Source_id / id columns as the DataFrame index for cross-mapping

    # Get train_x, train_y
    train_x = np.array([])
    train_y = np.array([])

    # Train algorithm
    predictor = algorithm(random_state=0)
    predictor.fit(train_x, train_y)

    return predictor
