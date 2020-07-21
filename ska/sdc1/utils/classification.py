import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

truth_cat_paths = {
    560: 'path',
    1400: 'path',
    9200: 'path'
}

def train_classifier(freq, source_df, col_name, algorithm=RandomForestRegressor):
    '''
    Given a source dataframe, build a classifier which can predict 'col_name'
    in the truth catalogue (e.g. 'class', 'b_maj' or 'b_min')

    For development, will require a separate method which performs diagnostic
    '''
    # Get truth cat
    truth_cat = truth_cat_paths[freq]

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
