import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from ska_sdc import Sdc1Scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from ska.sdc1.models.sdc1_image import Sdc1Image
from ska.sdc1.utils.bdsf_utils import cat_df_from_srl_df, load_truth_df
from ska.sdc1.utils.classification import SKLClassification
from ska.sdc1.utils.source_finder import SourceFinder

# Challenge frequency bands
#
FREQS = [560, 1400, 9200]


# Input data paths; assumes defaults from download_data.sh
#
def image_path(freq):
    return os.path.join("data", "images", "{}mhz_1000h.fits".format(freq))


def pb_path(freq):
    return os.path.join("data", "images", "{}mhz_pb.fits".format(freq))


def train_truth_path(freq):
    return os.path.join("data", "truth", "{}mhz_truth_train.txt".format(freq))


def full_truth_path(freq):
    return os.path.join("data", "truth", "{}mhz_truth_full.txt".format(freq))


# Output data paths
#
def train_source_df_path(freq):
    return os.path.join("data", "sources", "{}mhz_sources_train.csv".format(freq))


def full_source_df_path(freq):
    return os.path.join("data", "sources", "{}mhz_sources_full.csv".format(freq))


def submission_df_path(freq):
    return os.path.join("data", "sources", "{}mhz_submission.csv".format(freq))


def model_path(freq):
    return os.path.join("data", "sources", "{}mhz_classifier.pickle".format(freq))


def score_report_path(freq):
    return os.path.join("data", "score", "{}mhz_score.txt".format(freq))


def write_df_to_disk(df, out_path):
    """ Helper function to write DataFrame df to a file at out_path"""
    out_dir = os.path.dirname(out_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    """
    Run through a simple analysis workflow to solve SDC1

    1) Preprocess images (correct PB) and crop out the training area for
        building ML model
    2) Find sources in the PB-corrected training images
    3) Train a classifier for each band to predict the class of each source
    4) Find sources in the full PB-corrected image
    5) Predict the class of each source
    6) Calculate the score for each image band, and write out a short report
    """
    time_0 = time()

    # 3) Train classifiers for each frequency's source DataFrame:
    print("\nStep 3: Training classifiers; elapsed: {:.2f}s".format(time() - time_0))
    classifiers = {}
    for freq in FREQS:
        # Load truth catalogue for the training area into memory
        train_truth_cat_df = load_truth_df(train_truth_path(freq), skiprows=18)

        source_train_df = pd.read_csv(train_source_df_path(freq))

        # Construct and train classifier
        classifier = SKLClassification(
            algorithm=RandomForestClassifier,
            classifier_kwargs={"n_estimators": 100, "class_weight": "balanced"},
        )
        srl_df = classifier.train(
            source_train_df, train_truth_cat_df, regressand_col="class_t", freq=freq
        )

        # Store model for prediction later
        classifiers[freq] = classifier

        # (Optional) Write model to disk; allows later loading without retraining.
        classifier.save_model(model_path(freq))

    # 5) Source classification (full)
    print("\nStep 5: Classification; elapsed: {:.2f}s".format(time() - time_0))
    sources_full = {}
    for freq in FREQS:
        source_df = pd.read_csv(full_source_df_path(freq))
        source_df["class"] = classifiers[freq].test(source_df)
        class_prob = classifiers[freq].predict_proba(source_df)
        print("Class probabilities")
        print(class_prob.shape)
        print(len(source_df.index))
        print(class_prob[:25])
        print(np.amax(class_prob, axis=1))

        source_df["class_prob"] = np.amax(class_prob, axis=1)

        sources_full[freq] = source_df

        write_df_to_disk(source_df, submission_df_path(freq))

    # 6) Create final catalogues and calculate scores
    print("\nStep 6: Final score; elapsed: {:.2f}s".format(time() - time_0))
    for freq, source_df in sources_full.items():
        # Assemble submission and truth catalogues for scoring
        sub_cat_df = cat_df_from_srl_df(source_df, guess_class=False)
        truth_cat_df = load_truth_df(full_truth_path(freq), skiprows=0)

        # Calculate score
        scorer = Sdc1Scorer(sub_cat_df, truth_cat_df, freq)
        score = scorer.run(mode=0, train=False, detail=True)

        # Write short score report:
        score_path = score_report_path(freq)
        score_dir = os.path.dirname(score_path)
        Path(score_dir).mkdir(parents=True, exist_ok=True)

        with open(score_path, "w+") as report:
            report.write(
                "Image: {}, frequency: {} MHz\n".format(image_path(freq), freq)
            )
            report.write("Score was {}\n".format(score.value))
            report.write("Number of detections {}\n".format(score.n_det))
            report.write("Number of matches {}\n".format(score.n_match))
            report.write(
                "Number of matches too far from truth {}\n".format(score.n_bad)
            )
            report.write("Number of false detections {}\n".format(score.n_false))
            report.write("Score for all matches {}\n".format(score.score_det))
            report.write("Accuracy percentage {}\n".format(score.acc_pc))
            report.write("Classification report: \n")
            report.write(
                classification_report(
                    score.match_df["class_t"],
                    score.match_df["class"],
                    labels=[1, 2, 3],
                    target_names=["1 (SS-AGN)", "2 (FS-AGN)", "3 (SFG)"],
                    digits=4,
                )
            )

    print("\nComplete; elapsed: {:.2f}s".format(time() - time_0))
