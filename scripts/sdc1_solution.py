import os
from pathlib import Path
from time import time

from ska_sdc import Sdc1Scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from ska.sdc1.models.sdc1_image import Sdc1Image
from ska.sdc1.utils.bdsf_utils import cat_df_from_srl_df, load_truth_df
from ska.sdc1.utils.classification import SKLClassification
from ska.sdc1.utils.source_finder import SourceFinder

# Input data paths
image_paths = {
    560: "data/images/560mhz_1000h.fits",
    1400: "data/images/1400mhz_1000h.fits",
    9200: "data/images/9200mhz_1000h.fits",
}

pb_paths = {
    560: "data/images/560mhz_pb.fits",
    1400: "data/images/1400mhz_pb.fits",
    9200: "data/images/9200mhz_pb.fits",
}

train_truth_cat_paths = {
    560: "data/truth/560mhz_truth_train.txt",
    1400: "data/truth/1400mhz_truth_train.txt",
    9200: "data/truth/9200mhz_truth_train.txt",
}

full_truth_cat_paths = {
    560: "data/truth/560mhz_truth_full.txt",
    1400: "data/truth/1400mhz_truth_full.txt",
    9200: "data/truth/9200mhz_truth_full.txt",
}

# Output data paths
train_source_df_paths = {
    560: "data/sources/560mhz_sources_train.csv",
    1400: "data/sources/1400mhz_sources_train.csv",
    9200: "data/sources/9200mhz_sources_train.csv",
}

full_source_df_paths = {
    560: "data/sources/560mhz_sources_full.csv",
    1400: "data/sources/1400mhz_sources_full.csv",
    9200: "data/sources/9200mhz_sources_full.csv",
}

class_df_paths = {
    560: "data/sources/560mhz_class.csv",
    1400: "data/sources/1400mhz_class.csv",
    9200: "data/sources/9200mhz_class.csv",
}

submission_df_paths = {
    560: "data/sources/560mhz_submission.csv",
    1400: "data/sources/1400mhz_submission.csv",
    9200: "data/sources/9200mhz_submission.csv",
}

model_paths = {
    560: "data/models/560mhz_classifier.sav",
    1400: "data/models/1400mhz_classifier.sav",
    9200: "data/models/9200mhz_classifier.sav",
}

score_report_paths = {
    560: "data/score/560mhz_score.txt",
    1400: "data/score/1400mhz_score.txt",
    9200: "data/score/9200mhz_score.txt",
}


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
    6) Calculate the score for each image band
    """
    time_0 = time()
    times = []
    # 1) Create in-memory representation of image and preprocess
    print("\nStep 1: Preprocessing; elapsed: {:.2f}s".format(time() - time_0))
    sdc1_image_list = []
    for freq, path in image_paths.items():
        new_image = Sdc1Image(freq, path, pb_paths[freq])
        new_image.preprocess()
        sdc1_image_list.append(new_image)

    # In data/images, we now have PB-corrected and training images for each band

    # 2) Source finding (training):
    print("\nStep 2: Source finding (train); elapsed: {:.2f}s".format(time() - time_0))
    sources_training = {}
    for sdc1_image in sdc1_image_list:
        source_finder = SourceFinder(sdc1_image.train)
        sl_df = source_finder.run()
        sources_training[sdc1_image.freq] = sl_df

        # (Optional) Write source list DataFrame to disk
        sl_train_path = train_source_df_paths[sdc1_image.freq]
        write_df_to_disk(sl_df, sl_train_path)

        # Remove temp files:
        source_finder.reset()

    # <Additional feature engineering of the source DataFrames can be performed here>

    # 3) Train classifiers for each frequency's source DataFrame:
    print("\nStep 3: Training classifiers; elapsed: {:.2f}s".format(time() - time_0))
    classifiers = {}
    for freq, source_train_df in sources_training.items():
        # Load truth catalogue for the training area into memory
        train_truth_cat_df = load_truth_df(train_truth_cat_paths[freq], skiprows=18)

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

    # 4) Source finding (full):
    sources_full = {}
    print("\nStep 4: Source finding (full); elapsed: {:.2f}s".format(time() - time_0))
    for sdc1_image in sdc1_image_list:
        source_finder = SourceFinder(sdc1_image.pb_corr_image)
        sl_df = source_finder.run()
        sources_full[sdc1_image.freq] = sl_df

        # (Optional) Write source list DataFrame to disk
        sl_full_path = full_source_df_paths[sdc1_image.freq]
        write_df_to_disk(sl_df, sl_full_path)

        # Remove temp files:
        source_finder.reset()

    # 5) Source classification (full)
    print("\nStep 5: Classification; elapsed: {:.2f}s".format(time() - time_0))
    for freq, source_df in sources_full.items():
        source_df["class"] = classifiers[freq].test(source_df)
        source_df["class_prob"] = classifiers[freq].predict_proba(source_df)

        write_df_to_disk(source_df, submission_df_paths[freq])

    # 6) Create final catalogues and calculate scores
    print("\nStep 6: Final score; elapsed: {:.2f}s".format(time() - time_0))
    for freq, source_df in sources_full.items():
        # Assemble submission and truth catalogues for scoring
        sub_cat_df = cat_df_from_srl_df(source_df, guess_class=False)
        truth_cat_df = load_truth_df(full_truth_cat_paths[freq], skiprows=0)

        # Calculate score
        scorer = Sdc1Scorer(sub_cat_df, truth_cat_df, freq)
        score = scorer.run(mode=0, train=False, detail=True)

        # Write short score report:
        score_path = score_report_paths[freq]
        score_dir = os.path.dirname(score_path)
        Path(score_dir).mkdir(parents=True, exist_ok=True)

        with open(score_path, "w+") as report:
            report.write(
                "Image: {}, frequency: {} MHz\n".format(image_paths[freq], freq)
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
