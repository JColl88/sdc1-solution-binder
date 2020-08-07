import os
from pathlib import Path

from ska_sdc import Sdc1Scorer
from sklearn.ensemble import RandomForestClassifier

from ska.sdc1.models.image_2d import Image2d
from ska.sdc1.utils.bdsf_utils import cat_df_from_srl_df, load_truth_df
from ska.sdc1.utils.classification import SKLClassification
from ska.sdc1.utils.source_finder import SourceFinder

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

score_report_paths = {
    560: "data/score/560mhz_score.txt",
    1400: "data/score/1400mhz_score.txt",
    9200: "data/score/9200mhz_score.txt",
}

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
    # 1) Create in-memory representation of image and preprocess
    image2d_list = []
    for freq, path in image_paths.items():
        new_image = Image2d(freq, path, pb_paths[freq])
        new_image.preprocess()
        image2d_list.append(new_image)

    # In data/images, we now have PB-corrected and training images for each band

    # 2) Source finding (training):
    sources_training = {}
    for image2d in image2d_list:
        source_finder = SourceFinder(image2d.train)
        sources_training[image2d.freq] = source_finder.run()
        # Remove temp files:
        source_finder.reset()

    # <Additional feature engineering of the source DataFrames can be performed here>

    # 3) Train classifiers for each frequency's source DataFrame:
    classifiers = {}
    for freq, source_train_df in sources_training.items():
        # Load truth catalogue for the training area into memory
        train_truth_cat_df = load_truth_df(train_truth_cat_paths[freq], skiprows=18)

        # Construct and train classifier
        classifier = SKLClassification(algorithm=RandomForestClassifier)
        srl_df = classifier.train(
            source_train_df, train_truth_cat_df, regressand_col="class", freq=freq
        )

        # Store model for prediction later
        classifiers[freq] = classifier

    # 4) Source finding (full):
    sources_full = {}
    for image2d in image2d_list:
        source_finder = SourceFinder(image2d.pb_corr_image)
        sources_full[image2d.freq] = source_finder.run()
        source_finder.reset()

    # 5) Source classification (full)
    for freq, source_df in sources_full.items():
        source_df["class"] = classifiers[freq].test(source_df)
        print(source_df["class"].value_counts())

    # 6) Create final catalogues and calculate scores
    for freq, source_df in sources_full.items():
        # Assemble submission and truth catalogues for scoring
        sub_cat_df = cat_df_from_srl_df(source_df, guess_class=False)
        truth_cat_df = load_truth_df(full_truth_cat_paths[freq], skiprows=0)

        # Calculate score
        scorer = Sdc1Scorer(sub_cat_df, truth_cat_df, freq)
        score = scorer.run(mode=0, train=False, detail=False)

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
