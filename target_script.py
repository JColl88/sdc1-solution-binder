# Dir structure is:
# ska/
#   <analysis_code>
# tests/
#   <unit_tests>
# images/
#   560mhz_1000h.fits
#   1400mhz_1000h.fits
#   9200mhz_1000h.fits

from ska.sdc1.models.image_2d import Image2d
from ska.sdc1.models.image_3d import Image3d
from ska.sdc1.utils.classification import train_classifier
from ska.sdc1.utils.postprocess import dedupe_sources
from ska.sdc1.utils.source_finder import SourceFinder

# TODO: (Optional) Move image/catalogue paths to config.ini?
image_paths = {
    560: "images/560mhz_1000h.fits",
    1400: "images/1400mhz_1000h.fits",
    9200: "images/9200mhz_1000h.fits",
}

pb_paths = {
    560: "images/560mhz_pb.fits",
    1400: "images/1400mhz_pb.fits",
    9200: "images/9200mhz_pb.fits",
}

# Create in-memory representation of image metadata:
image2d_list = []
for freq, path in image_paths.items():
    image2d_list.append(Image2d(freq, path, pb_paths[freq], prep=False))

# Prep images: correct PB(?), cut out training area, drop unneeded dimensions
# optionally chop into sub-images
for image2d in image2d_list:
    image2d.preprocess(segments=3, overwrite=True)

# In the directory with the images, now have e.g. 560mhz_1000h_prep_1.fits
# These are referenced in the Image2D.segments property

# Source finding (training):
sources_training = {}
for image2d in image2d_list:
    source_finder = SourceFinder(image2d.train)
    sources_training[image2d.freq] = source_finder.run()

# Feature engineering

# Train classifiers for each frequency's source dataframe:
classifiers = {}
for freq, source_train_df in sources_training.items():
    # train_classifier will be a method which uses the passed frequency and the
    # truth catalogue to train an ML model which can predict each source class
    classifiers[freq] = train_classifier(freq, source_train_df, 'class')

# Source finding (full):
source_df_dict = {}
for image2d in image2d_list:
    source_df_list = []
    for image2d_segment in image2d.segments:
        source_finder = SourceFinder(image2d_segment)
        source_finder.run()
        source_df_list.append(source_finder.get_sources())
    source_df_dict[image2d.freq] = dedupe_sources(source_df_list)

# Source classification (full):
for freq, source_df in source_df_dict.items():
    source_df["class"] = classifiers[freq].predict(source_df)


# Now improve classifications in common sky region by calculating spectral index
image3d = Image3d(image2d_list)
image3d.preprocess()
source_finder_3d = SourceFinder(image3d.train_cube)
sources_training_3d = source_finder.run()

# Train better classifier to be used if spectral index is available
