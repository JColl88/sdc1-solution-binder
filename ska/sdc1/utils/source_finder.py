import glob
import os
import shutil

from astropy.io import fits

import bdsf
from ska.sdc1.models.exceptions import CatalogueException
from ska.sdc1.utils.bdsf_utils import gaul_as_df, srl_as_df


class SourceFinder:
    """
    Find sources using PyBDSF.


    Args:
        image_path (:obj:`str`): Path to the image to search for sources in
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self._run_complete = False

    @property
    def image_dirname(self):
        """
        Path of directory containing image
        """
        return os.path.dirname(self.image_path)

    @property
    def image_name(self):
        """
        Image file name
        """
        return os.path.basename(self.image_path)

    def get_srl_path(self):
        """
        Get BDSF source list path
        """
        return self.get_output_cat("srl")

    def get_gaul_path(self):
        """
        Get BDSF Gaussian list path
        """
        return self.get_output_cat("gaul")

    def get_bdsf_log_path(self):
        """
        Get BDSF log file path
        """
        return "{}.pybdsf.log".format(self.image_path)

    def get_bdsf_out_path(self):
        """
        Get BDSF output directory
        """
        return "{}_pybdsm".format(self.image_path[:-5])

    def get_output_cat(self, extn):
        srl_glob = glob.glob(
            "{}_pybdsm/*/catalogues/*.{}".format(self.image_path[:-5], extn)
        )
        if len(srl_glob) == 1:
            return srl_glob[0]
        elif len(srl_glob) < 1:
            raise Exception("No output catalogue of type {} found".format(extn))
        else:
            raise Exception("More than 1 catalogue of type {} found".format(extn))

    def run(self, beam=()):
        self._run_complete = False

        # Must switch the executor's working directory to the image directory to
        # run PyBDSF, and switch back after the run is complete.
        cwd = os.getcwd()
        os.chdir(self.image_basepath)

        # Get beam info automatically if not provided
        if not beam:
            beam = self.get_beam_from_hdu()

        # Run PyBDSF
        _img = bdsf.process_image(
            self.image_name,
            adaptive_rms_box=True,
            advanced_opts=True,
            atrous_do=False,
            psf_vary_do=True,
            psf_snrcut=5.0,
            psf_snrcutstack=10.0,
            output_opts=True,
            output_all=True,
            opdir_overwrite="overwrite",
            beam=beam,
            blank_limit=None,
            thresh="hard",
            thresh_isl=5.0,
            thresh_pix=7.0,
            psf_snrtop=0.30,
        )

        # Revert current working directory
        os.chdir(cwd)
        self._run_complete = True

        return self.get_source_df()

    def get_beam_from_hdu(self):
        try:
            with fits.open(self.image_name) as hdu:
                beam_maj = hdu[0].header["BMAJ"]
                beam_min = hdu[0].header["BMIN"]
                beam_pa = 0
                return (beam_maj, beam_min, beam_pa)
        except IndexError:
            raise Exception("Unable to automatically determine beam info")

    def get_source_df(self):
        """
        Given a gaussian list (.gaul) and source list (.srl) from PyBDSF, return a
        catalogue of sources, including the number of components.
        """
        gaul_df = gaul_as_df(self.get_gaul_path())
        srl_df = srl_as_df(self.get_srl_path())

        srl_df["n_gaussians"] = gaul_df["Source_id"].value_counts()

        return srl_df

    def reset(self):
        """
        Clean up previous BDSF run output
        """
        self._run_complete = False
        if os.path.isfile(self.get_bdsf_log_path()):
            os.remove(self.get_bdsf_log_path())
        if os.path.isdir(self.get_bdsf_out_path()):
            shutil.rmtree(self.get_bdsf_out_path())
