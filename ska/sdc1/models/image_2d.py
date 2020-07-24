import os

import montage_wrapper as montage
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS

from ska.sdc1.utils.image_utils import (
    crop_to_training_area,
    save_subimage,
    update_header_from_cutout2D,
)

# from astropy.nddata.utils import Cutout2D
# from astropy.wcs import WCS


class Image2d:
    def __init__(self, freq, path, pb_path, prep=False):
        self._freq = freq
        self._path = path
        self._pb_path = pb_path
        self._prep = prep

        self._train = None

    @property
    def freq(self):
        return self._freq

    @property
    def path(self):
        return self._path

    @property
    def pb_path(self):
        return self._pb_path

    @property
    def train(self):
        return self._train

    def preprocess(self, overwrite=True):
        """
        Perform preprocessing steps:
        1) Apply PB correction
        2) Output separate training image
        3) Only output necessary data dimensions
        """
        self._apply_pb_corr()
        self._create_train(overwrite)

    def _create_train(self, pad_factor=1.0):
        self._train = None
        train_path = self.path[:-5] + "_train.fits"
        crop_to_training_area(self.path, train_path, self.freq, pad_factor)
        self._train = train_path

    def _apply_pb_corr(self):
        """
        Currently fails at update_header_from_cutout2D step due to input image format
        """
        with fits.open(self.path) as image_hdu:
            # cutout pb field of view to match image field of view
            x_size = image_hdu[0].header["NAXIS1"]
            x_pixel_deg = image_hdu[0].header[
                "CDELT2"
            ]  # CDELT1 is negative, so take positive one
        size = (
            x_size * x_pixel_deg * u.degree,
            x_size * x_pixel_deg * u.degree,
        )
        with fits.open(self.pb_path) as pb_hdu:
            # RA and DEC of beam PB pointing
            pb_hdu_0 = pb_hdu[0]
            pb_pos = SkyCoord(
                pb_hdu_0.header["CRVAL1"] * u.degree,
                pb_hdu_0.header["CRVAL2"] * u.degree,
            )
            wcs = WCS(pb_hdu_0.header)
            pb_cutout_path = self.pb_path[:-5] + "_cutout.fits"
            pb_cor_path = self.pb_path[:-5] + "_pb_corr.fits"
            pb_cor_rg_path = self.pb_path[:-5] + "_pb_corr_regrid.fits"
            # save_subimage(self.pb_path, pb_cor_path, pb_pos, size)

            cutout = Cutout2D(
                pb_hdu_0.data[0, 0, :, :],
                position=pb_pos,
                size=size,
                mode="trim",
                wcs=wcs.celestial,
                copy=True,
            )

            pb_hdu_0 = update_header_from_cutout2D(pb_hdu_0, cutout)
            # write updated fits file to disk
            pb_hdu_0.writeto(
                pb_cutout_path, overwrite=True
            )  # Write the cutout to a new FITS file

        # TODO: Regrid PB image cutout to match pixel scale of the image FOV
        print(" Regridding image...")
        # get header of image to match PB to
        montage.mGetHdr(self.path, "hdu_tmp.hdr")
        # regrid pb image (270 pixels) to size of ref image (32k pixels)
        montage.reproject(
            in_images=pb_cor_path,
            out_images=pb_cor_rg_path,
            header="hdu_tmp.hdr",
            exact_size=True,
        )
        os.remove("hdu_tmp.hdr")  # get rid of header text file saved to disk

        # do pb correction
        with fits.open(pb_cor_rg_path) as pb_hdu:
            # fix nans introduced in primary beam by montage at edges
            print(pb_hdu[0].data)
            mask = np.isnan(pb_hdu[0].data)
            pb_hdu[0].data[mask] = np.interp(
                np.flatnonzero(mask), np.flatnonzero(~mask), pb_hdu[0].data[~mask]
            )
            pb_data = pb_hdu[0].data
        with fits.open(self.path) as hdu:
            hdu[0].data = hdu[0].data / pb_data
            hdu[0].writeto(pb_cor_path, overwrite=True)

    def do_primarybeam_correction(self, pbname, imagename):

        # regrid PB image cutout to match pixel scale of the image FOV
        print(" Regridding image...")
        # get header of image to match PB to
        montage.mGetHdr(imagename, "hdu_tmp.hdr")
        # regrid pb image (270 pixels) to size of ref image (32k pixels)
        montage.reproject(
            in_images=pbname[:-5] + "_cutout.fits",
            out_images=pbname[:-5] + "_cutout_regrid.fits",
            header="hdu_tmp.hdr",
            exact_size=True,
        )
        os.remove("hdu_tmp.hdr")  # get rid of header text file saved to disk

        # update montage output to float32
        pb = fits.open(pbname[:-5] + "_cutout_regrid.fits", mode="update")
        newdata = np.zeros(
            (1, 1, pb[0].data.shape[0], pb[0].data.shape[1]), dtype=np.float32
        )
        newdata[0, 0, :, :] = pb[0].data
        pb[0].data = newdata  # naxis will automatically update to 4 in the header

        # fix nans introduced in primary beam by montage at edges and write to new file
        print(
            " A small buffer of NaNs is introduced around the image by Montage when regridding to match the size, \n these have been set to the value of their nearest neighbours to maintain the same image dimensions"
        )
        mask = np.isnan(pb[0].data)
        pb[0].data[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), pb[0].data[~mask]
        )
        pb.flush()
        pb.close()

        # apply primary beam correction
        pb = fits.open(pbname[:-5] + "_cutout_regrid.fits")[0]
        hdu.data = hdu.data / pb.data
        hdu.writeto(imagename[:-5] + "_PBCOR.fits", overwrite=True)
        print(
            " Primary beam correction applied to {0}".format(
                imagename[:-5] + "_PBCOR.fits"
            )
        )

    def _delete_train(self):
        if self.train is None:
            return
        os.remove(self._train)
