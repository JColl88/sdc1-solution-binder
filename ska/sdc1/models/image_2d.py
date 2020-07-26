import os

import montage_wrapper as montage
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord, skycoord_to_pixel

from ska.sdc1.utils.image_utils import (
    crop_to_training_area,
    get_image_centre_coord,
    get_pixel_value_at_skycoord,
    save_subimage,
    update_header_from_cutout2D,
)


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
        How to perform PB correction depends on the ratio of image size to
        PB image pixel size; if image size is comparable or smaller than
        a single PB pixel, the reprojection with Montage will fail, but the
        correction can be approximated by taking the PB value at the input
        image centre.
        If the image is only a little larger than this, there can be a lack
        of overlap with the PB image when reprojecting, so increase the size
        """

        # Establish input image to PB image pixel size ratios:
        with fits.open(self.pb_path) as pb_hdu:
            pb_x_pixel_deg = pb_hdu[0].header["CDELT2"]
        with fits.open(self.path) as image_hdu:
            x_size = image_hdu[0].header["NAXIS1"]
            x_pixel_deg = image_hdu[0].header["CDELT2"]

        ratio_image_pb_pix = (x_size * x_pixel_deg) / pb_x_pixel_deg

        if ratio_image_pb_pix < 2.0:
            # Image not large enough to regrid (< 2 pixels in PB image);
            # apply simple correction
            coord_image_centre = get_image_centre_coord(self.path)
            pb_value = get_pixel_value_at_skycoord(self.pb_path, coord_image_centre)
            self._write_pb_corr(pb_value)
            return

        if ratio_image_pb_pix < 4.0:
            # Montage complains of a lack of overlap if image is small
            pad_cutout = 5.0
        else:
            pad_cutout = 1.0

        with fits.open(self.pb_path) as pb_hdu:
            # RA and DEC of beam PB pointing
            pb_pos = SkyCoord(
                pb_hdu[0].header["CRVAL1"] * u.degree,
                pb_hdu[0].header["CRVAL2"] * u.degree,
            )
            wcs = WCS(pb_hdu[0].header)
            pb_cutout_path = self.pb_path[:-5] + "_cutout.fits"
            pb_cutout_regrid_path = self.pb_path[:-5] + "_cutout_regrid.fits"

            size = (
                x_size * x_pixel_deg * u.degree * pad_cutout,
                x_size * x_pixel_deg * u.degree * pad_cutout,
            )

            cutout = Cutout2D(
                pb_hdu[0].data[0, 0, :, :],
                position=pb_pos,
                size=size,
                mode="trim",
                wcs=wcs.celestial,
                copy=True,
            )

            pb_hdu[0] = update_header_from_cutout2D(pb_hdu[0], cutout)
            # write updated fits file to disk
            pb_hdu[0].writeto(pb_cutout_path, overwrite=True)

        # TODO: Regrid PB image cutout to match pixel scale of the image FOV
        print(" Regridding image...")
        # get header of image to match PB to
        montage.mGetHdr(self.path, "hdu_tmp.hdr")
        # regrid pb image (270 pixels) to size of ref image (32k pixels)
        montage.reproject(
            in_images=pb_cutout_path,
            out_images=pb_cutout_regrid_path,
            header="hdu_tmp.hdr",
            exact_size=True,
        )
        os.remove("hdu_tmp.hdr")  # get rid of header text file saved to disk

        # do pb correction
        with fits.open(pb_cutout_regrid_path, mode="update") as pb_hdu:
            newdata = np.zeros(
                (1, 1, pb_hdu[0].data.shape[0], pb_hdu[0].data.shape[1]),
                dtype=np.float32,
            )
            newdata[0, 0, :, :] = pb_hdu[0].data
            pb_hdu[
                0
            ].data = newdata  # naxis will automatically update to 4 in the header

            # fix nans introduced in primary beam by montage at edges
            mask = np.isnan(pb_hdu[0].data)
            pb_hdu[0].data[mask] = np.interp(
                np.flatnonzero(mask), np.flatnonzero(~mask), pb_hdu[0].data[~mask]
            )
            pb_array = pb_hdu[0].data
            pb_hdu.flush()

        self._write_pb_corr(pb_array)

    def _write_pb_corr(self, pb_data):
        """
        Apply the PB correction and write to disk.
        
        pb_data can either be an array of the same dimensions as the image at
        self.path, or a scalar.
        """
        pb_corr_path = self.path[:-5] + "_pb_corr.fits"
        with fits.open(self.path) as image_hdu:
            image_hdu[0].data = image_hdu[0].data / pb_data
            image_hdu[0].writeto(pb_corr_path, overwrite=True)

    def _delete_train(self):
        if self.train is None:
            return
        os.remove(self._train)
