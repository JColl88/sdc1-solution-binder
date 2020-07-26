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
        Currently fails at update_header_from_cutout2D step due to input image format
        """
        with fits.open(self.pb_path) as pb_hdu:
            pb_x_pixel_deg = pb_hdu[0].header["CDELT2"]
        with fits.open(self.path) as image_hdu:
            # cutout pb field of view to match image field of view
            x_size = image_hdu[0].header["NAXIS1"]
            y_size = image_hdu[0].header["NAXIS2"]
            x_pixel_deg = image_hdu[0].header["CDELT2"]
        size = (
            x_size * x_pixel_deg * u.degree,
            x_size * x_pixel_deg * u.degree,
        )
        print("PB pixel is {} deg".format(pb_x_pixel_deg))
        print("Img total size is {} deg".format(x_size * x_pixel_deg))
        if ((x_size * x_pixel_deg) / 2.0) < pb_x_pixel_deg:
            print("Not large enough to regrid, (2 PB pixels)")
            pb_hdu = fits.open(self.pb_path)
            image_hdu = fits.open(self.path)
            # Take average pixel number:
            print(x_size / 2, y_size / 2)
            coord_image_centre = pixel_to_skycoord(
                x_size / 2, y_size / 2, WCS(image_hdu[0].header)
            )
            print(coord_image_centre)
            pb_x_pixel, pb_y_pixel = skycoord_to_pixel(
                coord_image_centre, WCS(pb_hdu[0].header)
            )
            pb_x_pixel = round(float(pb_x_pixel))
            pb_y_pixel = round(float(pb_y_pixel))
            pb_value = pb_hdu[0].data[0, 0, round(pb_x_pixel), round(pb_y_pixel)]

            image_hdu[0].data = image_hdu[0].data / pb_value
            image_hdu[0].writeto(self.path[:-5] + "_pb_corr.fits", overwrite=True)

            return

        if ((x_size * x_pixel_deg) / 4.0) < pb_x_pixel_deg:
            size = (
                x_size * x_pixel_deg * u.degree * 5.0,
                x_size * x_pixel_deg * u.degree * 5.0,
            )

        with fits.open(self.pb_path) as pb_hdu:
            # RA and DEC of beam PB pointing
            pb_pos = SkyCoord(
                pb_hdu[0].header["CRVAL1"] * u.degree,
                pb_hdu[0].header["CRVAL2"] * u.degree,
            )
            wcs = WCS(pb_hdu[0].header)
            pb_cutout_path = self.pb_path[:-5] + "_cutout.fits"
            pb_cutout_regrid_path = self.pb_path[:-5] + "_cutout_regrid.fits"
            # save_subimage(self.pb_path, pb_cor_path, pb_pos, size)

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
            pb_hdu[0].writeto(
                pb_cutout_path, overwrite=True
            )  # Write the cutout to a new FITS file

        # TODO: Regrid PB image cutout to match pixel scale of the image FOV
        print(" Regridding image...")
        print(self.path)
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
            # print(pb_hdu[0].data)
            mask = np.isnan(pb_hdu[0].data)
            pb_hdu[0].data[mask] = np.interp(
                np.flatnonzero(mask), np.flatnonzero(~mask), pb_hdu[0].data[~mask]
            )
            pb_data = pb_hdu[0].data
            pb_hdu.flush()
        with fits.open(self.path) as hdu:
            hdu[0].data = hdu[0].data / pb_data
            hdu[0].writeto(self.path[:-5] + "_pb_corr.fits", overwrite=True)

    def _delete_train(self):
        if self.train is None:
            return
        os.remove(self._train)
