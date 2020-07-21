class Image3d:
    def __init__(self, image_2d_list):
        self.image_2d_list = image_2d_list

        self._full_cube = None
        self._train_cube = None

    @property
    def full_cube(self):
        return self._full_cube

    @property
    def train_cube(self):
        return self._train_cube

    def preprocess(self):
        pass

    def calc_spectral_index(self):
        pass
