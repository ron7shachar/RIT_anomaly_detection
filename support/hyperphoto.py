import numpy as np
import torch
from matplotlib import pyplot as plt


class HyperPhoto():
    def __init__(self, img_path = None, shape= None,  dtype= None, data = None , target = None):
        if data is None:
            self.shape = shape
            self.dtype = dtype
            self.target = target
            self._load_img(img_path)
        else:
            self.shape = data.shape
            self.dtype = data.dtype
            self.target = target
            self.data = data



    # def __init__(self,data:torch.tensor):
    #     self.shape = data.shape
    #     self.dtype = data.dtype
    #     self.data = data
    def show(self, bands):
        data = self.data
        norm_data = ((data - data.min()) / (data.max() - data.min()))
        rgb = norm_data[:, :, bands]
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.title(f"2D Composite from Bands {bands}")
        plt.axis('off')
        plt.show()

    def initial_statistics(self):
        print('init error')
        print(self.compute_error8().shape)
        print('init cov')
        print(self.compute_cov().shape)


    def compute_error8(self):
        self.error8 = self.data - self._neighbor8_mean(self.data)
        return self.error8

    def compute_cov(self):
        self.cov = self._compute_cov(self.error8)
        return self.cov

    def compute_rx(self):
        self.rx = self._compute_rx(self.cov)
        return self.rx

    """ 
    
    sub functions
    
    """

    def compute_sub_cov(self, coordinates):
        return self._compute_cov(self._get_by_coords(self.error8, coordinates))

    def compute_sub_rx(self, coordinates):
        cov = self.compute_sub_cov(coordinates)
        return self._compute_rx(cov)
    def compute_sub_p(self, coordinates):
        cov = self.compute_sub_cov(coordinates)
        return self._compute_p(cov)
    """
    
    privet functions
    
    """

    def _load_img(self, img_path):

        H, W, B = self.shape  # (height, width, bands)

        # BIL shape = (H, B, W)
        photo = np.fromfile(img_path, dtype=self.dtype).reshape(H, B, W)


        # we want final tensor = (H, W, B)
        self.data = torch.from_numpy(photo).double().permute(0, 2, 1)

    def _get_by_coords(self, x, coords):
        coords = coords.long()
        xs = coords[:, 0]
        ys = coords[:, 1]
        return x[xs, ys]

    def _neighbor8_mean(self,data):
        # x shape: (W, H, C)
        H, W, C = tuple(self.shape)

        # reshape to (1, C, H, W)
        x_ = self.data.permute(2, 0, 1).unsqueeze(0)

        # kernel של שכנים בלבד (לא כולל מרכז)
        kernel = torch.tensor([
            [1., 1., 1.],
            [1., 0., 1.],
            [1., 1., 1.]
        ], dtype=self.data.dtype)

        # לכל ערוץ בנפרד
        kernel = kernel.expand(C, 1, 3, 3)

        # סכום שכנים (יש פחות בקצוות!)
        neigh_sum = torch.nn.functional.conv2d(x_, kernel, padding=1, groups=C)

        # כמה שכנים קיימים בפועל בקצוות
        ones = torch.ones((1, C, H, W), dtype=self.data.dtype)
        neigh_count = torch.nn.functional.conv2d(ones, kernel, padding=1, groups=C)

        # ממוצע לפי מספר השכנים בפועל
        mean = neigh_sum / neigh_count

        return mean.squeeze(0).permute(1, 2, 0)

    def _compute_cov(self, error):
        errorFlat = error.reshape(-1, error.size(-1))
        return errorFlat.T @ errorFlat / (errorFlat.size(0) - 1)

    def _compute_rx(self, cov):
        bands = cov.size(1)
        error_flat = self.error8.reshape(-1, bands)
        inv_cov = torch.inverse(cov)
        tmp = error_flat @ inv_cov
        r_flat = (tmp * error_flat).sum(dim=1)
        return r_flat.view(self.error8.shape[:2])

    def _compute_p(self, cov):
        bands = cov.size(1)
        error_flat = self.error8.reshape(-1, bands)
        inv_cov = torch.inverse(cov)
        tmp = self.target @ inv_cov
        r_flat = (tmp * error_flat).sum(dim=1)
        return r_flat.view(self.error8.shape[:2])