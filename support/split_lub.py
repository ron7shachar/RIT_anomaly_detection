import numpy as np
import torch
import matplotlib.pyplot as plt
from support.hyperphoto import HyperPhoto


class SplitBub:
    def __init__(self, source_path, shape, target_path, p):
        target = torch.load(target_path)

        self.hyperPhoto = HyperPhoto(source_path, shape, np.int16, target=target)
        self.hyperPhoto_t = HyperPhoto(data=(self.hyperPhoto.data + p * target), target=target)

        self.hyperPhoto.initial_statistics()
        self.hyperPhoto_t.error8 = self.hyperPhoto.data - self.hyperPhoto_t._neighbor8_mean(self.hyperPhoto_t.data)

    def compute_p(self, subPix):
        cov = self.hyperPhoto.compute_sub_cov(subPix)
        self.hyperPhoto.p = self.hyperPhoto._compute_p(cov)
        self.hyperPhoto_t.p = self.hyperPhoto_t._compute_p(cov)

    def show_analize(self, bins_=100, range_=(-200, 300), show=True):
        # First histogram calculations
        wtd, bins1 = np.histogram(
            self.hyperPhoto_t.p.view(-1).numpy(),
            bins=bins_,
            range=range_,
            density=True)
        ntd, bins2= np.histogram(
            self.hyperPhoto.p.view(-1).numpy(),
            bins=bins_,
            range=range_,
            density=True)

        # Compute CDFs
        pd = -np.cumsum(ntd * np.diff(bins2)) + 1
        pfa = -np.cumsum(wtd * np.diff(bins1)) + 1

        # Find first index
        n_ = next((i for i in range(len(pfa)) if pfa[i] < 1), 0)

        # ========================
        #   DRAW ONLY IF show=True
        # ========================
        if show:
            # Histogram plot
            plt.figure(figsize=(10, 6))
            plt.hist(self.hyperPhoto_t.p.view(-1).numpy(), bins=bins_, range=range_,
                     density=True, histtype='step', label='NT', color='blue')
            plt.hist(self.hyperPhoto.p.view(-1).numpy(), bins=bins_, range=range_,
                     density=True, histtype='step', label='WT', color='green')
            plt.legend()
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.title("Distribution Histogram (NT vs WT)")
            plt.show()

            # CDF plot
            plt.figure(figsize=(10, 6))
            plt.plot(bins2[:-1], pd, label="pd", linewidth=2)
            plt.plot(bins1[:-1], pfa, label="pfa", linewidth=2)
            plt.xlabel("Value")
            plt.ylabel("Density / CDF")
            plt.title("error")
            plt.legend()
            plt.show()

            # ROC
            plt.figure(figsize=(10, 6))
            plt.plot(pfa[n_:], pd[n_:], label="R", color="blue")
            plt.xlabel("pfa")
            plt.ylabel("pd")
            plt.title("ROC")
            plt.legend()
            plt.show()

        return pfa[n_:], pd[n_:]

    # def show_analize(self,bins_ = 100,range_ = (-200,300),show = True):
    #     wtd, bins1, _ = plt.hist(
    #         self.hyperPhoto_t.p.view(-1).numpy(),  # שימוש בנתוני NT
    #         bins=bins_,
    #         range=range_,
    #         density=True,
    #         histtype='step',
    #         label='NT',  # הוספת התווית NT למקרא
    #         color='blue'  # צבע כחול
    #     )
    #
    #     # היסטוגרמה 2 (WT - ירוק): נצבע ונשים תווית WT שנייה במקרא
    #     # **שימו לב**: הסרנו את density=True
    #     ntd, bins2, _ = plt.hist(
    #         self.hyperPhoto.p.view(-1).numpy(),  # שימוש בנתוני WT
    #         bins=bins_,
    #         range=range_,
    #         density=True,
    #         histtype='step',
    #         label='WT',  # הוספת התווית WT למקרא
    #         color='green'  # צבע ירוק
    #     )
    #
    #     # הגדרות הגרף
    #     plt.legend()
    #     plt.xlabel("Value")
    #     plt.ylabel("Count")  # שינינו ל-Count כי הסרנו density=True
    #     plt.title("Distribution Histogram (NT vs WT)")
    #     if show:plt.show()
    #     plt.figure(figsize=(10, 6))
    #
    #     # Histogram 1
    #
    #     pd = -np.cumsum(ntd * np.diff(bins1)) + 1
    #     plt.plot(bins1[:-1], pd, linewidth=2, label='pd')
    #
    #     # Histogram 2
    #     pfa = -np.cumsum(wtd * np.diff(bins2)) + 1
    #     plt.plot(bins2[:-1], pfa, linewidth=2, label='pfa')
    #
    #     plt.xlabel("Value")
    #     plt.ylabel("Density / CDF")
    #     plt.title("error")
    #     plt.legend()
    #     if show:plt.show()
    #
    #
    #     n_ = 0
    #     for n in range(len(pfa)):
    #         if pfa[n] < 1:
    #             n_ = n
    #             break
    #
    #     plt.plot(pfa[n_:], pd[n_:], label="R", color='blue')
    #
    #
    #     plt.xlabel("pfa")
    #     plt.ylabel("pd")
    #     plt.title("ROC ")
    #     plt.legend()
    #     if show:plt.show()
    #
    #     return pfa[n_:] , pd[n_:]
