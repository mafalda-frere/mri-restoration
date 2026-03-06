import torch
import torch.nn as nn
import torch.nn.functional as F


class Automap(nn.Module):
    def __init__(self, m, K, dim_bottleneck=64, conv_channels=64):
        """PyTorch implementation of AUTOMAP
        Zhu, B., Liu, J. Z., Cauley, S. F., Rosen, B. R., & Rosen, M. S. (2018). 
        Image reconstruction by domain-transform manifold learning. Nature, 555(7697), 487-492
        """
        super().__init__()

        self.K=K

        # on aurait pu init que les nnlinear et conv et les appeler dans le forward...
        self.res=nn.Sequential(
            nn.Linear(in_features=2*m, out_features=dim_bottleneck),
            nn.Tanh(),
            nn.Linear(in_features=dim_bottleneck,out_features=K**2),
            # nn.BatchNorm1d(K**2), # on enlève batchnorm car on peut pas compute sur un batch size de 1
            nn.Unflatten(dim=1, unflattened_size=(1, K, K)),    # pour avoir 4 dim (équivalent à x=x.view(-1,1,self.K,self.K))
            nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=5, padding=5//2),  # padding pour garder la taille de l'image
            nn.Tanh(),
            nn.Conv2d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=5, padding=5//2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=conv_channels, out_channels=1, kernel_size=7, padding=3) # obligés de rajouter du padding pour matcher 320*320 en sortie
        )

    def forward(self, kspace, mask):
        # ici kspace vaut 1*320*320 soit batch*m*m
        # on applique le masque sur les colonnes
        #print("kspace shape:",kspace.shape)
        #print("mask shape:",mask.shape)
        obs_kspace = kspace[:, :, mask]
        #print("obs_kspace shape:",obs_kspace.shape)
        obs_kspace = torch.view_as_real(obs_kspace)
        #print("obs_kspace post view as real shape:",obs_kspace.shape)
        obs_kspace = torch.flatten(obs_kspace,start_dim=1)
        #print("obs_kspace post flatten shape:",obs_kspace.shape)
        y=self.res(obs_kspace)
        #print("y post NN:",y.shape)
        return y.squeeze(1)