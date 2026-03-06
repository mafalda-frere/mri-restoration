import os
import yaml
import argparse
import pprint

import torch
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from dataset import fastMriData
from models.automap import Automap
from utils import get_params

def main(cfg):
    os.makedirs(os.path.dirname(cfg['res_dir']), exist_ok=True)

    with open(os.path.join(cfg['res_dir'], 'train_config.yaml'), 'w') as file:
        yaml.dump(cfg, file)

    mask_path = os.path.join(cfg['res_dir'], 'kspace_mask.pt')
    if os.path.exists(mask_path):
        mask = torch.load(mask_path)
    else:
        np.random.seed(cfg['seed'])
        mask = fastMriData.kspace_mask(cfg['kspace_shape'], acceleration_factor=cfg['acceleration_factor'])
        torch.save(mask, os.path.join(cfg['res_dir'], 'kspace_mask.pt'))

    train_dataset = fastMriData(cfg['data_folder'], mask=mask, mean=cfg['data_mean'], std=cfg['data_std'])

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True
    )

    model_cfg = cfg['model']
    K = cfg['kspace_shape'][0]
    m = cfg['kspace_shape'][0] * (cfg['kspace_shape'][1] // cfg['acceleration_factor'] - 1)
    model = Automap(m, K, model_cfg['dim_bottleneck'], model_cfg['conv_channels'])

    params = get_params(model)
    pprint.pprint("Total number of parameters: {:.2f}M".format(params / 10**6))

    epochs = cfg['epochs']
    lr = float(cfg['lr'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    if cfg['device'] == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = model.to(device)
    mask = mask.to(device)

    metrics = {
        'train': {
            'loss': [],
        }
    }

    model.train()
    best_loss = np.inf
    print("Start training...")
    for epoch in range(epochs):
        train_loss = 0
        
        for image, kspace, target_image, _ in train_data_loader:
            image = image.to(device)
            kspace = kspace.to(device)
            target_image = target_image.to(device)

            pred = model(kspace,mask)
            loss = criterion(pred,target_image)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_data_loader)

        if (epoch + 1) % cfg['verbose'] == 0:
            print(" === Epoch {} - L1 loss: {:.4f} ===".format(epoch+1, train_loss))

    
        metrics['train']['loss'].append(train_loss)

        if train_loss <= best_loss:
            best_loss = train_loss
            torch.save({'epoch': epoch, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                        os.path.join(cfg['res_dir'], 'best_model.pth.tar'))
            
        if train_loss < cfg['min_loss']:
            print('Final loss: ', train_loss)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(pred[0].detach().cpu(), cmap='gray')
            ax[1].imshow(target_image[0].cpu(), cmap='gray')
            plt.show()
            break
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
                        help='Path to the training config.')
   
    parser = parser.parse_args()

    with open(parser.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)

    pprint.pprint(cfg)
    main(cfg)
