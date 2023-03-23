import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 불필요한 success 메시지 끄기

import neptune.new as neptune
run = neptune.init(
    project="youngandbin/n2d",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMGFlNjA5MC1lN2Y3LTQ2ZmItOWM0Yi1jYWZlOWIwOTk1ZjAifQ==",
)  

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import datetime
import warnings; warnings.filterwarnings("always"); warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
import umap.umap_ as umap
from sklearn import mixture

from model import creditDataloader, Autoencoder

parser = argparse.ArgumentParser(description='N2D Training')

# 실험 세팅
parser.add_argument('-v', '--ver', default=1, type=int,
                    help='version of preprocessed data {1, 2}')
parser.add_argument('-k', '--k-list', nargs='+', type=int,
                    help='list of numbers of clusters')

# 모델 학습
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='number of batch size for AutoEncoder')
parser.add_argument('-o', '--optimizer', default='adam', type=str,
                    help='adam or adamw')
parser.add_argument('-l', '--loss-function', default='mse', type=str,
                    help='mse or smoothl1')

# 기타
parser.add_argument('-u', '--umap', default=3, type=int,
                    help='dimension reduction by umap')
parser.add_argument('-g', '--gpu-id', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# neptune
params = {"epoch": 1}
run["parameters"] = params

# random seed
SEED = 2021 
np.random.seed(SEED)
torch.manual_seed(SEED)

# data
data_dir = './data/'
df = pd.read_feather(data_dir + 'clustering_data_1901_2004_ver{0}.ftr'.format(args.ver))
info_features = df.loc[:, :'지역'].columns.tolist()
features = df.loc[:,'가정용품':'주거'].columns.tolist()
CSVDATA = df.copy()

# AutoEncoder
def train_AE(model, optimizer, loss_fn, train_loader, n_epochs, device):

    for epoch in tqdm(range(n_epochs)):
        loss_train = 0.0
        for data in train_loader:
            data = data.to(device=device).view(data.shape[0], -1)
            data = torch.tensor(data, dtype=torch.float32)
            outputs = model(data) ################################################################################왜 forward 안해도되는거??
            loss = loss_fn(outputs, data)
            loss_train += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 1 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch + 1, loss_train / len(train_loader)))

        # neptune
        run["train/loss"].log(loss_train / len(train_loader))

    return model

# UMAP, GMM
def UMAP_GMM(latent_1, cluster):

    latent_1 = latent_1.cpu().data.numpy()
    latent_2 = umap.UMAP(random_state=2021, n_components = args.umap).fit_transform(latent_1)
    
    # clustering on new manifold of autoencoded embedding
    gmm = mixture.GaussianMixture(covariance_type= "full", n_components=cluster, random_state=2021).fit(latent_2)
    pred_prob = gmm.predict_proba(latent_2)
    pred = pred_prob.argmax(1)

    return pred, latent_2



if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)

    for k in tqdm(args.k_list):

        print('\n----- LOOP: k={} -----\n'.format(k))

        dataset = creditDataloader(CSVDATA).data
        device = (torch.device('cuda') if torch.cuda.is_available()
                  else torch.device('cpu'))

        # model
        model = Autoencoder(numLayers=[18, 16, 16, k]).to(device)

        # optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)  
        # loss
        if args.loss_function == 'mse':
            loss_fn = nn.MSELoss()
        elif args.loss_function == 'smoothl1':
            loss_fn = nn.SmoothL1Loss() 

        # train
        print('\n----- LOOP: train AE -----\n')
        trainLoader_AE = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=16) # num_workers=16
        trained_model = train_AE(model=model, optimizer=optimizer, loss_fn=loss_fn, train_loader=trainLoader_AE, n_epochs=args.epochs, device=device)
        encoder = nn.Sequential(*[model.layers[i] for i in range(5)]).to(device)

        print('\n----- AE, UMAP, GMM -----n')
        dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, num_workers=16)
        for data in dataloader:
            data = data.to(device).view(data.shape[0], -1)
            data = torch.tensor(data, dtype=torch.float32)
            latent_1 = encoder(data)

            pd.DataFrame(latent_1.cpu().data.numpy()).to_csv(data_dir + 'clustering_results/ver{0}/latent_1_cluster{1}_umap{2}_epochs{3}_batchsize{4}.csv'.format(
                args.ver, k, args.umap, args.epochs, args.batch_size))

            pred, latent_2 = UMAP_GMM(latent_1, k)

        pd.DataFrame(pred).to_csv(data_dir + 'clustering_results/ver{0}/pred_cluster{1}_umap{2}_epochs{3}_batchsize{4}.csv'.format(
            args.ver, k, args.umap, args.epochs, args.batch_size))
        pd.DataFrame(latent_2).to_csv(data_dir + 'clustering_results/ver{0}/latent_2_cluster{1}_umap{2}_epochs{3}_batchsize{4}.csv'.format(
            args.ver, k, args.umap, args.epochs, args.batch_size))