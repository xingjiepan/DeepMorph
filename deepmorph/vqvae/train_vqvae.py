import os
import json

import numpy as np
from tqdm import tqdm
import torch

import deepmorph.distributed
import deepmorph.data.dataset
import deepmorph.vqvae.vqvae


def train(epoch, loader, model, optimizer, scheduler, device, normalization_factor=255):
    if deepmorph.distributed.is_primary():
        loader = tqdm(loader)
    
    criterion = torch.nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum = 0, 0, 0, 0
    
    for i, (img, y) in enumerate(loader):
        
        model.zero_grad()
        
        # Send the image to GPU and normalize the image
        img = img.to(device, dtype=torch.float32)
        img = img / normalization_factor
                
        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()
                        
        optimizer.step()
                
        # Record the loss history
        comm = {
            'n_samples': img.shape[0],
            'total_loss_sum' : loss.item() * img.shape[0],
            'recon_loss_sum' : recon_loss.item() * img.shape[0],
            'latent_loss_sum' : latent_loss.item() * img.shape[0],
        }
        comm = deepmorph.distributed.all_gather(comm)
                
        for part in comm:
            n_samples += part["n_samples"]
            total_loss_sum += part["total_loss_sum"]
            recon_loss_sum += part["recon_loss_sum"]
            latent_loss_sum += part["latent_loss_sum"]
        
        
    return n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum

def build_model_and_train(args):
    '''Build and train a VQVAE model.
    Args:
        args: A dictionary of arguments.
    '''
    device = args.get('device', 'cuda')
    args['distributed'] = deepmorph.distributed.get_world_size() > 1
    
    # Create the dataset
    dataset = deepmorph.data.dataset.DiskDataset(args['data_path'])
    sampler = deepmorph.distributed.data_sampler(dataset, shuffle=True, distributed=args['distributed'])
    loader = torch.utils.data.DataLoader(dataset, args['batch_size'], sampler=sampler, num_workers=0)
    
    # Build the model
    model = deepmorph.vqvae.vqvae.VQVAE(in_channel=dataset[0][0].shape[0]).to(device)
    
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[deepmorph.distributed.get_local_rank()],
            output_device=deepmorph.distributed.get_local_rank(),
        )
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    scheduler = None
    
    # Train the model
    if deepmorph.distributed.is_primary():
        os.makedirs(os.path.join(args['output_path'], 'check_points'), exist_ok=True)
    
    loss_history = {'total_loss': [], 'recon_loss' : [], 'latent_loss' : []}
                     
    for i in range(args['epoch']):
        n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum = train(i, loader, model, optimizer, scheduler, device)
        
        if n_samples > 0:
            loss_history['total_loss'].append(total_loss_sum / n_samples)
            loss_history['recon_loss'].append(recon_loss_sum / n_samples)
            loss_history['latent_loss'].append(latent_loss_sum / n_samples)
        
        if deepmorph.distributed.is_primary():
            torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                        f'vqvae_{str(i + 1).zfill(4)}.pt'))
    
    # Save the training history
    if deepmorph.distributed.is_primary():
        with open(os.path.join(args['output_path'], 'loss_history.json'), 'w') as f:
            json.dump(loss_history, f)
    


    
