import os
import json

import numpy as np
from tqdm import tqdm
import torch

import deepmorph.distributed
import deepmorph.data.dataset
import deepmorph.vqvae.vqvae


def train(epoch, loader, model, optimizer, scheduler, device,
          n_categories=1, normalization_factor=1):
    if deepmorph.distributed.is_primary():
        loader = tqdm(loader)
    
    recon_loss_fn = torch.nn.MSELoss()
    classify_loss_fn = torch.nn.CrossEntropyLoss()

    latent_loss_weight = 0.25
    classify_loss_weight = 1

    (n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum,
     t_classify_loss_sum, b_classify_loss_sum) = 0, 0, 0, 0, 0, 0
    
    for i, (img, y) in enumerate(loader):
        
        model.zero_grad()
        
        # Send the image to GPU and normalize the image
        img = img.to(device, dtype=torch.float32)
        img = img / normalization_factor
        y = y.to(device)
                
        out, latent_loss, class_t, class_b = model(img)
        
        # Calcualte the losses
        recon_loss = recon_loss_fn(out, img)
        latent_loss = latent_loss.mean()
        
        if n_categories > 1:
            t_classify_loss = classify_loss_fn(class_t, y)
            b_classify_loss = classify_loss_fn(class_b, y)
        else:
            t_classify_loss = torch.tensor(0)
            b_classify_loss = torch.tensor(0)
        
        loss = recon_loss + latent_loss_weight * latent_loss \
               + classify_loss_weight * (t_classify_loss + b_classify_loss)
        loss.backward()
        
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
                
        # Record the loss history
        comm = {
            'n_samples': img.shape[0],
            'total_loss_sum' : loss.item() * img.shape[0],
            'recon_loss_sum' : recon_loss.item() * img.shape[0],
            'latent_loss_sum' : latent_loss.item() * img.shape[0],
            't_classify_loss_sum': t_classify_loss.item() * img.shape[0],
            'b_classify_loss_sum': b_classify_loss.item() * img.shape[0],
        }
        comm = deepmorph.distributed.all_gather(comm)
                
        for part in comm:
            n_samples += part["n_samples"]
            total_loss_sum += part["total_loss_sum"]
            recon_loss_sum += part["recon_loss_sum"]
            latent_loss_sum += part["latent_loss_sum"]
            t_classify_loss_sum += part["t_classify_loss_sum"]
            b_classify_loss_sum += part["b_classify_loss_sum"]
        
        
    return (n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum,
           t_classify_loss_sum, b_classify_loss_sum)

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
    model = deepmorph.vqvae.vqvae.VQVAE(in_channel=dataset[0][0].shape[0],
                       n_categories=args['n_categories'], img_xy_shape=args['img_xy_shape']).to(device)
    
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[deepmorph.distributed.get_local_rank()],
            output_device=deepmorph.distributed.get_local_rank()
        )
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    
    # Anneal the learning rate if required
    scheduler = None
    if args.get('anneal_lr', False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                              T_max = args['epoch'], # Maximum number of iterations.
                              eta_min = 0.1 * args['lr']) # Minimum learning rate.
    
    # Train the model
    if deepmorph.distributed.is_primary():
        os.makedirs(os.path.join(args['output_path'], 'check_points'), exist_ok=True)
    
    loss_history = {'total_loss': [], 'recon_loss' : [], 'latent_loss' : [],
                    't_classify_loss': [], 'b_classify_loss': []}
                    
    best_loss = np.float('inf')
        
    for i in range(args['epoch']):
        (n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum, 
         t_classify_loss_sum, b_classify_loss_sum) = train(i, loader, model, optimizer, scheduler, device,
                            n_categories=args['n_categories'], normalization_factor=args['normalization_factor'])
        
        if n_samples > 0:
            loss_history['total_loss'].append(total_loss_sum / n_samples)
            loss_history['recon_loss'].append(recon_loss_sum / n_samples)
            loss_history['latent_loss'].append(latent_loss_sum / n_samples)
            loss_history['t_classify_loss'].append(t_classify_loss_sum / n_samples)
            loss_history['b_classify_loss'].append(b_classify_loss_sum / n_samples)
        
        if deepmorph.distributed.is_primary():
            # Save the model
            torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                       'vqvae_latest.pt'))
            
            if best_loss > loss_history['total_loss'][-1]:
                best_loss = loss_history['total_loss'][-1]
                torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                       'vqvae_best.pt'))
            
            # Save the entire history, which takes a lot of disk space
            if False:
                torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                        f'vqvae_{str(i + 1).zfill(4)}.pt'))
    
            # Save the training history
            with open(os.path.join(args['output_path'], 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f)
    


    
