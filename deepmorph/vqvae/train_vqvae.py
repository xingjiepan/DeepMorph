import os
import json

import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms

import deepmorph.distributed
import deepmorph.data.dataset
import deepmorph.vqvae.vqvae


def transform_img_stack(img_stack):
    
    # Transform the images
    policy = torchvision.transforms.Compose([
        torchvision.transforms.RandomApply(
            transforms=[torchvision.transforms.GaussianBlur(5, sigma=(0.1, 2))], p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomAffine(degrees=(-180, 180), translate=(0.2, 0.2))
    ])
    for i in range(len(img_stack)):
        img_stack[i] = policy(img_stack[i])
        

def calc_losses(loader, model, device, n_categories=1, normalization_factor=1):
    '''Calculate the losses on a dataset.'''
    model.eval()
    
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
        
       
    model.train()
    return (n_samples, total_loss_sum, recon_loss_sum, latent_loss_sum,
           t_classify_loss_sum, b_classify_loss_sum)
    
    

def train(epoch, loader, model, optimizer, scheduler, device,
          n_categories=1, normalization_factor=1):
    '''Train one epoch.'''
    
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
        transform_img_stack(img)

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
    
    # Create the datasets
    train_dataset = deepmorph.data.dataset.DiskDataset(args['train_data_path'])
    train_sampler = deepmorph.distributed.data_sampler(train_dataset, shuffle=True, distributed=args['distributed'])
    train_loader = torch.utils.data.DataLoader(train_dataset, args['batch_size'], sampler=train_sampler, num_workers=0)
    
    val_dataset = deepmorph.data.dataset.DiskDataset(args['validation_data_path'])
    val_sampler = deepmorph.distributed.data_sampler(val_dataset, shuffle=True, distributed=args['distributed'])
    val_loader = torch.utils.data.DataLoader(val_dataset, args['batch_size'], sampler=val_sampler, num_workers=0)
    
    # Build the model
    model = deepmorph.vqvae.vqvae.VQVAE(in_channel=train_dataset[0][0].shape[0],
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
    
    loss_history = {'train_total_loss': [], 'train_recon_loss' : [], 'train_latent_loss' : [],
                    'train_t_classify_loss': [], 'train_b_classify_loss': [],
                    'val_total_loss': [], 'val_recon_loss' : [], 'val_latent_loss' : [],
                    'val_t_classify_loss': [], 'val_b_classify_loss': [],
                   }
                    
    best_loss = np.float('inf')
    for i in range(args['epoch']):
        
        # Train one epoch
        (train_n_samples, train_total_loss_sum, train_recon_loss_sum, train_latent_loss_sum, 
         train_t_classify_loss_sum, train_b_classify_loss_sum) = train(
                            i, train_loader, model, optimizer, scheduler, device,
                            n_categories=args['n_categories'], normalization_factor=args['normalization_factor'])
        
        # Calculate losses on the validation dataset
        (val_n_samples, val_total_loss_sum, val_recon_loss_sum, val_latent_loss_sum, 
         val_t_classify_loss_sum, val_b_classify_loss_sum) = calc_losses(val_loader, model, device, 
                    n_categories=args['n_categories'], normalization_factor=args['normalization_factor'])
        
        # Record the train history
        if train_n_samples > 0:
            loss_history['train_total_loss'].append(train_total_loss_sum / train_n_samples)
            loss_history['train_recon_loss'].append(train_recon_loss_sum / train_n_samples)
            loss_history['train_latent_loss'].append(train_latent_loss_sum / train_n_samples)
            loss_history['train_t_classify_loss'].append(train_t_classify_loss_sum / train_n_samples)
            loss_history['train_b_classify_loss'].append(train_b_classify_loss_sum / train_n_samples)
            
            loss_history['val_total_loss'].append(val_total_loss_sum / val_n_samples)
            loss_history['val_recon_loss'].append(val_recon_loss_sum / val_n_samples)
            loss_history['val_latent_loss'].append(val_latent_loss_sum / val_n_samples)
            loss_history['val_t_classify_loss'].append(val_t_classify_loss_sum / val_n_samples)
            loss_history['val_b_classify_loss'].append(val_b_classify_loss_sum / val_n_samples)
        
        
        if deepmorph.distributed.is_primary():
            # Save the model
            torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                       'vqvae_latest.pt'))
            
            if best_loss > loss_history['val_total_loss'][-1]:
                best_loss = loss_history['val_total_loss'][-1]
                torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                       'vqvae_best.pt'))
            
            # Save the entire history, which takes a lot of disk space
            if False:
                torch.save(model.state_dict(), os.path.join(args['output_path'], 'check_points',
                                                        f'vqvae_{str(i + 1).zfill(4)}.pt'))
    
            # Save the training history
            with open(os.path.join(args['output_path'], 'loss_history.json'), 'w') as f:
                json.dump(loss_history, f)

        
        # Stop the training if the validation loss do not decrease for a number of epochs
        if 'n_stop_threhold' in args:
            best_loss = np.min(loss_history['val_total_loss'])
            if best_loss < np.min(loss_history['val_total_loss'][-args['n_stop_threhold']:]):
                break

    
