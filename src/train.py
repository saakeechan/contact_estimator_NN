import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

from torch.utils.tensorboard import SummaryWriter

def compute_accuracy(dataloader, model):

    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(2)  # 2 legs for biped
    for sample in tqdm(dataloader):
        input_data = sample['data']
        gt_label = sample['label']

        output = model(input_data)
        _, prediction = torch.max(output,1)


        bin_pred = decimal2binary(prediction)
        bin_gt = decimal2binary(gt_label)

        correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
        num_data += input_data.size(0)
        num_correct += (prediction==gt_label).sum().item()

    return num_correct/num_data, correct_per_leg/num_data

def compute_accuracy_and_loss(dataloader, model, criterion):

    num_correct = 0
    num_data = 0
    loss_sum = 0
    correct_per_leg = np.zeros(2)  # 2 legs for biped
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']

            output = model(input_data)
            _, prediction = torch.max(output,1)

            loss = criterion(output, gt_label)

            bin_pred = decimal2binary(prediction)
            bin_gt = decimal2binary(gt_label)

            correct_per_leg += (bin_pred==bin_gt).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            num_correct += (prediction==gt_label).sum().item()

            loss_sum += loss.item()

    return num_correct/num_data, correct_per_leg/num_data, loss_sum/len(dataloader)

def decimal2binary(x):
    mask = 2**torch.arange(2-1,-1,-1).to(x.device, x.dtype)  # 2 legs for biped

    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def save_onnx_model(model, checkpoint_path, window_size):
    """
    Save ONNX version of the model for C++ deployment.
    This is called automatically after saving checkpoints.
    ONNX provides 1.5-3x faster inference than TorchScript.
    """
    try:
        # Create ONNX path (replace .pt with .onnx)
        onnx_path = checkpoint_path.replace('.pt', '.onnx')
        
        # Create example input for export
        device = next(model.parameters()).device
        example_input = torch.randn(1, window_size, 54).to(device)
        
        # Model must be in eval mode
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"  ✓ ONNX model saved: {onnx_path}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Failed to save ONNX model: {e}")


def train(model, train_dataloader, val_dataloader, config):

    writer = SummaryWriter(config['log_writer_path'],comment=config['model_description'])
    writer.add_text("data_folder: ",config['data_folder'])
    writer.add_text("model_save_path: ",config['model_save_path'])
    writer.add_text("log_writer_path: ",config['log_writer_path'])
    writer.add_text("window_size: ",str(config['window_size']))
    writer.add_text("shuffle: ",str(config['shuffle']))
    writer.add_text("batch_size: ",str(config['batch_size']))
    writer.add_text("init_lr: ",str(config['init_lr']))
    writer.add_text("num_epoch: ",str(config['num_epoch']))


    # create criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['init_lr'])

    best_acc = 0
    best_leg_acc = 0
    best_loss = 1000000000
    for epoch in range(config['num_epoch']):
        running_loss = 0.0
        loss_sum = 0.0
        model.train()
        for i, samples in tqdm(enumerate(train_dataloader, start=0)):
            input_data = samples['data'] 
            label = samples['label'] 

            optimizer.zero_grad()
            output = model(input_data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # cur_training_loss_history.append(loss.item())
            running_loss += loss.item()
            loss_sum += loss.item()

            if i % config['print_every'] == 0:
                print("epoch %d / %d, iteration %d / %d, loss: %.8f" %\
                    (epoch, config['num_epoch'], i, len(train_dataloader), running_loss/config['print_every']))
                running_loss = 0.0

        # calculate training and validation accuracy
        model.eval()
        train_acc, train_acc_per_leg = compute_accuracy(train_dataloader, model)
        train_loss_avg = loss_sum/len(train_dataloader)

        val_acc, val_acc_per_leg, val_loss_avg = compute_accuracy_and_loss(val_dataloader, model, criterion)

        train_acc_per_leg_avg = (np.sum(train_acc_per_leg)/2.0)  # 2 legs for biped
        val_acc_per_leg_avg = (np.sum(val_acc_per_leg)/2.0)  # 2 legs for biped

        # log down info in tensorboard
        writer.add_scalar('training loss', train_loss_avg, epoch)
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('training acc leg0', train_acc_per_leg[0], epoch)
        writer.add_scalar('training acc leg1', train_acc_per_leg[1], epoch)
        writer.add_scalar('training acc leg avg', train_acc_per_leg_avg, epoch)
        writer.add_scalar('validation loss', val_loss_avg, epoch)
        writer.add_scalar('validation accuracy', val_acc, epoch)
        writer.add_scalar('validation acc leg0', val_acc_per_leg[0], epoch)
        writer.add_scalar('validation acc leg1', val_acc_per_leg[1], epoch)
        writer.add_scalar('validation acc leg avg', val_acc_per_leg_avg, epoch)

        # if we achieve best val acc, save the model.
        if val_acc > best_acc:
            best_acc = val_acc
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_acc.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
        
        # if we achieve best val leg acc, save the model.
        if val_acc_per_leg_avg > best_leg_acc:
            best_leg_acc = val_acc_per_leg_avg
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_leg_acc.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])

        # if we achieve best val loss, save the model
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            
            state = {'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss_avg,
                    'acc': train_acc,
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc}

            checkpoint_path = config['model_save_path']+'_best_val_loss.pt'
            torch.save(state, checkpoint_path)
            # Also save ONNX version for faster C++ deployment
            save_onnx_model(model, checkpoint_path, config['window_size'])
            

        print("Finished epoch %d / %d, training acc: %.4f, validation acc: %.4f" %\
            (epoch, config['num_epoch'], train_acc, val_acc)) 
        print("train leg0 acc: %.4f, train leg1 acc: %.4f, train leg acc avg: %.4f" %\
            (train_acc_per_leg[0],train_acc_per_leg[1],train_acc_per_leg_avg))    
        print("val leg0 acc: %.4f, val leg1 acc: %.4f, val leg acc avg: %.4f" %\
            (val_acc_per_leg[0],val_acc_per_leg[1],val_acc_per_leg_avg))
    
    # save model     
    state = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss_avg,
            'acc': train_acc,
            'val_loss': val_loss_avg,
            'val_acc': val_acc}

    checkpoint_path = config['model_save_path']+'_final_epo.pt'
    torch.save(state, checkpoint_path)
    # Also save ONNX version for faster C++ deployment
    save_onnx_model(model, checkpoint_path, config['window_size'])

    writer.close()

def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)


    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/network_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    print("Using the following params: ")
    print("-------------path-------------")
    print("data_folder: ",config['data_folder'])
    print("model_save_path: ",config['model_save_path'])
    print("log_writer_path: ",config['log_writer_path'])
    print("--------network params--------")
    print("window_size: ",config['window_size'])
    print("shuffle: ",config['shuffle'])
    print("batch_size: ",config['batch_size'])
    print("init_lr: ",config['init_lr'])
    print("num_epoch: ",config['num_epoch'])

    
    # Load ALL data (not pre-split) - windowing happens first, then splitting
    all_dataset = contact_dataset(data_path=config['data_folder']+"all_data.npy",\
                                  label_path=config['data_folder']+"all_labels.npy",\
                                  window_size=config['window_size'],device=device)
    
    # Split the windows into train/val/test
    dataset_size = len(all_dataset)
    indices = list(range(dataset_size))
    
    train_ratio = config.get('train_ratio', 0.7)
    val_ratio = config.get('val_ratio', 0.15)
    
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    # Shuffle indices if specified
    if config['shuffle']:
        np.random.seed(config.get('random_seed', 42))
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Total windows: {dataset_size}")
    print(f"  Train windows: {len(train_indices)}")
    print(f"  Val windows: {len(val_indices)}")
    print(f"  Test windows: {len(test_indices)}")
    
    # Create Subset datasets for deterministic sampling
    from torch.utils.data import Subset
    train_dataset = Subset(all_dataset, train_indices)
    val_dataset = Subset(all_dataset, val_indices)
    
    # Create dataloaders
    # Train: shuffle each epoch for better training (but with reproducible seed from config)
    # Val: don't shuffle - we want consistent validation across epochs
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'],\
                                  shuffle=config['shuffle'])
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'],\
                                shuffle=False)

    # init network
    model = contact_cnn(window_size=config['window_size'])
    model = model.to(device)

    train(model, train_dataloader, val_dataloader, config)

   

if __name__ == '__main__':
    main()
