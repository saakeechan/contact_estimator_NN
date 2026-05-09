import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import scipy.io as sio

import lcm
from lcm_types.python import contact_t, leg_control_data_lcmt, microstrain_lcmt
import time

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *

def inference(dataloader, model, device):
    """
    Run inference and return contact and velocity predictions.
    Returns:
        contact_results: (N, 2) contact predictions [left, right]
        velocity_results: (N, 2) velocity predictions [left, right]
    """
    contact_results = torch.empty(0, 2, dtype=torch.uint8).to(device)  # 2 legs for biped [left, right]
    velocity_results = torch.empty(0, 2, dtype=torch.float32).to(device)  # 2 legs for biped [left, right]
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            contact_output, velocity_output = model(input_data)  # Two outputs: (batch, 2) each
            contact_prediction = (torch.sigmoid(contact_output) > 0.5).byte()  # Binary predictions
            contact_results = torch.cat((contact_results, contact_prediction), 0)
            velocity_results = torch.cat((velocity_results, velocity_output), 0)

    return contact_results, velocity_results


def inference_and_compute_acc(dataloader, model, device):
    """
    Run inference and compute accuracy metrics for contact and velocity.
    Returns:
        contact_results: (N, 2) contact predictions [left, right]
        velocity_results: (N, 2) velocity predictions [left, right]
        accuracy: overall contact accuracy
        per_leg_accuracy: (2,) per-leg contact accuracy [left, right]
        velocity_mse: overall velocity MSE
    """
    num_correct = 0
    num_data = 0
    correct_per_leg = np.zeros(2)  # 2 legs for biped [left, right]
    contact_results = torch.empty(0, 2, dtype=torch.uint8).to(device)  # 2 legs for biped
    velocity_results = torch.empty(0, 2, dtype=torch.float32).to(device)  # 2 legs for biped
    velocity_mse_sum = 0.0
    
    with torch.no_grad():
        for sample in tqdm(dataloader):
            input_data = sample['data']
            gt_label = sample['label']  # Shape: (batch, 2) - binary labels [left, right]
            gt_velocity = sample['velocity']  # Shape: (batch, 2) - velocity [left, right]

            contact_output, velocity_output = model(input_data)  # Two outputs: (batch, 2) each
            contact_prediction = (torch.sigmoid(contact_output) > 0.5).float()  # Binary predictions
            contact_results = torch.cat((contact_results, contact_prediction.byte()), 0)
            velocity_results = torch.cat((velocity_results, velocity_output), 0)

            # Per-leg contact accuracy
            correct_per_leg += (contact_prediction == gt_label).sum(axis=0).cpu().numpy()
            num_data += input_data.size(0)
            # Overall contact accuracy (averaged across both legs)
            num_correct += (contact_prediction == gt_label).sum().item()
            
            # Velocity MSE
            velocity_mse_sum += ((velocity_output - gt_velocity) ** 2).sum().item()

    # Total accuracy considers all predictions (both legs)
    total_predictions = num_data * 2  # 2 legs per sample
    velocity_mse = velocity_mse_sum / total_predictions
    
    return contact_results, velocity_results, num_correct/total_predictions, correct_per_leg/num_data, velocity_mse

def decimal2binary(x):
    mask = 2**torch.arange(2-1,-1,-1).to(x.device, x.dtype)  # 2 legs for biped

    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def save2mat(pred, config):
    mat_raw_data = sio.loadmat(config['mat_data_path'])
    data = np.load(config['data_path'])
    label = np.load(config['label_path'])  # Now loads binary labels directly from data_handler

    # Convert to proper shape if needed
    if label.ndim == 1:
        # Old decimal format - convert to binary
        label_binary = np.zeros((len(label), 2), dtype=np.float32)
        label_binary[:, 0] = (label & 2) >> 1  # Left foot
        label_binary[:, 1] = label & 1          # Right foot
        label = label_binary

    out = {}
    out['contacts_est'] = pred.cpu().numpy()
    out['contacts_gt'] = label[config['window_size']-1:,:]
    out['q'] = data[config['window_size']-1:,:12]
    out['qd'] = data[config['window_size']-1:,12:24]
    out['imu_acc'] = data[config['window_size']-1:,24:27]
    out['imu_omega'] = data[config['window_size']-1:,27:30]
    out['p'] = data[config['window_size']-1:,30:36]
    out['v'] = data[config['window_size']-1:,36:42]

    # data not used in the network but needed for visualization.
    out['control_time'] = mat_raw_data['control_time'].flatten().tolist()[config['window_size']-1:]
    out['imu_time'] = mat_raw_data['imu_time'].flatten().tolist()[config['window_size']-1:]
    out['tau_est'] = mat_raw_data['tau_est'][config['window_size']-1:]
    out['F'] = mat_raw_data['F'][config['window_size']-1:]

    sio.savemat(config['mat_save_path'],out)

    print("Saved data to mat!")

def save2lcm(pred, config):
    mat_data = sio.loadmat(config['mat_data_path'])
    log = lcm.EventLog(config['lcm_save_path'], mode='w', overwrite=True)
    
    utime = int(time.time() * 10**6)

    imu_time = mat_data['imu_time'].flatten().tolist()


    
    for idx,_ in enumerate(imu_time[config['window_size']-1:]):

        data_idx = idx + config['window_size']-1
        
        leg_control_data_msg = leg_control_data_lcmt()
        leg_control_data_msg.q = mat_data['q'][data_idx]
        leg_control_data_msg.p = mat_data['p'][data_idx]
        leg_control_data_msg.qd = mat_data['qd'][data_idx]
        leg_control_data_msg.v = mat_data['v'][data_idx]
        leg_control_data_msg.tau_est = mat_data['tau_est'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                    'leg_control_data', leg_control_data_msg.encode())
        
        contact_msg = contact_t()
        contact_msg.num_legs = 2  # 2 legs for biped
        contact_msg.timestamp = imu_time[data_idx]
        contact_msg.contact = pred[idx]

        # if we want to use GT contact for verification
        # contact_msg.contact = mat_data['contacts'][data_idx]
        
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'contact', contact_msg.encode())
        
        imu_msg = microstrain_lcmt()
        imu_msg.acc = mat_data['imu_acc'][data_idx]
        imu_msg.omega = mat_data['imu_omega'][data_idx]
        imu_msg.rpy = mat_data['imu_rpy'][data_idx]
        imu_msg.quat = mat_data['imu_quat'][data_idx]
        log.write_event(utime + int(10**6 * imu_time[data_idx]),\
                        'microstrain', imu_msg.encode())
        
    print("Saved data to lcm!")



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/inference_one_seq_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name), Loader=yaml.FullLoader)
    
    dataset = contact_dataset(data_path=config['data_path'],\
                                label_path=config['label_path'],\
                                window_size=config['window_size'],device=device)
    dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])

    model = contact_cnn(window_size=config['window_size'])
    from contact_cnn import ContactCNNWithNormalization
    model = ContactCNNWithNormalization(model)
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    pred_contact = []
    pred_velocity = []
    if(config['calculate_accuracy']):
        pred_contact, pred_velocity, acc, acc_per_leg, velocity_mse = inference_and_compute_acc(dataloader, model, device)
        print("Contact Accuracy (both legs): %.4f" % acc)
        print("Accuracy of leg 0 (left): %.4f" % acc_per_leg[0])
        print("Accuracy of leg 1 (right): %.4f" % acc_per_leg[1])
        print("Average leg accuracy: %.4f" % (np.sum(acc_per_leg)/2.0))  # 2 legs for biped
        print("Velocity MSE (both legs): %.6f" % velocity_mse)
    else:
        pred_contact, pred_velocity = inference(dataloader, model, device)

    

    if(config['save_mat']):
        save2mat(pred_contact, config)

    if(config['save_lcm']):
        save2lcm(pred_contact, config)

if __name__ == '__main__':
    main()