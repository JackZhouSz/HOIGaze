from utils import adt_dataset, seed_torch
from model import gaze_estimation
from utils.opt import options
from utils import log
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
import torch.optim as optim
import torch.nn.functional as F
import os
os.nice(5)
import math


def main(opt):
    # set the random seed to ensure reproducibility
    seed_torch.seed_torch(seed=0)
    torch.set_num_threads(1)

    data_dir = opt.data_dir
    seq_len = opt.seq_len
    opt.joint_number = opt.body_joint_number + opt.hand_joint_number*2
    learning_rate = opt.learning_rate
    print('>>> create model')
    net = gaze_estimation.gaze_estimation(opt=opt).to(opt.cuda_idx)    
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    print('>>> loading datasets')
    
    train_data_path = os.path.join(opt.ckpt, "attended_hand_recognition_train.npy")
    valid_data_path = os.path.join(opt.ckpt, "attended_hand_recognition_test.npy")
    
    train_dataset = np.load(train_data_path)    
    train_data_size = train_dataset.shape
    print("Training data size: {}".format(train_data_size))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)    
    valid_dataset = np.load(valid_data_path)    
    valid_data_size = valid_dataset.shape
    print("Validation data size: {}".format(valid_data_size))                
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # training
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTraining starts at ' + local_time)
    start_time = datetime.datetime.now()
    start_epoch = 1

    err_best = 1000
    best_epoch = 0
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma, last_epoch=-1)
    for epo in range(start_epoch, opt.epoch + 1):
        is_best = False            
        learning_rate = exp_lr.optimizer.param_groups[0]["lr"]
            
        train_start_time = datetime.datetime.now()
        result_train = run_model(net, optimizer, is_train=1, data_loader=train_loader, opt=opt)        
        train_end_time = datetime.datetime.now()
        train_time = (train_end_time - train_start_time).seconds*1000
        train_batch_num = math.ceil(train_data_size[0]/opt.batch_size)
        train_time_per_batch = math.ceil(train_time/train_batch_num)
        #print('\nTraining time per batch: {} ms'.format(train_time_per_batch))
        
        exp_lr.step()
        rng_state = torch.get_rng_state()
        if epo % opt.validation_epoch == 0:                        
            print('>>> training epoch: {:d}, lr: {:.12f}'.format(epo, learning_rate))
            print('Training data size: {}'.format(train_data_size))          
            print('Average baseline error: {:.2f} degree'.format(result_train['baseline_error_average']))
            print('Average training error: {:.2f} degree'.format(result_train['prediction_error_average']))
            
            test_start_time = datetime.datetime.now()
            result_valid = run_model(net, is_train=0, data_loader=valid_loader, opt=opt)
            test_end_time = datetime.datetime.now()
            test_time = (test_end_time - test_start_time).seconds*1000
            test_batch_num = math.ceil(valid_data_size[0]/opt.test_batch_size)
            test_time_per_batch = math.ceil(test_time/test_batch_num)
            #print('\nTest time per batch: {} ms'.format(test_time_per_batch))
            print('Validation data size: {}'.format(valid_data_size))
            
            print('Average baseline error: {:.2f} degree'.format(result_valid['baseline_error_average']))
            print('Average validation error: {:.2f} degree'.format(result_valid['prediction_error_average']))
            
            if result_valid['prediction_error_average'] < err_best:
                err_best = result_valid['prediction_error_average']
                is_best = True
                best_epoch = epo
                
            print('Best validation error: {:.2f} degree, best epoch: {}'.format(err_best, best_epoch))
            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds/60
            print('\nTotal training time: {:.2f} min'.format(total_training_time))
            local_time = time.asctime(time.localtime(time.time()))
            print('\nTraining ends at ' + local_time)
            
            result_log = np.array([epo, learning_rate])
            head = np.array(['epoch', 'lr'])
            for k in result_train.keys():
                result_log = np.append(result_log, [result_train[k]])
                head = np.append(head, [k])
            for k in result_valid.keys():
                result_log = np.append(result_log, [result_valid[k]])
                head = np.append(head, ['valid_' + k])

            csv_name = 'gaze_estimation_results'            
            log.save_csv_log(opt, head, result_log, is_create=(epo == 1), file_name=csv_name)
            last_model_name = 'gaze_estimation_model_last.pt'
            log.save_ckpt({'epoch': epo,
                           'lr': learning_rate,
                           'err': result_valid['prediction_error_average'],
                           'state_dict': net.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            opt=opt,
                            file_name = last_model_name)
            if epo == best_epoch:
                best_model_name = 'gaze_estimation_model_best.pt'
                log.save_ckpt({'epoch': epo,
                               'lr': learning_rate,
                               'err': result_valid['prediction_error_average'],
                               'state_dict': net.state_dict(),
                               'optimizer': optimizer.state_dict()},
                                opt=opt,
                                file_name = best_model_name)
                                
        torch.set_rng_state(rng_state)

        
def eval(opt):
    data_dir = opt.data_dir
    seq_len = opt.seq_len
    opt.joint_number = opt.body_joint_number + opt.hand_joint_number*2
    
    print('>>> create model')
    net = gaze_estimation.gaze_estimation(opt=opt).to(opt.cuda_idx)    
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    #load model    
    model_name = 'gaze_estimation_model_best.pt'
    model_path = os.path.join(opt.ckpt, model_name)    
    print(">>> loading ckpt from '{}'".format(model_path))
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))
    
    print('>>> loading datasets')    
    test_data_path = os.path.join(opt.ckpt, "attended_hand_recognition_test.npy")
    test_dataset = np.load(test_data_path)
    test_data_size = test_dataset.shape
    print("Test data size: {}".format(test_data_size))
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # test
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest starts at ' + local_time)
    start_time = datetime.datetime.now()    
    if opt.save_predictions:
        result_test, predictions = run_model(net, is_train=0, data_loader=test_loader, opt=opt)
    else:
        result_test = run_model(net, is_train=0, data_loader=test_loader, opt=opt)
    
    print('Average baseline error: {:.2f} degree'.format(result_test['baseline_error_average']))
    print('Average prediction error: {:.2f} degree'.format(result_test['prediction_error_average']))
    
    end_time = datetime.datetime.now()
    total_test_time = (end_time - start_time).seconds/60
    print('\nTotal test time: {:.2f} min'.format(total_test_time))
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest ends at ' + local_time)
    
    if opt.save_predictions:    
        # ground_truth + joints + head_directions + object_positions + attended_hand_prd + attended_hand_gt + predictions
        batch_size, seq_n, dim = predictions.shape
        predictions = predictions.reshape(-1, dim)
        ground_truth = predictions[:, :3]
        head_directions = predictions[:, 3+opt.joint_number*3:6+opt.joint_number*3]        
        head_cos = np.sum(head_directions*ground_truth, 1)
        head_cos = np.clip(head_cos, -1, 1)        
        head_errors = np.arccos(head_cos)/np.pi * 180.0
        print('Average baseline error: {:.2f} degree'.format(np.mean(head_errors)))
        
        prediction = predictions[:, -3:]
        prd_cos = np.sum(prediction*ground_truth, 1)
        prd_cos = np.clip(prd_cos, -1, 1)        
        prediction_errors = np.arccos(prd_cos)/np.pi * 180.0
        print('Average prediction error: {:.2f} degree'.format(np.mean(prediction_errors)))
        
        attended_hand_gt = predictions[:, -4]        
        attended_hand_prd_left = predictions[:, -6]
        attended_hand_prd_right = predictions[:, -5]
        attended_hand_correct = attended_hand_prd_left
        for i in range(attended_hand_correct.shape[0]):
            if attended_hand_gt[i] == 0 and attended_hand_prd_left[i] > attended_hand_prd_right[i]:
                attended_hand_correct[i] = 1
            elif attended_hand_gt[i] == 1 and attended_hand_prd_left[i] < attended_hand_prd_right[i]:
                attended_hand_correct[i] = 1
            else:
                attended_hand_correct[i] = 0

        correct_ratio = np.sum(attended_hand_correct)/attended_hand_correct.shape[0]
        print("hand recognition acc: {:.2f}%".format(correct_ratio*100))
        attended_hand_wrong = 1 - attended_hand_correct
        wrong_ratio = np.sum(attended_hand_wrong)/attended_hand_wrong.shape[0]
        
        head_errors_correct = np.sum(head_errors*attended_hand_correct)/np.sum(attended_hand_correct)
        print("hand recognition correct size: {}".format(np.sum(attended_hand_correct)))
        print("hand recognition correct, average baseline error: {:.2f} degree".format(head_errors_correct))
        head_errors_wrong = np.sum(head_errors*attended_hand_wrong)/np.sum(attended_hand_wrong)
        print("hand recognition wrong size: {}".format(np.sum(attended_hand_wrong)))
        print("hand recognition wrong, average baseline error: {:.2f} degree".format(head_errors_wrong))
        head_errors_avg = head_errors_correct*correct_ratio + head_errors_wrong*wrong_ratio
        print('Average baseline error: {:.2f} degree'.format(head_errors_avg))
        
        prediction_errors_correct = np.sum(prediction_errors*attended_hand_correct)/np.sum(attended_hand_correct)
        print("hand recognition correct, average prediction error: {:.2f} degree".format(prediction_errors_correct))
        prediction_errors_wrong = np.sum(prediction_errors*attended_hand_wrong)/np.sum(attended_hand_wrong)
        print("hand recognition wrong, average prediction error: {:.2f} degree".format(prediction_errors_wrong))
        prediction_errors_avg = prediction_errors_correct*correct_ratio + prediction_errors_wrong*wrong_ratio
        print('Average prediction error: {:.2f} degree'.format(prediction_errors_avg))
        
        predictions_path = os.path.join(opt.ckpt, "gaze_predictions.npy")
        np.save(predictions_path, predictions)        
        prediction_errors_path = os.path.join(opt.ckpt, "prediction_errors.npy")
        np.save(prediction_errors_path, prediction_errors)        
        attended_hand_correct_path = os.path.join(opt.ckpt, "attended_hand_correct.npy")
        np.save(attended_hand_correct_path, attended_hand_correct)
        
        
def acos_safe(x, eps=1e-6):
    slope = np.arccos(1-eps) / eps
    buf = torch.empty_like(x)
    good = abs(x) <= 1-eps
    bad = ~good
    sign = torch.sign(x[bad])
    buf[good] = torch.acos(x[good])
    buf[bad] = torch.acos(sign * (1 - eps)) - slope*sign*(abs(x[bad]) - 1 + eps)
    return buf

    
def run_model(net, optimizer=None, is_train=1, data_loader=None, opt=None):
    if is_train == 1:
        net.train()
    else:
        net.eval()
            
    if opt.is_eval and opt.save_predictions:
        predictions = []
                        
    prediction_error_average = 0
    baseline_error_average = 0
    criterion = torch.nn.MSELoss(reduction='none')
    
    n = 0
    input_n = opt.seq_len
    
    for i, (data) in enumerate(data_loader):
        batch_size, seq_n, dim = data.shape
        joint_number = opt.joint_number
        object_num = opt.object_num
        # when only one sample in this batch
        if batch_size == 1 and is_train == 1:
            continue        
        n += batch_size
        data = data.float().to(opt.cuda_idx)
        
        ground_truth = data.clone()[:, :, :3]
        joints = data.clone()[:, :, 3:(joint_number+1)*3]
        head_directions = data.clone()[:, :, (joint_number+1)*3:(joint_number+2)*3]        
        attended_hand_prd = data.clone()[:, :, (joint_number+2+8*object_num*2)*3:(joint_number+2+8*object_num*2)*3+2]
        attended_hand_gt = data.clone()[:, :, (joint_number+2+8*object_num*2)*3+2:(joint_number+2+8*object_num*2)*3+3]
        
        input = torch.cat((joints, head_directions), dim=2)
        if object_num > 0:
            object_positions = data.clone()[:, :, (joint_number+2)*3:(joint_number+2+8*object_num*2)*3]
            input = torch.cat((input, object_positions), dim=2)            
        input = torch.cat((input, attended_hand_prd), dim=2)
        input = torch.cat((input, attended_hand_gt), dim=2)        
        prediction = net(input, input_n=input_n)
        
        if opt.is_eval and opt.save_predictions:
            # ground_truth + joints + head_directions + object_positions + attended_hand_prd + attended_hand_gt + predictions
            prediction_cpu = torch.cat((ground_truth, input), dim=2)            
            prediction_cpu = torch.cat((prediction_cpu, prediction), dim=2)
            prediction_cpu = prediction_cpu.cpu().data.numpy()
            if len(predictions) == 0:
                predictions = prediction_cpu                
            else:
                predictions = np.concatenate((predictions, prediction_cpu), axis=0)           
                
        gaze_head_cos = torch.sum(ground_truth*head_directions, dim=2, keepdim=True)        
        gaze_weight = torch.where(gaze_head_cos>opt.gaze_head_cos_threshold, opt.gaze_head_loss_factor, 1.0)
        
        loss = criterion(ground_truth, prediction)           
        loss = torch.mean(loss*gaze_weight)
        
        if is_train == 1:            
            optimizer.zero_grad()
            loss.backward()                        
            optimizer.step()
            
        # Calculate prediction errors
        error = torch.mean(acos_safe(torch.sum(ground_truth*prediction, 2)))/torch.tensor(math.pi) * 180.0
        prediction_error_average += error.cpu().data.numpy() * batch_size
        
        # Use head directions as the baseline
        baseline_error = torch.mean(acos_safe(torch.sum(ground_truth*head_directions, 2)))/torch.tensor(math.pi) * 180.0
        baseline_error_average += baseline_error.cpu().data.numpy() * batch_size
            
    result = {}
    result["prediction_error_average"] = prediction_error_average / n
    result["baseline_error_average"] = baseline_error_average / n
    
    if opt.is_eval and opt.save_predictions:        
        return result, predictions
    else:
        return result
        
if __name__ == '__main__':    
    option = options().parse()
    if option.is_eval == False:
        main(option)
    else:
        eval(option)