from utils import hot3d_aria_dataset, seed_torch
from model import attended_hand_recognition
from utils.opt import options
from utils import log
from torch.utils.data import DataLoader
import torch
import numpy as np
import time
import datetime
import torch.optim as optim
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
    net = attended_hand_recognition.attended_hand_recognition(opt=opt).to(opt.cuda_idx)
    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=learning_rate, weight_decay=opt.weight_decay)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))
    print('>>> loading datasets')
    
    actions = opt.actions
    test_user_id = opt.test_user_id
    if actions == 'all':            
        if test_user_id == 1:
            train_actions = 'all'
            test_actions = 'all'
            train_subjects = ['P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
            test_subjects = ['P0001', 'P0002', 'P0003']
            opt.ckpt = opt.ckpt + '/user1/'
        if test_user_id == 2:    
            train_actions = 'all'
            test_actions = 'all'        
            train_subjects = ['P0001', 'P0002', 'P0003', 'P0012', 'P0014', 'P0015']
            test_subjects = ['P0009', 'P0010', 'P0011']
            opt.ckpt = opt.ckpt + '/user2/'
        if test_user_id == 3:
            train_actions = 'all'
            test_actions = 'all'        
            train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011']
            test_subjects = ['P0012', 'P0014', 'P0015']
            opt.ckpt = opt.ckpt + '/user3/'
    elif actions == 'room':
        train_actions = ['kitchen', 'office']
        test_actions = ['room']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene1/'
    elif actions == 'kitchen':
        train_actions = ['room', 'office']
        test_actions = ['kitchen']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene2/'
    elif actions == 'office':
        train_actions = ['room', 'kitchen']
        test_actions = ['office']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene3/'
    else:
        raise( ValueError, "Unrecognised actions: %d" % actions)
        
    if not os.path.isdir(opt.ckpt):
        os.makedirs(opt.ckpt)
        
    train_dataset = hot3d_aria_dataset.hot3d_aria_dataset(data_dir, train_subjects, seq_len, train_actions, opt.object_num, opt.sample_rate)
    train_data_size = train_dataset.dataset.shape
    print("Training data size: {}".format(train_data_size))
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dataset = hot3d_aria_dataset.hot3d_aria_dataset(data_dir, test_subjects, seq_len, test_actions, opt.object_num, opt.sample_rate)
    valid_data_size = valid_dataset.dataset.shape
    print("Validation data size: {}".format(valid_data_size))                
    valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # training
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTraining starts at ' + local_time)
    start_time = datetime.datetime.now()
    start_epoch = 1

    acc_best = 0
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
            if actions == 'all':
                print("\ntest user id: {}\n".format(test_user_id))
            elif actions == 'room':
                print("\ntest scene/action: room\n")
            elif actions == 'kitchen':
                print("\ntest scene/action: kitchen\n")
            elif actions == 'office':
                print("\ntest scene/action: office\n")               
            print('>>> training epoch: {:d}, lr: {:.12f}'.format(epo, learning_rate))
            print('Training data size: {}'.format(train_data_size))          
            print('Average baseline acc: {:.2f}%'.format(result_train['baseline_acc_average']*100))
            print('Average training acc: {:.2f}%'.format(result_train['prediction_acc_average']*100))
            
            test_start_time = datetime.datetime.now()
            result_valid = run_model(net, is_train=0, data_loader=valid_loader, opt=opt)                        
            test_end_time = datetime.datetime.now()
            test_time = (test_end_time - test_start_time).seconds*1000
            test_batch_num = math.ceil(valid_data_size[0]/opt.test_batch_size)
            test_time_per_batch = math.ceil(test_time/test_batch_num)
            #print('\nTest time per batch: {} ms'.format(test_time_per_batch))
            print('Validation data size: {}'.format(valid_data_size))
            
            print('Average baseline acc: {:.2f}%'.format(result_valid['baseline_acc_average']*100))
            print('Average validation acc: {:.2f}%'.format(result_valid['prediction_acc_average']*100))
            
            if result_valid['prediction_acc_average'] > acc_best:
                acc_best = result_valid['prediction_acc_average']
                is_best = True
                best_epoch = epo
                
            print('Best validation error: {:.2f}%, best epoch: {}'.format(acc_best*100, best_epoch))                                                
            end_time = datetime.datetime.now()
            total_training_time = (end_time - start_time).seconds/60
            print('\nTotal training time: {:.1f} min'.format(total_training_time))
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
            
            csv_name = 'attended_hand_recognition_results'
            model_name = 'attended_hand_recognition_model.pt'
            log.save_csv_log(opt, head, result_log, is_create=(epo == 1), file_name=csv_name)
            log.save_ckpt({'epoch': epo,
                           'lr': learning_rate,
                           'acc': result_valid['prediction_acc_average'],
                           'state_dict': net.state_dict(),
                           'optimizer': optimizer.state_dict()},
                            opt=opt,
                            file_name = model_name)
                            
        torch.set_rng_state(rng_state)

        
def eval(opt):
    data_dir = opt.data_dir
    seq_len = opt.seq_len
    opt.joint_number = opt.body_joint_number + opt.hand_joint_number*2
    
    print('>>> create model')
    net = attended_hand_recognition.attended_hand_recognition(opt=opt).to(opt.cuda_idx)    
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))    
    #load model    
    actions = opt.actions
    test_user_id = opt.test_user_id
    if actions == 'all':            
        if test_user_id == 1:
            train_actions = 'all'
            test_actions = 'all'
            train_subjects = ['P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
            test_subjects = ['P0001', 'P0002', 'P0003']
            opt.ckpt = opt.ckpt + '/user1/'
        if test_user_id == 2:    
            train_actions = 'all'
            test_actions = 'all'        
            train_subjects = ['P0001', 'P0002', 'P0003', 'P0012', 'P0014', 'P0015']
            test_subjects = ['P0009', 'P0010', 'P0011']
            opt.ckpt = opt.ckpt + '/user2/'
        if test_user_id == 3:
            train_actions = 'all'
            test_actions = 'all'        
            train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011']
            test_subjects = ['P0012', 'P0014', 'P0015']
            opt.ckpt = opt.ckpt + '/user3/'
    elif actions == 'room':
        train_actions = ['kitchen', 'office']
        test_actions = ['room']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene1/'
    elif actions == 'kitchen':
        train_actions = ['room', 'office']
        test_actions = ['kitchen']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene2/'
    elif actions == 'office':
        train_actions = ['room', 'kitchen']
        test_actions = ['office']        
        train_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        test_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
        opt.ckpt = opt.ckpt + '/scene3/'
    else:
        raise( ValueError, "Unrecognised actions: %d" % actions)
            
    model_name = 'attended_hand_recognition_model.pt'
    model_path = os.path.join(opt.ckpt, model_name)    
    print(">>> loading ckpt from '{}'".format(model_path))
    ckpt = torch.load(model_path)
    net.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt loaded (epoch: {} | acc: {})".format(ckpt['epoch'], ckpt['acc']))
    
    print('>>> loading datasets')                  
    train_dataset = hot3d_aria_dataset.hot3d_aria_dataset(data_dir, train_subjects, seq_len, train_actions, opt.object_num, opt.sample_rate)
    train_data_size = train_dataset.dataset.shape
    print("Train data size: {}".format(train_data_size))                
    train_loader = DataLoader(train_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)            
    test_dataset = hot3d_aria_dataset.hot3d_aria_dataset(data_dir, test_subjects, seq_len, test_actions, opt.object_num, opt.sample_rate)
    test_data_size = test_dataset.dataset.shape
    print("Test data size: {}".format(test_data_size))                
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # test
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest starts at ' + local_time)
    start_time = datetime.datetime.now()
    if actions == 'all':
        print("\ntest user id: {}\n".format(test_user_id))
    elif actions == 'room':
        print("\ntest scene/action: room\n")
    elif actions == 'kitchen':
        print("\ntest scene/action: kitchen\n")
    elif actions == 'office':
        print("\ntest scene/action: office\n")    
    if opt.save_predictions:
        result_train, predictions_train = run_model(net, is_train=0, data_loader=train_loader, opt=opt)
        result_test, predictions_test = run_model(net, is_train=0, data_loader=test_loader, opt=opt)
    else:
        result_train = run_model(net, is_train=0, data_loader=train_loader, opt=opt)
        result_test = run_model(net, is_train=0, data_loader=test_loader, opt=opt)

    print('Average train baseline acc: {:.2f}%'.format(result_train['baseline_acc_average']*100))
    print('Average train method acc: {:.2f}%'.format(result_train['prediction_acc_average']*100))    
    print('Average test baseline acc: {:.2f}%'.format(result_test['baseline_acc_average']*100))
    print('Average test method acc: {:.2f}%'.format(result_test['prediction_acc_average']*100))
    
    end_time = datetime.datetime.now()
    total_test_time = (end_time - start_time).seconds/60
    print('\nTotal test time: {:.1f} min'.format(total_test_time))
    local_time = time.asctime(time.localtime(time.time()))
    print('\nTest ends at ' + local_time)
    
    if opt.save_predictions:
        prediction = predictions_train[:, :, -3:-1].reshape(-1, 2)
        attended_hand_gt = predictions_train[:, :, -1:].reshape(-1)
        y_prd = np.argmax(prediction, axis=1)
        acc = np.sum(y_prd == attended_hand_gt)/prediction.shape[0]
        print('Average train acc: {:.2f}%'.format(acc*100))        
        predictions_train_path = os.path.join(opt.ckpt, "attended_hand_recognition_train.npy")
        np.save(predictions_train_path, predictions_train)        
        
        prediction = predictions_test[:, :, -3:-1].reshape(-1, 2)
        attended_hand_gt = predictions_test[:, :, -1:].reshape(-1)
        y_prd = np.argmax(prediction, axis=1)
        acc = np.sum(y_prd == attended_hand_gt)/prediction.shape[0]
        print('Average test acc: {:.2f}%'.format(acc*100))        
        predictions_test_path = os.path.join(opt.ckpt, "attended_hand_recognition_test.npy")
        np.save(predictions_test_path, predictions_test)        

        
def run_model(net, optimizer=None, is_train=1, data_loader=None, opt=None):
    if is_train == 1:
        net.train()
    else:
        net.eval()
            
    if opt.is_eval and opt.save_predictions:
        predictions = []
    
    prediction_acc_average = 0
    baseline_acc_average = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    n = 0    
    input_n = opt.seq_len
    
    for i, (data) in enumerate(data_loader):
        batch_size, seq_n, dim = data.shape
        joint_number = opt.joint_number
        object_num = opt.object_num
        # when only one sample in this batch
        if batch_size == 1 and is_train == 1:
            continue        
        n += batch_size*seq_n
        data = data.float().to(opt.cuda_idx)
                
        eye_gaze = data.clone()[:, :, :3]
        joints = data.clone()[:, :, 3:(joint_number+1)*3]
        head_directions = data.clone()[:, :, (joint_number+1)*3:(joint_number+2)*3]
        attended_hand_gt = data.clone()[:, :, (joint_number+2+8*object_num*2)*3:(joint_number+2+8*object_num*2)*3+1].type(torch.LongTensor).to(opt.cuda_idx)
        attended_hand_baseline = data.clone()[:, :, (joint_number+2+8*object_num*2)*3+1:(joint_number+2+8*object_num*2)*3+2].type(torch.LongTensor).to(opt.cuda_idx)
                        
        input = torch.cat((joints, head_directions), dim=2)
        if object_num > 0:
            object_positions = data.clone()[:, :, (joint_number+2)*3:(joint_number+2+8*object_num*2)*3]
            input = torch.cat((input, object_positions), dim=2)                        
        prediction = net(input, input_n=input_n)            
        
        if opt.is_eval and opt.save_predictions:
            # eye_gaze + joints + head_directions + object_positions + predictions + attended_hand_gt
            prediction = torch.nn.functional.softmax(prediction, dim=2)
            prediction_cpu = torch.cat((eye_gaze, input), dim=2)            
            prediction_cpu = torch.cat((prediction_cpu, prediction), dim=2)
            prediction_cpu = torch.cat((prediction_cpu, attended_hand_gt), dim=2)
            prediction_cpu = prediction_cpu.cpu().data.numpy()
            if len(predictions) == 0:
                predictions = prediction_cpu                
            else:
                predictions = np.concatenate((predictions, prediction_cpu), axis=0)
                
        attended_hand_gt = attended_hand_gt.reshape(batch_size*input_n)
        attended_hand_baseline = attended_hand_baseline.reshape(batch_size*input_n)
        prediction = prediction.reshape(-1, 2)        
        loss = criterion(prediction, attended_hand_gt)
                
        if is_train == 1:            
            optimizer.zero_grad()
            loss.backward()                        
            optimizer.step()

        # calculate prediction accuracy
        _, y_prd = torch.max(prediction.data, 1)
        acc = torch.sum(y_prd == attended_hand_gt)/(batch_size*input_n)
        prediction_acc_average += acc.cpu().data.numpy() * batch_size*input_n
        
        acc = torch.sum(attended_hand_gt == attended_hand_baseline)/(batch_size*input_n)
        baseline_acc_average += acc.cpu().data.numpy() * batch_size*input_n
                    
    result = {}
    result["baseline_acc_average"] = baseline_acc_average / n
    result["prediction_acc_average"] = prediction_acc_average / n
        
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