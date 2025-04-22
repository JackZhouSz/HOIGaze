from torch.utils.data import Dataset
import numpy as np
import os


class hot3d_aria_dataset(Dataset):
    def __init__(self, data_path, seq_len, object_num=1):
        self.dataset = self.load_data(data_path, seq_len, object_num)
        
    def load_data(self, data_path, seq_len, object_num):
        dataset = []       
        gaze_file_name = data_path + 'gaze.npy'
        hand_file_name = data_path + 'hand.npy'
        hand_joint_file_name = data_path + 'handjoints.npy'
        head_file_name = data_path + 'head.npy'
        object_left_file_name = data_path + 'object_bbxleft.npy'
        object_right_file_name = data_path + 'object_bbxright.npy'
        
        gaze_data_path = gaze_file_name
        gaze_data = np.load(gaze_data_path)          
        num_frames = gaze_data.shape[0]
        hand_data_path = hand_file_name
        hand_data = np.load(hand_data_path)
        hand_joint_data_path = hand_joint_file_name
        hand_joint_data_all = np.load(hand_joint_data_path)
        hand_joint_data = hand_joint_data_all[:, :120]            
        attended_hand_gt = hand_joint_data_all[:, 120:121]            
        attended_hand_baseline = hand_joint_data_all[:, 121:122]       
        
        head_data_path = head_file_name
        head_data = np.load(head_data_path)            
        object_left_data_path = object_left_file_name
        object_left_data = np.load(object_left_data_path)
        object_right_data_path = object_right_file_name
        object_right_data = np.load(object_right_data_path)
        
        left_hand_translation = hand_data[:, 0:3]
        right_hand_translation = hand_data[:, 22:25]
        head_direction = head_data[:, 0:3]
        head_translation = head_data[:, 3:6]
        gaze_direction = gaze_data[:, 0:3]
        object_left_bbx = []
        object_right_bbx = []
        for item in range(object_num):
            left_bbx = object_left_data[:, item*24:item*24+24]
            right_bbx = object_right_data[:, item*24:item*24+24]
            if len(object_left_bbx) == 0:
                object_left_bbx = left_bbx
                object_right_bbx = right_bbx
            else:
                object_left_bbx = np.concatenate((object_left_bbx, left_bbx), axis=1)
                object_right_bbx = np.concatenate((object_right_bbx, right_bbx), axis=1)
        
        data = gaze_direction
        data = np.concatenate((data, left_hand_translation), axis=1)
        data = np.concatenate((data, right_hand_translation), axis=1)
        data = np.concatenate((data, head_translation), axis=1)   
        data = np.concatenate((data, hand_joint_data), axis=1)            
        data = np.concatenate((data, head_direction), axis=1)
        if object_num > 0:
            data = np.concatenate((data, object_left_bbx), axis=1)
            data = np.concatenate((data, object_right_bbx), axis=1)            
        data = np.concatenate((data, attended_hand_gt), axis=1)
        data = np.concatenate((data, attended_hand_baseline), axis=1)
                
        fs = np.arange(0, num_frames - seq_len + 1)
        fs_sel = fs
        for i in np.arange(seq_len - 1):
            fs_sel = np.vstack((fs_sel, fs + i + 1))
        fs_sel = fs_sel.transpose()
        seq_sel = data[fs_sel, :]
        seq_sel = seq_sel[0::seq_len, :, :]
        if len(dataset) == 0:
            dataset = seq_sel
        else:
            dataset = np.concatenate((dataset, seq_sel), axis=0)
        return dataset
        
    def __len__(self):
        return np.shape(self.dataset)[0]

    def __getitem__(self, item):
        return self.dataset[item]

        
if __name__ == "__main__":
    data_path = '/scratch/hu/pose_forecast/hot3d_hoigaze/P0001_10a27bf7_room_721_890_'
    seq_len = 15
    object_num = 1    
    train_dataset = hot3d_aria_dataset(data_path, seq_len, object_num)
    print("Training data size: {}".format(train_dataset.dataset.shape))