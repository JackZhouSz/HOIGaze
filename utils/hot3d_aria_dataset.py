from torch.utils.data import Dataset
import numpy as np
import os


class hot3d_aria_dataset(Dataset):
    def __init__(self, data_dir, subjects, seq_len, actions = 'all', object_num=1, sample_rate=1):
        if actions == 'all':
            actions = ['room', 'kitchen', 'office']        
        self.sample_rate = sample_rate
        self.dataset = self.load_data(data_dir, subjects, seq_len, actions, object_num)
        
    def load_data(self, data_dir, subjects, seq_len, actions, object_num):
        dataset = []
        file_names = sorted(os.listdir(data_dir))
        gaze_file_names = []
        hand_file_names = []                
        hand_joint_file_names = []
        head_file_names = []
        object_left_file_names = []
        object_right_file_names = []
        for name in file_names:
            name_split = name.split('_')
            subject = name_split[0]
            action = name_split[2]
            if subject in subjects and action in actions:
                data_type = name_split[-1][:-4]
                if(data_type == 'gaze'):
                    gaze_file_names.append(name)
                if(data_type == 'hand'):
                    hand_file_names.append(name)
                if(data_type == 'handjoints'):
                    hand_joint_file_names.append(name)
                if(data_type == 'head'):
                    head_file_names.append(name)
                if(data_type == 'bbxleft'):
                    object_left_file_names.append(name)
                if(data_type == 'bbxright'):
                    object_right_file_names.append(name)
                    
        segments_number = len(hand_file_names)
        # print("segments number {}".format(segments_number))
        for i in range(segments_number):
            gaze_data_path = data_dir + gaze_file_names[i]
            gaze_data = np.load(gaze_data_path)          
            num_frames = gaze_data.shape[0]
            if num_frames < seq_len:
                continue            
            hand_data_path = data_dir + hand_file_names[i]
            hand_data = np.load(hand_data_path)
            hand_joint_data_path = data_dir + hand_joint_file_names[i]
            hand_joint_data_all = np.load(hand_joint_data_path)
            hand_joint_data = hand_joint_data_all[:, :120]            
            attended_hand_gt = hand_joint_data_all[:, 120:121]            
            attended_hand_baseline = hand_joint_data_all[:, 121:122]       
            
            head_data_path = data_dir + head_file_names[i]
            head_data = np.load(head_data_path)            
            object_left_data_path = data_dir + object_left_file_names[i]
            object_left_data = np.load(object_left_data_path)
            object_right_data_path = data_dir + object_right_file_names[i]
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

            #object_left_positions = np.mean(object_left_bbx.reshape(num_frames, object_num, 8, 3), axis=2).reshape(num_frames, -1)
            #object_right_positions = np.mean(object_right_bbx.reshape(num_frames, object_num, 8, 3), axis=2).reshape(num_frames, -1)
            
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
            seq_sel = seq_sel[0::self.sample_rate, :, :]
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
    data_dir = "/scratch/hu/pose_forecast/hot3d_hoigaze/"
    seq_len = 15
    actions = 'all'
    all_subjects = ['P0001', 'P0002', 'P0003', 'P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']
    train_subjects = ['P0009', 'P0010', 'P0011', 'P0012', 'P0014', 'P0015']    
    object_num = 1
    sample_rate = 10
    
    train_dataset = hot3d_aria_dataset(data_dir, train_subjects, seq_len, actions, object_num, sample_rate)
    print("Training data size: {}".format(train_dataset.dataset.shape))
    
    hand_joint_dominance = train_dataset[:, :, -2:-1].flatten()
    print("right hand ratio: {:.2f}".format(np.sum(hand_joint_dominance)/hand_joint_dominance.shape[0]*100))
    
    #test_subjects = ['P0001', 'P0002', 'P0003']
    #sample_rate = 8
    #test_dataset = hot3d_aria_dataset(data_dir, test_subjects, seq_len, actions, #object_num, sample_rate)
   # print("Test data size: {}".format(test_dataset.dataset.shape))
    
    #hand_joint_dominance = test_dataset[:, :, -2:-1].flatten()
    #print("right hand ratio: {:.2f}".format(np.sum(hand_joint_dominance)/hand_joint_dominance.shape[0]*100))       