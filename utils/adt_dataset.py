from torch.utils.data import Dataset
import numpy as np
import os


class adt_dataset(Dataset):
    def __init__(self, data_dir, seq_len, actions = 'all', train_flag = 1, object_num=1, hand_joint_number=1, sample_rate=1):
        actions = self.define_actions(actions)
        self.sample_rate = sample_rate
        if train_flag == 1:
            data_dir = data_dir + 'train/'
        if train_flag == 0:
            data_dir = data_dir + 'test/'
            
        self.dataset = self.load_data(data_dir, seq_len, actions, object_num, hand_joint_number)
        
    def define_actions(self, action):
        """
        Define the list of actions we are using.

        Args
        action: String with the passed action. Could be "all"
        Returns
        actions: List of strings of actions
        Raises
        ValueError if the action is not included.
        """
        
        actions = ['work', 'decoration', 'meal']
        if action in actions:
            return [action]

        if action == "all":
            return actions
        raise( ValueError, "Unrecognised action: %d" % action )
                
    def load_data(self, data_dir, seq_len, actions, object_num, hand_joint_number):
        action_number = len(actions)
        dataset = []        
        file_names = sorted(os.listdir(data_dir))
        gaze_file_names = {}
        hand_file_names = {}
        hand_joint_file_names = {}
        head_file_names = {}
        object_left_file_names = {}
        object_right_file_names = {}
        for action_idx in np.arange(action_number):
            gaze_file_names[actions[ action_idx ]] = []
            hand_file_names[actions[ action_idx ]] = []
            hand_joint_file_names[actions[ action_idx ]] = []
            head_file_names[actions[ action_idx ]] = []
            object_left_file_names[actions[ action_idx ]] = []
            object_right_file_names[actions[ action_idx ]] = []
            
        for name in file_names:
            name_split = name.split('_')
            action = name_split[2]
            if action in actions:
                data_type = name_split[-1][:-4]
                if(data_type == 'gaze'):
                    gaze_file_names[action].append(name)
                if(data_type == 'hand'):
                    hand_file_names[action].append(name)
                if(data_type == 'handjoints'):
                    hand_joint_file_names[action].append(name)
                if(data_type == 'head'):
                    head_file_names[action].append(name)
                if(data_type == 'bbxleft'):
                    object_left_file_names[action].append(name)
                if(data_type == 'bbxright'):
                    object_right_file_names[action].append(name)
                                                
        for action_idx in np.arange(action_number):
            action = actions[ action_idx ]
            segments_number = len(gaze_file_names[action])
            print("Reading action {}, segments number {}".format(action, segments_number))
            for i in range(segments_number):
                gaze_data_path = data_dir + gaze_file_names[action][i]
                gaze_data = np.load(gaze_data_path)
                gaze_direction = gaze_data[:, :3]
                num_frames = gaze_data.shape[0]
                if num_frames < seq_len:
                    continue                                                        
                hand_data_path = data_dir + hand_file_names[action][i]
                hand_translation = np.load(hand_data_path)
                hand_joint_data_path = data_dir + hand_joint_file_names[action][i]
                hand_joint_data_all = np.load(hand_joint_data_path)
                hand_joint_number_default = 15
                hand_joint_data = hand_joint_data_all[:, :hand_joint_number_default*6]
                left_hand_center = np.mean(hand_joint_data[:, :hand_joint_number_default*3].reshape(hand_joint_data.shape[0], hand_joint_number_default, 3), axis=1)
                right_hand_center = np.mean(hand_joint_data[:, hand_joint_number_default*3:].reshape(hand_joint_data.shape[0], hand_joint_number_default, 3), axis=1)                
                if hand_joint_number == 1:
                    hand_joint_data = np.concatenate((left_hand_center, right_hand_center), axis=1)
                    
                attended_hand_gt = hand_joint_data_all[:, hand_joint_number_default*6:hand_joint_number_default*6+1]
                attended_hand_baseline = hand_joint_data_all[:, hand_joint_number_default*6+1:hand_joint_number_default*6+2]
                
                head_data_path = data_dir + head_file_names[action][i]
                head_data = np.load(head_data_path)
                head_direction = head_data[:, :3]
                head_translation = head_data[:, 3:]
                
                object_left_data_path = data_dir + object_left_file_names[action][i]
                object_left_data = np.load(object_left_data_path)
                object_left_data = object_left_data.reshape(object_left_data.shape[0], -1)
                object_right_data_path = data_dir + object_right_file_names[action][i]
                object_right_data = np.load(object_right_data_path)
                object_right_data = object_right_data.reshape(object_right_data.shape[0], -1)
                
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
                data = np.concatenate((data, hand_translation), axis=1)                
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
                #print(fs_sel)
                seq_sel = data[fs_sel, :]
                seq_sel = seq_sel[0::self.sample_rate, :, :]
                #print(seq_sel.shape)
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
    data_dir = "/scratch/hu/pose_forecast/adt_hoigaze/"
    seq_len = 15
    actions = 'all'
    sample_rate = 1
    train_flag = 1
    object_num = 1
    hand_joint_number = 1
    train_dataset = adt_dataset(data_dir, seq_len, actions, train_flag, object_num, hand_joint_number, sample_rate)
    print("Training data size: {}".format(train_dataset.dataset.shape))
    
    hand_joint_dominance = train_dataset[:, :, -2:-1].flatten()
    print("right hand ratio: {:.2f}".format(np.sum(hand_joint_dominance)/hand_joint_dominance.shape[0]*100))
    