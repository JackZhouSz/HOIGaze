# visualise data in the ADT dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# play human pose using a skeleton
class Player_Skeleton:
    def __init__(self, fps=30.0, object_num=10):
    
        self._fps = fps
        self.object_num = object_num
        # names of all the joints: head + left_hand + right_hand + left_hand_joint + right_hand_joint + gaze_direction + head_direction
        self._joint_names = ['Head', 'LHand', 'RHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1', 'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', 'LPinky1', 'LPinky2', 'LPinky3', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1', 'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3', 'Gaze_direction', 'Head_direction']
        
        self._joint_ids = {name: idx for idx, name in enumerate(self._joint_names)}
        
        # parent of every joint
        self._joint_parent_names = {
            # root                    
            'Head': 'Head',
            'LHand': 'LHand',
            'RHand': 'RHand',
            'LThumb1': 'LHand',
            'LThumb2': 'LThumb1',
            'LThumb3': 'LThumb2',
            'LIndex1': 'LHand',
            'LIndex2': 'LIndex1',
            'LIndex3': 'LIndex2',
            'LMiddle1': 'LHand',
            'LMiddle2': 'LMiddle1',
            'LMiddle3': 'LMiddle2',
            'LRing1': 'LHand',
            'LRing2': 'LRing1',
            'LRing3': 'LRing2',
            'LPinky1': 'LHand',
            'LPinky2': 'LPinky1',
            'LPinky3': 'LPinky2',
            'RThumb1': 'RHand',
            'RThumb2': 'RThumb1',
            'RThumb3': 'RThumb2',
            'RIndex1': 'RHand',
            'RIndex2': 'RIndex1',
            'RIndex3': 'RIndex2',
            'RMiddle1': 'RHand',
            'RMiddle2': 'RMiddle1',
            'RMiddle3': 'RMiddle2',
            'RRing1': 'RHand',
            'RRing2': 'RRing1',
            'RRing3': 'RRing2',
            'RPinky1': 'RHand',
            'RPinky2': 'RPinky1',
            'RPinky3': 'RPinky2',
            'Gaze_direction': 'Head',
            'Head_direction': 'Head',}
            
        # id of joint parent
        self._joint_parent_ids = [self._joint_ids[self._joint_parent_names[child_name]] for child_name in self._joint_names]
        self._joint_links = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]        
        # colors: 0 for head, 1 for left, 2 for right
        self._link_colors = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4]
        
        self._fig = plt.figure()
        self._ax = plt.gca(projection='3d')
            
        self._plots = []
        for i in range(len(self._joint_links)):
            if self._link_colors[i] == 0:
                color = "#3498db"
            if self._link_colors[i] == 1:
                color = "#3498db"
            if self._link_colors[i] == 2:
                color = "#3498db"
            if self._link_colors[i] == 3:
                color = "#6aa84f"
            if self._link_colors[i] == 4:
                color = "#a64d79"                
            self._plots.append(self._ax.plot([0, 0], [0, 0], [0, 0], lw=2.0, c=color))

        for i in range(self.object_num):
            self._plots.append(self._ax.plot([0, 0], [0, 0], [0, 0], lw=1.0, c='#ff0000'))

        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")
        self._ax.set_zlabel("z")

    # play the sequence of human pose in xyz representations
    def play_xyz(self, pose_xyz, gaze, head, objects):
        gaze_direction = pose_xyz[:, :3] + gaze[:, :3]*0.5
        head_direction = pose_xyz[:, :3] + head[:, :3]*0.5
        pose_xyz = np.concatenate((pose_xyz, gaze_direction), axis = 1)        
        pose_xyz = np.concatenate((pose_xyz, head_direction), axis = 1)        
        
        for i in range(pose_xyz.shape[0]):        
            joint_number = len(self._joint_names)
            pose_xyz_tmp = pose_xyz[i].reshape(joint_number, 3)
            objects_xyz = objects[i, :, :, :]
            for j in range(len(self._joint_links)):
                idx = self._joint_links[j]
                start_point = pose_xyz_tmp[idx]
                end_point = pose_xyz_tmp[self._joint_parent_ids[idx]]
                x = np.array([start_point[0], end_point[0]])
                y = np.array([start_point[2], end_point[2]])
                z = np.array([start_point[1], end_point[1]])
                self._plots[j][0].set_xdata(x)
                self._plots[j][0].set_ydata(y)
                self._plots[j][0].set_3d_properties(z)

            for j in range(len(self._joint_links), len(self._joint_links) + objects_xyz.shape[0]):
                object_xyz = objects_xyz[j - len(self._joint_links), :, :]
                self._plots[j][0].set_xdata(object_xyz[:, 0])
                self._plots[j][0].set_ydata(object_xyz[:, 2])
                self._plots[j][0].set_3d_properties(object_xyz[:, 1])                
                                                                      
            r = 1.0
            x_root, y_root, z_root = pose_xyz_tmp[0, 0], pose_xyz_tmp[0, 2], pose_xyz_tmp[0, 1]
            self._ax.set_xlim3d([-r + x_root, r + x_root])
            self._ax.set_ylim3d([-r + y_root, r + y_root])
            self._ax.set_zlim3d([-r + z_root, r + z_root])
            #self._ax.view_init(elev=30, azim=-110)

            self._ax.grid(False)
            #self._ax.axis('off')
            
            self._ax.set_aspect('auto')
            plt.show(block=False)
            self._fig.canvas.draw()
            past_time = f"{i / self._fps:.1f}"
            plt.title(f"Time: {past_time} s", fontsize=15)
            plt.pause(0.000000001)

            
if __name__ == "__main__":
    data_path = '/scratch/hu/pose_forecast/adt_hoigaze/test/Apartment_release_meal_skeleton_seq132_'
    gaze_path = data_path + 'gaze.npy'
    head_path = data_path + 'head.npy'
    hand_path = data_path + 'hand.npy'
    hand_joint_path = data_path + 'handjoints.npy'        
    object_left_hand_path = data_path + 'object_left.npy'
    object_right_hand_path = data_path + 'object_right.npy'
    
    gaze = np.load(gaze_path) # gaze_direction (3) + gaze_2d (2) + frame_id (1)
    print("Gaze shape: {}".format(gaze.shape))    
    gaze_direction = gaze[:, :3]
    head = np.load(head_path) # head_direction (3) + head_translation (3)
    print("Head shape: {}".format(head.shape))
    head_direction = head[:, :3]    
    head_translation = head[:, 3:]    
    hand_translation = np.load(hand_path) # left_hand_translation (3) + right_hand_translation (3)
    print("Hand shape: {}".format(hand_translation.shape))    
    hand_joint = np.load(hand_joint_path) # left_hand (15*3) + right_hand (15*3) + hand_dominance + closest_hand
    print("Hand joint shape: {}".format(hand_joint.shape))      
    hand_joint = hand_joint[:, :90]    
    pose = np.concatenate((head_translation, hand_translation), axis=1)
    pose = np.concatenate((pose, hand_joint), axis=1)
    object_left = np.load(object_left_hand_path)[:, :, :, :]
    object_right = np.load(object_right_hand_path)[:, :, :, :]
    object_all = np.concatenate((object_left, object_right), axis=1)
    print("Object shape: {}".format(object_all.shape))
    
    player = Player_Skeleton(object_num = object_all.shape[1])
    player.play_xyz(pose, gaze_direction, head_direction, object_all)