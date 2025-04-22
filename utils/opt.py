import os
import argparse
from pprint import pprint


class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None
        
    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--cuda_idx', type=str, default='cuda:0', help='cuda idx')
        self.parser.add_argument('--data_dir', type=str,
                                 default='./dataset/',
                                 help='path to dataset')
        self.parser.add_argument('--is_eval', dest='is_eval', action='store_true',
                                 help='whether to evaluate existing models or not')                                 
        self.parser.add_argument('--ckpt', type=str, default='./checkpoints/', help='path to save checkpoints')               
        self.parser.add_argument('--test_user_id', type=int, default=1, help='id of the test participants')        
        self.parser.add_argument('--actions', type=str, default='all', help='actions to use')
        self.parser.add_argument('--sample_rate', type=int, default=2, help='sample the data')
        self.parser.add_argument('--save_predictions', dest='save_predictions', action='store_true',
                                 help='whether to save the prediction results or not')
        # ===============================================================
        #                     Model options
        # ===============================================================
        self.parser.add_argument('--body_joint_number', type=int, default=3, help='number of body joints to use')        
        self.parser.add_argument('--hand_joint_number', type=int, default=20, help='number of hand joints to use')
        self.parser.add_argument('--head_cnn_channels', type=int, default=32, help='number of channels used in the head_CNN')        
        self.parser.add_argument('--gcn_latent_features', type=int, default=8, help='number of latent features used in the gcn')
        self.parser.add_argument('--residual_gcns_num', type=int, default=4, help='number of residual gcns to use')
        self.parser.add_argument('--gcn_dropout', type=float, default=0.3, help='drop out probability in the gcn')
        self.parser.add_argument('--gaze_cnn_channels', type=int, default=64, help='number of channels used in the gaze_CNN')        
        self.parser.add_argument('--recognition_cnn_channels', type=int, default=64, help='number of channels used in the recognition_CNN')        
        self.parser.add_argument('--object_num', type=int, default=1, help='number of scene objects for gaze estimation')
        self.parser.add_argument('--use_self_att', type=int, default=1, help='use self attention or not')
        self.parser.add_argument('--self_att_head_num', type=int, default=1, help='number of heads used in self attention')
        self.parser.add_argument('--self_att_dropout', type=float, default=0.1, help='drop out probability in self attention')        
        self.parser.add_argument('--use_cross_att', type=int, default=1, help='use cross attention or not')
        self.parser.add_argument('--cross_att_head_num', type=int, default=1, help='number of heads used in cross attention')
        self.parser.add_argument('--cross_att_dropout', type=float, default=0.1, help='drop out probability in cross attention')
        self.parser.add_argument('--use_attended_hand', type=int, default=1, help='use attended hand or use both hands')
        self.parser.add_argument('--use_attended_hand_gt', type=int, default=0, help='use attended hand ground truth or not')
        # ===============================================================
        #                     Running options
        # ===============================================================       
        self.parser.add_argument('--seq_len', type=int, default=15, help='the length of the used sequence')
        self.parser.add_argument('--learning_rate', type=float, default=0.005)
        self.parser.add_argument('--gaze_head_loss_factor', type=float, default=4.0)
        self.parser.add_argument('--gaze_head_cos_threshold', type=float, default=0.8)                
        self.parser.add_argument('--weight_decay', type=float, default=0.0)
        self.parser.add_argument('--gamma', type=float, default=0.95, help='decay learning rate by gamma')
        self.parser.add_argument('--epoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--validation_epoch', type=int, default=10, help='interval of epoches to test')
        self.parser.add_argument('--test_batch_size', type=int, default=32)
        
    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, make_dir=True):
        self._initial()
        self.opt = self.parser.parse_args()               
        ckpt = self.opt.ckpt
        if make_dir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
        self._print()
        return self.opt