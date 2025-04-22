python attended_hand_recognition_hot3d.py --data_dir /scratch/hu/pose_forecast/hot3d_hoigaze/ --ckpt ./checkpoints/hot3d/ --cuda_idx cuda:2 --test_user_id 3 --seq_len 15 --sample_rate 8 --gcn_dropout 0.3 --residual_gcns_num 2 --gamma 0.95 --epoch 60 --object_num 1 --weight_decay 0.05;

python attended_hand_recognition_hot3d.py --data_dir /scratch/hu/pose_forecast/hot3d_hoigaze/ --ckpt ./checkpoints/hot3d/ --cuda_idx cuda:2 --test_user_id 3 --seq_len 15 --sample_rate 8 --gcn_dropout 0.3 --residual_gcns_num 2 --gamma 0.95 --epoch 60 --object_num 1 --weight_decay 0.05 --is_eval --save_predictions;

python gaze_estimation_hot3d.py --data_dir /scratch/hu/pose_forecast/hot3d_hoigaze/ --ckpt ./checkpoints/hot3d/ --cuda_idx cuda:2 --test_user_id 3 --seq_len 15 --residual_gcns_num 4 --gamma 0.95 --learning_rate 0.005 --epoch 80 --object_num 1;

python gaze_estimation_hot3d.py --data_dir /scratch/hu/pose_forecast/hot3d_hoigaze/ --ckpt ./checkpoints/hot3d/ --cuda_idx cuda:2 --test_user_id 3 --seq_len 15 --residual_gcns_num 4 --gamma 0.95 --learning_rate 0.005 --epoch 80 --object_num 1 --is_eval;