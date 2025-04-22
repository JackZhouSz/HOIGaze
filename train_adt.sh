python attended_hand_recognition_adt.py --data_dir /scratch/hu/pose_forecast/adt_hoigaze/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:6 --seq_len 15 --sample_rate 2 --residual_gcns_num 2 --gamma 0.95 --learning_rate 0.005 --epoch 60 --object_num 1 --hand_joint_number 1;

python attended_hand_recognition_adt.py --data_dir /scratch/hu/pose_forecast/adt_hoigaze/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:6 --seq_len 15 --sample_rate 2 --residual_gcns_num 2 --gamma 0.95 --learning_rate 0.005 --epoch 60 --object_num 1 --hand_joint_number 1 --is_eval --save_predictions;

python gaze_estimation_adt.py --data_dir /scratch/hu/pose_forecast/adt_hoigaze/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:6 --seq_len 15 --residual_gcns_num 4 --gamma 0.8 --learning_rate 0.005 --epoch 80 --object_num 1 --hand_joint_number 1;

python gaze_estimation_adt.py --data_dir /scratch/hu/pose_forecast/adt_hoigaze/ --ckpt ./checkpoints/adt/ --cuda_idx cuda:6 --seq_len 15 --residual_gcns_num 4 --gamma 0.8 --learning_rate 0.005 --epoch 80 --object_num 1 --hand_joint_number 1 --is_eval;