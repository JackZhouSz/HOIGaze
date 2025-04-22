## Code to process the HOT3D dataset


## Usage:
Step 1: Follow the instructions at the official repository https://github.com/facebookresearch/hot3d to prepare the environment and download the dataset. You should also enable jupyter because the processing codes are run on jupyter.

Step 2: Set 'dataset_path', 'dataset_processed_path', 'object_library_path', and 'mano_hand_model_path' in 'hot3d_aria_preprocessing.ipynb', put 'hot3d_aria_preprocessing.ipynb', 'hot3d_aria_scene.csv', 'hot3d_objects.csv', and 'utils' into the official repository ('hot3d/hot3d/'), and run it to process the dataset.

Step 3: It is optional but highly recommended to set 'data_path', 'object_library_path', and 'mano_hand_model_path' in 'hot3d_aria_visualisation.ipynb', put 'hot3d_aria_visualisation.ipynb' and 'mano_hand_pose_init' into the official repository ('hot3d/hot3d/'), and run it to visualise and get familiar with the dataset.


## Citations

```bibtex
@inproceedings{hu25hoigaze,
	title={HOIGaze: Gaze Estimation During Hand-Object Interactions in Extended Reality Exploiting Eye-Hand-Head Coordination},
	author={Hu, Zhiming and Haeufle, Daniel and Schmitt, Syn and Bulling, Andreas},
	booktitle={Proceedings of the 2025 ACM Special Interest Group on Computer Graphics and Interactive Techniques},
	year={2025}}
	
@article{banerjee2024introducing,
	title={Introducing HOT3D: An Egocentric Dataset for 3D Hand and Object Tracking},
	author={Banerjee, Prithviraj and Shkodrani, Sindi and Moulon, Pierre and Hampali, Shreyas and Zhang, Fan and Fountain, Jade and Miller, Edward and Basol, Selen and Newcombe, Richard and Wang, Robert and others},
	journal={arXiv preprint arXiv:2406.09598},
	year={2024}}
```