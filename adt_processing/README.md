## Code to process the ADT dataset

Note: processing the ADT dataset is much more complicated than other datasets because it relies on the Project Aria Tools. It would be easier to get started with other datasets first.


## Usage:
Step 1: Follow https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/dataset_download to prepare the environment and download the dataset. Please note that in our paper we used the 1.1.0 version of the dataset with Project Aria Tools 1.1.0 and Python 3.8. We use 'python setup.py build_py' to build Project Aria Tools.

Step 2: Set 'dataset_path' and 'dataset_processed_path' in 'adt_preprocessing.py', put 'adt_preprocessing.py', 'adt.csv', and 'utils' into the codebase of the Project Aria Tools, and run it to process the dataset.

Step 3: It is optional but highly recommended to set 'data_path' in 'dataset_visualisation.py' to visualise and get familiar with the dataset.


## Citations

```bibtex
@inproceedings{hu25hoigaze,
	title={HOIGaze: Gaze Estimation During Hand-Object Interactions in Extended Reality Exploiting Eye-Hand-Head Coordination},
	author={Hu, Zhiming and Haeufle, Daniel and Schmitt, Syn and Bulling, Andreas},
	booktitle={Proceedings of the ACM Special Interest Group on Computer Graphics and Interactive Techniques},
	year={2025},
	pages = {1--10}}
	
@inproceedings{pan2023aria,
	title={Aria digital twin: A new benchmark dataset for egocentric 3d machine perception},
	author={Pan, Xiaqing and Charron, Nicholas and Yang, Yongqian and Peters, Scott and Whelan, Thomas and Kong, Chen and Parkhi, Omkar and Newcombe, Richard and Ren, Yuheng Carl},
	booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
	pages={20133--20143},
	year={2023}}
```
