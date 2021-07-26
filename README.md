1. Request access to the BEHAVIOR assets using [this form](https://forms.gle/ecyoPtEcCBMrQ3qF9):
- Fill out the license agreement as suggested in the form
- When done, copy the key you receive (igibson.key) into the root directory of this repository
- Download the behavior data bundle (ig_dataset)
- unzip behavior_data_bundle.zip 
2. Run the experiment
- bash run_ray.sh
- bash run_sb3.sh
3. View the results with tensorboard 
- Install tensoboard
- tensorboard --logdir ./results/stable-baselines3
- tensorboard --logdir ./results/ray_example
