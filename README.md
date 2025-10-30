# servobot-rl-genesis

To initialize and update the conda environment:
`conda env update`
`conda activate genesis`

To train the robot for 1000 iterations:
`python train.py config/default.yaml --max_iterations 1001`

To view logs:
`tensorboard --logdir logs`

To drive the robot with a ps4 controller:
`python eval.py --ckpt 100 --teleop ps4`
