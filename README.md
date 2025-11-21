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

To conduct teacher student training: 
`python train.py config\distill.yaml --resume "saved_models\servobot-energy\model_6800.pt"`
    - Make sure rsl-rl-lib version is at least 3.1.3 (otherwise teacher model has wrong number of input neurons)
    - Must resume from either normally trained model (loaded as teacher) or from teacher student model (both teacher and student loaded)
    - If using an RNN-equiped model as teacher, set teacher_recurrent: true
    - If needed to remove RNN layer of student, set class_name: StudentTeacher