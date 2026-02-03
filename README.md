# servobot-rl-genesis

## Setup

To initialize the conda environment, choose the appropriate file for your platform:

**For Linux/Windows with NVIDIA GPU (CUDA support):**
```bash
conda env create -f environment-cuda.yml
conda activate genesis
```

**For macOS (CPU only):**
```bash
conda env create -f environment-macos.yml
conda activate genesis
```

To update an existing environment:
```bash
# For CUDA systems
conda env update -f environment-cuda.yml

# For macOS
conda env update -f environment-macos.yml
```

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

## Config Schemas

There exists a JSON schema for the configuration of training at [config/schemas/config.json](config/schemas/config.json). This allows for intellisense
to be used when editing the config to mitigate improper formatting. Additionally, the types of the config objects when imported from yaml files
are provided at [src/config/config.py](src/config/config.py), which are generated via jsonschema-gentypes. To regenerate the config types when 
changing the config schema, run the following command at the top level of the repository:

```bash
jsonschema-gentypes --json-schema config/schemas/config.json --python src/config/config.py
```
