# servobot-rl-genesis

## Setup

To initialize the virtual environment, use the [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager.

```bash
uv venv
source .venv/bin/activate
uv sync
```

This will install the necessary dependencies, as well as the scripts for training and evaluation, which can be run as so:

```bash
# to train with the given config
train --config_path=config/servobot.yaml -n 1

# to evaluate the latest saved model
eval --input=keyboard
```

To conduct teacher student training: 
`train config\distill.yaml --resume "saved_models\servobot-energy\model_6800.pt"`
- Make sure rsl-rl-lib version is at least 3.1.3 (otherwise teacher model has wrong number of input neurons)
- Must resume from either normally trained model (loaded as teacher) or from teacher student model (both teacher and student loaded)
- If using an RNN-equiped model as teacher, set teacher_recurrent: true
- If needed to remove RNN layer of student, set class_name: StudentTeacher
