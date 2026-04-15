import re
with open('src/env/catbot_leg.py', 'r') as f:
    text = f.read()

# Replace block 1:
# <<<<<<< ours
#         self.near_limits = torch.zeros(
#             (self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device
# =======
# ...
block_1_pattern = r'<<<<<<< ours\n        self\.near_limits = torch\.zeros\([\s\S]*?=======\n[\s\S]*?>>>>>>> theirs'
# wait, I should just use terminal to open vim, or use python script to replace the blocks cleanly.
