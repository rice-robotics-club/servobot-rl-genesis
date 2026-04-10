# Joint Limit Safety Reward Function

## Overview

The joint limit safety reward function protects the simulated leg from moving into positions that would damage a real robot. It applies a sharp quadratic penalty as joint positions approach their mechanical limits.

## Why This Matters

The Catbot leg has physical joint limits beyond which damage would occur:
- **Joint 'a'**: -2.705 to 0.262 radians (-155° to 15°)
- **Joint 'l'**: -1.047 to 2.182 radians (-60° to 125°)

Without this penalty, an untrained RL agent might discover that pushing joints to their limits produces interesting dynamics, leading to unsafe movement patterns that wouldn't work on real hardware.

## How It Works

### The Penalty Function

For each joint, the algorithm:

1. **Normalize position**: Maps the joint position to [0, 1] range where 0 = lower limit and 1 = upper limit
2. **Calculate distance to nearest limit**: How far from the edge (0 at limit, 1 at center)
3. **Apply quadratic penalty within margin**: 
   - Beyond margin: penalty = 0 (no punishment)
   - At margin edge: penalty = 0 (safe zone boundary)
   - At actual limit: penalty = 1 (maximum punishment)

The quadratic penalty is: `penalty = ((margin - distance) / margin)²`

This creates a sharp increase in penalty as the joint approaches its limit:
- At 50% of margin distance from edge: 25% penalty
- At 25% of margin distance from edge: 56% penalty
- At actual limit: 100% penalty

### Aggregation

Penalties from all joints are averaged to create the overall reward signal, so no single joint dominates.

## Configuration

### Environment Config

```yaml
env:
  joint_limit_margin: 0.1  # Normalized margin [0-1], default 0.1 means 10% of joint range
```

This margin is applied to **normalized** joint positions (0-1 range), not absolute radians. A margin of 0.1 means the penalty starts when the joint enters the outer 10% of its range on either side.

**Margin sizes by joint** (with default 0.1 margin):
- Joint 'a' (2.967 rad range): ~0.297 rad margin = ~17°
- Joint 'l' (3.229 rad range): ~0.323 rad margin = ~18.5°

### Reward Config

```yaml
reward:
  rewards:
    joint_limit_safety: -0.5  # Negative because it's a penalty
```

The weight `-0.5` means this penalty contributes -0.5 to the total reward when a joint is at its limit (before scaling by dt). Typical values:
- `-0.5` to `-2.0`: Strong enforcement (recommended for safety-critical training)
- `-0.1` to `-0.5`: Moderate enforcement
- `0.0`: Disabled

## Implementation Details

### Reward Function Location

File: `src/env/catbot_leg.py`, method `_reward_joint_limit_safety()`

```python
def _reward_joint_limit_safety(self):
    # Sharp penalty at joint limits
    # Returns tensor of shape (num_envs,) with penalties in [0, 1]
```

### Joint Limits Source

Limits are extracted from the URDF file at initialization:

```python
self.dofs_limit = torch.tensor(
    [self.robot.get_joint(name).dofs_limit[0] for name in self.joint_names],
    dtype=gs.tc_float,
    device=gs.device,
)
```

This ensures the reward function always uses the actual physical limits from the robot definition.

## Behavior Examples

### Example 1: Safe Motion

```
Position:    0.0 rad (middle of range)
Normalized:  0.5
Distance:    0.5 to limit
Penalty:     0.0 (safe zone)
```

The agent can move freely in the middle of the range.

### Example 2: Approaching Limit

```
Position:    0.12 rad (near upper limit of 0.26)
Normalized:  0.93
Distance:    0.07 to limit (beyond 0.1 margin)
Penalty:     0.0
```

Still safe, but approaching the warning zone.

### Example 3: In Danger Zone

```
Position:    0.20 rad (very close to upper limit)
Normalized:  0.97
Distance:    0.03 to limit (within 0.1 margin)
Penalty:     0.49 (sharp increase!)
```

Agent receives strong negative reward, learns to back away.

### Example 4: At Limit (Damage)

```
Position:    0.262 rad (at upper limit)
Normalized:  1.0
Distance:    0.0 to limit
Penalty:     1.0 (maximum punishment)
```

Agent will aggressively learn to avoid this position.

## Testing

A test script is provided to verify the penalty function:

```bash
python scripts/test_joint_limits.py
```

This script:
- Shows joint limits and margin configuration
- Displays penalty values across the full joint range
- Verifies the quadratic penalty curve
- Confirms the reward function integration

## Training with Joint Limits

When training, the agent learns that:
1. Moving within safe limits (beyond the margin) = 0 penalty
2. Approaching the margin = increasing penalty
3. Touching the limit = severe punishment

This naturally creates a learned safety buffer larger than the minimum margin, providing robust protection against small errors or simulation-to-reality transfer.

## Tuning the Margin

**Too small margin (0.05 or less)**:
- Allows positions closer to actual limits
- Less conservative, but risky for real deployment
- Agent might learn boundary behaviors

**Recommended margin (0.1)**:
- ~10% safe zone on each side
- Balances performance with safety
- Good for most training scenarios

**Large margin (0.2 or more)**:
- Very conservative, large safe zone
- Limits agent mobility
- Good for final deployment safety checks

## Integration with Other Rewards

The joint limit safety penalty is independent and always active when configured. It works alongside other rewards:

```yaml
rewards:
  tracking_vel: 1.0          # Primary objective
  base_height: -0.1          # Secondary constraint
  joint_limit_safety: -0.5   # Safety constraint
  soft_landing: -0.025       # Style constraint
```

The relative weights control the priority. With `-0.5` weight, joint safety is a significant factor preventing unsafe motion.

## Real Robot Considerations

When deploying to real hardware:
1. Joint limits are **hard constraints** (servos won't move beyond)
2. Approaching limits causes **increased power draw** and heating
3. Repeated limit impacts **damage servo gears** over time

The RL penalty function encourages the policy to maintain a safety margin learned during training, protecting the real leg from:
- Accidental limit collisions
- High-current stalls
- Cumulative wear from boundary impacts

## Troubleshooting

### Penalty not activating?
- Check that `joint_limit_safety` is in the reward config
- Verify `joint_limit_margin` is set in env config
- Ensure the reward weight is negative (penalty) or positive (rare inverse penalty)

### Agent still hits limits?
- Increase the margin: `joint_limit_margin: 0.15`
- Increase the weight: `joint_limit_safety: -1.0`
- Check that the agent can physically satisfy all constraints

### Agent too conservative?
- Decrease the margin: `joint_limit_margin: 0.05`
- Decrease the weight: `joint_limit_safety: -0.2`

## References

- Joint limits defined in: `robots/catbot_leg_description/urdf/robot.urdf`
- Reward function: `src/env/catbot_leg.py::_reward_joint_limit_safety()`
- Config schema: `config/schemas/config.json`
- Test script: `scripts/test_joint_limits.py`
