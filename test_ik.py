from src.env.ik_solver import convert_ta_tl_to_160_225
import math

l0, l1, l2, l3 = 8.5, 6.2, 8.7, 16.63

for offset in [0, 90, -90, 180]:
    print(f"--- Offset {offset} ---")
    # Simulate a few sim angles around 0 (e.g. -10, 0, 10)
    for sim_angle in [-10, 0, 10]:
        ik_a = sim_angle + offset
        ik_l = sim_angle + offset
        res = convert_ta_tl_to_160_225(ik_a, ik_l, l0, l1, l2, l3)
        if res is not None:
            print(f"Sim {sim_angle} -> IK {ik_a}: t_c = {res:.4f} rad ({math.degrees(res):.1f} deg)")
        else:
            print(f"Sim {sim_angle} -> IK {ik_a}: IMPOSSIBLE")
