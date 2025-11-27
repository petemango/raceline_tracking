import sys
import numpy as np
# Mock matplotlib to avoid display errors if any imports trigger it
import sys
from unittest.mock import MagicMock
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.path"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()
sys.modules["matplotlib.axes"] = MagicMock()

# Now import the project modules
from racetrack import RaceTrack
from racecar import RaceCar
from controllers import controller, lower_controller, reset_lower_controller_state
from controllers.utils import get_track_errors

def run_simulation(track_file, raceline_file, max_steps=5000):
    print(f"--- Running Simulation for {track_file} ---")
    # We can't use the normal RaceTrack because it imports matplotlib in __init__
    # Wait, looking at racetrack.py, it does:
    # import matplotlib.path as path ...
    # So my mocks above should handle it.
    
    # Reset low-level PID state for a fresh run.
    reset_lower_controller_state()

    try:
        rt = RaceTrack(track_file, raceline_file)
    except Exception as e:
        print(f"Failed to load racetrack: {e}")
        # If it fails due to the mocks not behaving like real mpl objects (e.g. path.Path), 
        # I might need a more robust mock or just comment out the plotting parts in racetrack.py.
        # Let's try running.
        return []

    car = RaceCar(rt.initial_state.T)
    
    history = []

    for step in range(max_steps):
        try:
            desired = controller(car.state, car.parameters, rt)
            cont = lower_controller(car.state, desired, car.parameters)
        except Exception as e:
            print(f"Controller crashed at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break

        car.update(cont)
        
        # Calculate errors for logging
        cte, he = 0.0, 0.0
        if hasattr(rt, 'last_idx') and rt.last_idx is not None:
            # We need to import get_track_errors from controller
            cte, he = get_track_errors(car.state, rt.raceline, known_index=rt.last_idx)
        
        pos = car.state[0:2]
        vel = car.state[3]
        steer = car.state[2]

        steer_rate = cont[0]
        accel = cont[1]

        desired_steer = desired[0]
        desired_vel = desired[1]

        last_idx = getattr(rt, "last_idx", None)

        history.append({
            "step": step,
            "time": step * 0.1,
            "x": pos[0], "y": pos[1],
            "vel": vel,
            "steer": steer,
            "steer_rate": steer_rate,
            "accel": accel,
            "desired_steer": desired_steer,
            "desired_vel": desired_vel,
            "cte": cte,
            "he": he,
            "idx": last_idx
        })
    
    return history

if __name__ == "__main__":
    # IMS
    print("Analyzing IMS...")
    # IMS_raceline.csv might look different or have specific start points
    ims_hist = run_simulation("racetracks/IMS.csv", "racetracks/IMS_raceline.csv", max_steps=100) 
    
    print("\n[IMS Start Analysis (First 3s)]")
    for i in range(0, min(30, len(ims_hist)), 2):
        h = ims_hist[i]
        print(f"T={h['time']:.1f}s | V={h['vel']:.2f} | Steer={h['steer']:.3f} | CTE={h['cte']:.3f} | HE={h['he']:.3f}")

    print("\n[Montreal Analysis]")
    montreal_hist = run_simulation("racetracks/Montreal.csv", "racetracks/Montreal_raceline.csv", max_steps=3000)
    
    # Check for CTE violations and progress
    violations = [h for h in montreal_hist if abs(h['cte']) > 2.0]
    print(f"Steps with CTE > 2.0m: {len(violations)}")
    if violations:
        v = violations[0]
        print(f"First violation at T={v['time']:.1f}s, CTE={v['cte']:.3f}")

    # Print checkpoints
    for t in range(0, 300, 20):
        # Find nearest step
        step_idx = int(t * 10)
        if step_idx < len(montreal_hist):
            h = montreal_hist[step_idx]
            print(f"T={h['time']:.1f}s | V={h['vel']:.2f} | Steer={h['steer']:.3f} | CTE={h['cte']:.3f} | Idx={h['idx']}")

    print("\n[Montreal High CTE Events]")
    max_cte = 0
    max_cte_idx = -1
    for i, h in enumerate(montreal_hist):
        if abs(h['cte']) > max_cte:
            max_cte = abs(h['cte'])
            max_cte_idx = i
            
    if max_cte_idx != -1:
        print(f"Max CTE found at Step {max_cte_idx} (T={montreal_hist[max_cte_idx]['time']:.1f}s)")
        start_window = max(0, max_cte_idx - 5)
        end_window = min(len(montreal_hist), max_cte_idx + 5)
        for i in range(start_window, end_window):
            h = montreal_hist[i]
            print(f"T={h['time']:.1f}s | V={h['vel']:.2f} | Steer={h['steer']:.3f} | CTE={h['cte']:.3f} | HE={h['he']:.3f}")

    # Monza
    print("\n[Monza Analysis]")
    monza_hist = run_simulation("racetracks/Monza.csv", "racetracks/Monza_raceline.csv", max_steps=3000)
    
    violations = [h for h in monza_hist if abs(h['cte']) > 2.0]
    print(f"Steps with CTE > 2.0m: {len(violations)}")
    
    print("\n[Monza High CTE Events]")
    max_cte = 0
    max_cte_idx = -1
    for i, h in enumerate(monza_hist):
        if abs(h['cte']) > max_cte:
            max_cte = abs(h['cte'])
            max_cte_idx = i
            
    if max_cte_idx != -1:
        print(f"Max CTE found at Step {max_cte_idx} (T={monza_hist[max_cte_idx]['time']:.1f}s)")
        start_window = max(0, max_cte_idx - 5)
        end_window = min(len(monza_hist), max_cte_idx + 5)
        for i in range(start_window, end_window):
            h = monza_hist[i]
            print(f"T={h['time']:.1f}s | V={h['vel']:.2f} | Steer={h['steer']:.3f} | CTE={h['cte']:.3f} | HE={h['he']:.3f}")