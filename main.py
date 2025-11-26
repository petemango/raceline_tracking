from sys import argv
import numpy as np

from simulator import RaceTrack, Simulator, plt


def offset_raceline_inward(raceline, centerline, offset_distance=1.5):
    """
    Offset the raceline inward (toward centerline) to account for car width.
    
    Args:
        raceline: Nx2 array of raceline points
        centerline: Nx2 array of centerline points  
        offset_distance: How much to move toward center (meters)
    """
    offset_raceline = np.copy(raceline)
    
    for i in range(len(raceline)):
        # Find closest centerline point
        dists = np.linalg.norm(centerline - raceline[i], axis=1)
        closest_center_idx = np.argmin(dists)
        
        # Direction from raceline to centerline (inward)
        to_center = centerline[closest_center_idx] - raceline[i]
        dist_to_center = np.linalg.norm(to_center)
        
        if dist_to_center > 0.1:  # Only offset if not already at center
            # Normalize and offset
            direction = to_center / dist_to_center
            # Offset proportionally - more offset when further from center
            actual_offset = min(offset_distance, dist_to_center * 0.4)
            offset_raceline[i] = raceline[i] + direction * actual_offset
    
    return offset_raceline


if __name__ == "__main__":
    assert(len(argv) == 3)
    racetrack = RaceTrack(argv[1])
    raceline_path = argv[2]
    
    # Load the optimal raceline
    raceline = np.loadtxt(raceline_path, comments="#", delimiter=",")
    
    # Offset raceline inward to account for car width (larger offset for safety)
    raceline_offset = offset_raceline_inward(raceline, racetrack.centerline, offset_distance=2.0)
    
    simulator = Simulator(racetrack, raceline_offset)
    simulator.start()
    plt.show()