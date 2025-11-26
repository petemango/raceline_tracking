#!/usr/bin/env python3
"""
Controller tuning script - runs simulation and analyzes performance.
"""
import numpy as np
import csv
import os
import json

LOG_FILE = "controller_log.csv"

def analyze_log(log_file=LOG_FILE):
    """Analyze the controller log file."""
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found")
        return None
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not data:
        print("No data in log file")
        return None
    
    velocities = [float(d['velocity']) for d in data]
    errors = [float(d['cross_track_error']) for d in data]
    curvatures = [float(d['curvature']) for d in data]
    
    # Find high cross-track error events (potential track violations)
    # Track limits are typically 5-6m from centerline
    high_error_threshold = 4.0  # meters - conservative
    high_error_events = [(i, float(d['cross_track_error']), float(d['velocity']), float(d['curvature'])) 
                         for i, d in enumerate(data) if float(d['cross_track_error']) > high_error_threshold]
    
    # Analyze errors by curvature
    corner_errors = [(e, c) for e, c in zip(errors, curvatures) if c > 0.01]
    straight_errors = [(e, c) for e, c in zip(errors, curvatures) if c <= 0.001]
    
    results = {
        'total_samples': len(data),
        'duration': float(data[-1]['time']) if data else 0,
        'avg_velocity': np.mean(velocities),
        'max_velocity': np.max(velocities),
        'min_velocity': np.min(velocities),
        'avg_cross_track_error': np.mean(errors),
        'max_cross_track_error': np.max(errors),
        'high_error_events': len(high_error_events),
        'high_error_details': high_error_events[:10],
        'avg_corner_error': np.mean([e for e, c in corner_errors]) if corner_errors else 0,
        'avg_straight_error': np.mean([e for e, c in straight_errors]) if straight_errors else 0,
    }
    
    return results


def print_analysis(results):
    """Print analysis results."""
    if results is None:
        return
    
    print("\n" + "="*60)
    print("CONTROLLER PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Total samples: {results['total_samples']}")
    print()
    print("VELOCITY:")
    print(f"  Average: {results['avg_velocity']:.2f} m/s ({results['avg_velocity']*3.6:.1f} km/h)")
    print(f"  Maximum: {results['max_velocity']:.2f} m/s ({results['max_velocity']*3.6:.1f} km/h)")
    print(f"  Minimum: {results['min_velocity']:.2f} m/s ({results['min_velocity']*3.6:.1f} km/h)")
    print()
    print("CROSS-TRACK ERROR:")
    print(f"  Average: {results['avg_cross_track_error']:.3f} m")
    print(f"  Maximum: {results['max_cross_track_error']:.3f} m")
    print(f"  Avg in corners (curvature>0.01): {results['avg_corner_error']:.3f} m")
    print(f"  Avg on straights (curvature<0.001): {results['avg_straight_error']:.3f} m")
    print(f"  High error events (>4m): {results['high_error_events']}")
    print()
    
    if results['high_error_events'] > 0:
        print("HIGH ERROR DETAILS (potential track violations):")
        for idx, error, vel, curv in results['high_error_details']:
            print(f"  Sample {idx}: error={error:.2f}m, velocity={vel:.1f}m/s, curvature={curv:.6f}")
    
    # Performance score (higher is better)
    # Penalize high cross-track errors, reward high velocity
    penalty = results['max_cross_track_error'] * 10 + results['high_error_events'] * 20
    score = results['avg_velocity'] * 2 - penalty
    
    print()
    print(f"PERFORMANCE SCORE: {score:.1f}")
    print("  (Higher is better. Score = avg_vel*2 - max_error*10 - high_events*20)")
    print("="*60)
    
    return score


def get_tuning_recommendations(results):
    """Get recommendations for tuning based on analysis."""
    recommendations = []
    
    if results is None:
        return ["Run simulation first to get data"]
    
    if results['max_cross_track_error'] > 4.0:
        recommendations.append(
            f"Max error is {results['max_cross_track_error']:.2f}m - REDUCE grip_margin or max_lateral_accel"
        )
    
    if results['high_error_events'] > 0:
        recommendations.append(
            f"Found {results['high_error_events']} high error events - INCREASE curvature_lookahead"
        )
    
    if results['avg_corner_error'] > 2.0:
        recommendations.append(
            f"Corner error is high ({results['avg_corner_error']:.2f}m) - REDUCE min_velocity or grip_margin"
        )
    
    if results['max_cross_track_error'] < 3.0 and results['high_error_events'] == 0:
        recommendations.append(
            "Tracking is good! You can try INCREASING grip_margin or max_lateral_accel for speed"
        )
    
    if results['avg_velocity'] < 40:
        recommendations.append(
            "Average velocity is low - INCREASE max_straight_velocity or grip_margin"
        )
    
    return recommendations


if __name__ == "__main__":
    import sys
    
    results = analyze_log()
    score = print_analysis(results)
    
    if results:
        print("\nTUNING RECOMMENDATIONS:")
        for rec in get_tuning_recommendations(results):
            print(f"  • {rec}")
        print()
