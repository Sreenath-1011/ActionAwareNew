"""
generate_test_video.py - Generates a synthetic test egocentric video with distinct activities
and motion phases to test the ActionAware pipeline end-to-end.
"""

import os
import cv2
import numpy as np

def generate_synthetic_video(output_path="tests/sample_kitchen.mp4", duration_sec=30.0, fps=30.0):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 320, 240
    total_frames = int(duration_sec * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Generating synthetic video '{output_path}' ({duration_sec}s, {total_frames} frames)...")
    
    for f in range(total_frames):
        t = f / fps
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Phase 1 (0-8s): Opening cupboard / taking object
        if t < 8.0:
            color = (0, 140, 255) # Orange
            act_text = "Phase 1: Open cupboard / Take item"
            # Moving hand/object block
            x = int(50 + 100 * np.sin(t * 2))
            y = 120
            cv2.rectangle(frame, (x, y), (x + 60, y + 60), color, -1)
            
        # Phase 2 (8-16s): Cutting / chopping on board (High motion)
        elif t < 16.0:
            color = (50, 205, 50) # Green
            act_text = "Phase 2: Cut vegetable on board (Active)"
            # Rapid vertical chopping motion
            chop_y = int(100 + 40 * np.abs(np.sin(t * 8)))
            cv2.rectangle(frame, (130, chop_y), (190, chop_y + 30), color, -1)
            
        # Phase 3 (16-22s): Idle / static resting
        elif t < 22.0:
            color = (80, 80, 80) # Gray
            act_text = "Phase 3: Idle / waiting"
            cv2.circle(frame, (160, 120), 20, color, -1)
            
        # Phase 4 (22-30s): Washing dishes / cleaning
        else:
            color = (255, 180, 0) # Blue
            act_text = "Phase 4: Wash dish at sink"
            x = int(160 + 50 * np.cos(t * 3))
            y = int(120 + 30 * np.sin(t * 3))
            cv2.circle(frame, (x, y), 35, color, -1)
            
        # Add timestamp & phase overlay
        cv2.putText(frame, f"Time: {t:.2f}s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, act_text, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Video saved to {output_path}.")
    return output_path

if __name__ == "__main__":
    generate_synthetic_video()
