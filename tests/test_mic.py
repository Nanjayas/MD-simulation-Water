import numpy as np

def test_minimum_image_convention():
    box_size = 15.0
    
    # 1. Define our two atoms
    pos_A = 1.0
    pos_B = 14.0
    
    print(f"Box Size: {box_size}")
    print(f"Atom A position: {pos_A}")
    print(f"Atom B position: {pos_B}\n")
    
    # 2. Calculate distance WITHOUT the Minimum Image Convention
    raw_distance = pos_A - pos_B
    print(f"1. Raw Distance (Without MIC): {abs(raw_distance)} Å")
    
    # 3. Calculate distance WITH the Minimum Image Convention
    # Same as in nonbonded.py
    mic_vector = raw_distance - box_size * np.round(raw_distance / box_size)
    print(f"2. MIC Distance (With Ghost):  {abs(mic_vector)} Å")
    
    # Let's break down WHY the math works:
    print("\n--- The Math Behind the Magic ---")
    print(f"Step A: raw_distance / box_size = {raw_distance / box_size}")
    print(f"Step B: round(Step A) = {np.round(raw_distance / box_size)}")
    print("This tells us Atom B's ghost is exactly 1 box length away!")

if __name__ == "__main__":
    test_minimum_image_convention()