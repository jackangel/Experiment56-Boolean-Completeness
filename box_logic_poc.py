import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 4
# Dim 0: Is Fruit
# Dim 1: Is Vegetable
# Dim 2: Is Red
# Dim 3: Is Machine

class BoxGate(nn.Module):
    def __init__(self):
        super().__init__()
        
    def get_intersection_vol(self, box_a, box_b):
        """
        Calculates Intersection Volume between two boxes defined by (min, max).
        """
        # Intersect
        inter_min = torch.max(box_a[0], box_b[0])
        inter_max = torch.min(box_a[1], box_b[1])
        
        # Sides
        sides = F.relu(inter_max - inter_min)
        
        # Volume
        return torch.prod(sides, dim=-1)

    def forward(self, candidates, q_fruit, q_veg, q_red):
        """
        Logic: (Fruit OR Veg) AND (NOT Red)
        """
        # --- 1. THE OR GATE (Multi-Head) ---
        # Head 1: Checks Fruit Intersection
        vol_fruit = self.get_intersection_vol(candidates, q_fruit)
        
        # Head 2: Checks Veg Intersection
        vol_veg = self.get_intersection_vol(candidates, q_veg)
        
        # Logical OR = Sum of volumes (Simplified)
        # If it matches EITHER, score is positive.
        or_score = vol_fruit + vol_veg
        
        # --- 2. THE NOT GATE (Disjointness) ---
        # We check intersection with "Red".
        vol_red = self.get_intersection_vol(candidates, q_red)
        
        # Logical NOT: We want Volume to be ZERO.
        # We create a penalty gate. If vol_red > 0, gate closes.
        # Using a steep exponential decay for the gate: e^(-10 * vol)
        not_gate = torch.exp(-100.0 * vol_red)
        
        # --- 3. FINAL COMBINATION ---
        # (A OR B) * (NOT C)
        final_score = or_score * not_gate
        
        return final_score

# --- SETUP DATA ---

# Helper to make a box (min, max)
def make_box(coords):
    # coords is a center point. We make a small box around it.
    center = torch.tensor(coords, dtype=torch.float32)
    return (center - 0.4, center + 0.4) # Radius 0.4

print("--- LOGIC: (Fruit OR Veg) AND (NOT Red) ---\n")

# DEFINITIONS (Center points)
# Dims: [Fruit, Veg, Red, Machine]

# Candidates (Keys)
# Banana: Fruit=1, Veg=0, Red=0, Mach=0
c_banana = make_box([1.0, 0.0, 0.0, 0.0]) 

# Spinach: Fruit=0, Veg=1, Red=0, Mach=0
c_spinach = make_box([0.0, 1.0, 0.0, 0.0])

# Apple: Fruit=1, Veg=0, Red=1, Mach=0 (TRAP! It is Red)
c_apple   = make_box([1.0, 0.0, 1.0, 0.0])

# Car: Fruit=0, Veg=0, Red=1, Mach=1 (TRAP! Not Fruit/Veg)
c_car     = make_box([0.0, 0.0, 1.0, 1.0])

candidates_min = torch.stack([c_banana[0], c_spinach[0], c_apple[0], c_car[0]])
candidates_max = torch.stack([c_banana[1], c_spinach[1], c_apple[1], c_car[1]])
candidates = (candidates_min, candidates_max)

# Query Logic Concepts
q_fruit = make_box([1.0, 0.0, 0.0, 0.0]) # Matches Fruit dim
q_veg   = make_box([0.0, 1.0, 0.0, 0.0]) # Matches Veg dim
q_red   = make_box([0.0, 0.0, 1.0, 0.0]) # Matches Red dim

# --- EXECUTION ---
model = BoxGate()
scores = model(candidates, q_fruit, q_veg, q_red)

# --- RESULTS ---
names = ["Banana", "Spinach", "Apple (Red)", "Car"]

for i, score in enumerate(scores):
    status = "PASS" if score > 0.001 else "REJECT"
    print(f"{names[i]:<12} | Score: {score:.4f} | {status}")
    
    if i == 2: # Apple
        if score < 0.001: print("   >> SUCCESS: Logic 'NOT Red' filtered the Apple.")
        else: print("   >> FAIL: Apple leaked through.")