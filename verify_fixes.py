#!/usr/bin/env python3
"""
Verification script to confirm all fixes are properly implemented.
Run this before training to validate the changes.
"""

import os
import re

def check_file_content(filepath, pattern, description):
    """Check if a pattern exists in a file."""
    if not os.path.exists(filepath):
        print(f"✗ {filepath} not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description}")
        return False

print("=" * 80)
print("VERIFICATION: Training Fixes Implementation")
print("=" * 80)
print()

filepath = "main.py"
checks = [
    (r"LR\s*=\s*5e-5", "LR reduced to 5e-5"),
    (r"SAMPLES_PER_VIEW\s*=\s*1024", "SAMPLES_PER_VIEW set to 1024"),
    (r"WEIGHT_MV\s*=\s*2\.0", "WEIGHT_MV increased to 2.0"),
    (r"WEIGHT_EDGE\s*=\s*0\.02", "WEIGHT_EDGE reduced to 0.02"),
    (r"WEIGHT_SURF\s*=\s*0\.01", "WEIGHT_SURF reduced to 0.01"),
    (r"WEIGHT_MAG\s*=\s*0\.005", "WEIGHT_MAG reduced to 0.005"),
    (r"def huber_loss", "Huber loss function defined"),
    (r"torch\.abs\(delta_z\)\s*/\s*\(depth_init\s*\+", "Magnitude loss uses depth_init"),
    (r"torch\.abs\(Z_sel\s*-\s*sampled2\)", "Occlusion masking implemented"),
    (r"occl_mask.*0\.25", "Occlusion tolerance set to 25%"),
    (r"huber_loss\(dist\)", "Huber loss applied to multiview"),
    (r"Δz_mean|delta_mean", "Diagnostic metrics added"),
]

all_passed = True
for pattern, description in checks:
    if not check_file_content(filepath, pattern, description):
        all_passed = False

print()
print("=" * 80)

if all_passed:
    print("✓ ALL CHECKS PASSED")
    print("=" * 80)
    print("\nFixes properly implemented. Ready to train!")
    print("\nRun with: python main.py")
    print("\nFor evaluation: python eval.py")
else:
    print("✗ SOME CHECKS FAILED")
    print("=" * 80)
    print("\nPlease verify main.py has all the fixes applied.")
    exit(1)

# Print summary
print("\nFixes Summary:")
print("  [1] Hyperparameters tuned for better learning")
print("  [2] Huber loss for robust reprojection")
print("  [3] Fixed magnitude loss denominator")
print("  [4] Occlusion masking to prevent folded geometry")
print("  [5] Diagnostic metrics for monitoring")
print()
print("Expected behavior:")
print("  • Δz_mean will be non-zero and structured")
print("  • Δz_std will increase with training")
print("  • L_mv will decrease steadily")
print("  • Fused geometry will sharpen and improve")
print()
