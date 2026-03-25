import torch
import os

# Check model files
model_files = ["checkpoint.pt", "full_model.pt", "model_weights.pt"]

for fname in model_files:
    if os.path.exists(fname):
        size_mb = os.path.getsize(fname) / (1024 * 1024)
        print(f"\n{fname}: {size_mb:.1f} MB")

        try:
            # Try loading with safe settings
            checkpoint = torch.load(fname, map_location='cpu', weights_only=False)
            print(f"  Loaded successfully!")

            if isinstance(checkpoint, dict):
                print(f"  Keys: {list(checkpoint.keys())}")
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # Show first few keys
                keys = list(state_dict.keys())[:5]
                print(f"  State dict sample keys: {keys}")
            else:
                print(f"  Type: {type(checkpoint)}")
                if hasattr(checkpoint, 'state_dict'):
                    print(f"  Has state_dict method")

        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print(f"\n{fname}: NOT FOUND")
