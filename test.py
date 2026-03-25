import sys
from app import load_model

model, device = load_model("checkpoint.pt")
if model:
    print("Success")
else:
    print("Failed")
