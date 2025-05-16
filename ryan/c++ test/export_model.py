import torch
import sys
import os
from pathlib import Path

# Add parent directory to path to import ChessNet
sys.path.append(str(Path(__file__).parent.parent))
from chessnet import ChessNet

def export_model_to_torchscript(model_path, output_path):
    # Load the model
    model = ChessNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create an example input (match dimensions from board_to_tensor_nnue)
    example_input = torch.zeros(64, 64, 10, 2, dtype=torch.float32)
    
    # Trace the model with example input
    traced_script_module = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_script_module.save(output_path)
    print(f"Model successfully exported to {output_path}")

if __name__ == "__main__":
    # Path to the trained model
    model_path = "../chessnet_model.pth"  # Update this to your model's path
    
    # Path to save the TorchScript model
    output_path = "chessnet_model.pt"
    
    export_model_to_torchscript(model_path, output_path)
