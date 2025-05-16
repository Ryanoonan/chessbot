# ChessNet C++ Agent

This project demonstrates how to load and use a PyTorch ChessNet model in C++ for chess position evaluation and move selection.

## Prerequisites

- LibTorch (C++ PyTorch library)
- CMake 3.0 or higher
- C++17 compatible compiler

## Setup Steps

### 1. Export your PyTorch model to TorchScript

First, export your trained ChessNet model to TorchScript format:

```bash
cd /Users/ryannoonan/Desktop/Cornell\ Classes/Spring\ 2025/CS-4701/chessbot/ryan/c++\ test/
python export_model.py
```

This will create a file called `chessnet_model.pt` that can be loaded in C++.

### 2. Build with CMake

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```

### 3. Alternative: Build with Makefile

If you prefer using the Makefile:

```bash
# Edit the TORCH_DIR in Makefile if necessary
make
```

## Running the Chess Agent

```bash
./chess_agent --model chessnet_model.pt --depth 3
```

Options:
- `--model`: Path to the exported model (default: "chessnet_model.pt")
- `--depth`: Search depth for minimax algorithm (default: 3)

## Implementation Details

- `chess_agent.cpp`: Main file containing the chess board representation and search algorithm
- `chessnet_evaluator.h`: Header-only implementation of the neural network evaluator
- `export_model.py`: Python script to export the trained model to TorchScript format

## Board Representation

The chess position is represented using bitboards - a collection of 12 64-bit integers:
- Bitboards 0-5: White pawns, knights, bishops, rooks, queens, kings
- Bitboards 6-11: Black pawns, knights, bishops, rooks, queens, kings

## Neural Network Input Format

The neural network expects input in the format similar to the `board_to_tensor_nnue` function in Python:
- A 4D tensor of shape [64, 64, 10, 2]
- The first dimension represents king positions
- The second dimension represents piece positions
- The third dimension represents piece types (5 white pieces + 5 black pieces)
- The fourth dimension represents perspective (white king vs black king)
