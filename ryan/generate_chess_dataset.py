import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import chess
import chess.engine
import numpy as np
from helper.board_to_tensor import board_to_tensor_1
import random
from tqdm import tqdm
import os
import pickle
from dotenv import load_dotenv
import argparse

from helper.static_exchange_eval import is_quiet

load_dotenv()

STOCKFISH_PATH = os.getenv('STOCKFISH_PATH')
if (STOCKFISH_PATH is None):
    STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/17/bin/stockfish"




class ChessDataset(Dataset):
    """Dataset of chess positions and their evaluations."""

    def filter_noisy_positions(self, max_positions):
        new_positions = []
        new_evaluations = []
        count = 0
        nb_quiet = 0
        for i, board in enumerate(self.positions):
            count += 1
            if count % 1000 == 0:
                # Print every 1000th position
                print(f"Filtering position number {count}, \n number of quiet positions: {nb_quiet} \n\n")
            if is_quiet(board):
                new_positions.append(board)
                new_evaluations.append(self.evaluations[i])
                nb_quiet += 1
            if count >= max_positions:
                break
        self.positions = new_positions
        self.evaluations = new_evaluations

    def __init__(self, num_positions = 10000, load_path=None, normalize=False, means=None, stds=None):
        self.positions = []
        self.evaluations = []
        self.normalize = normalize
        self.means = means
        self.stds = stds
        
        if load_path and os.path.exists(load_path):
            print(f"Loading dataset from {load_path}")
            with open(load_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.positions = saved_data['positions']
                self.evaluations = saved_data['evaluations']

    @classmethod
    def load_from_file(cls, file_path):
        # 1) unpickle the stored dict (or tuple) 
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        # 2) create a fresh instance
        ds = cls()
        # 3) assign loaded lists to its attributes
        #    adjust these keys if your pickle stored them under different names
        ds.positions = data['positions']
        ds.evaluations = data['evaluations']
        return ds
    
    def save_to_file(self, file_path):
        data = {
            'positions': self.positions,
            'evaluations': self.evaluations
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {file_path}")
        return 
    
    def stockfish_generate_data(self, num_positions=10000):
        """Generate chess positions and evaluations using Stockfish engine."""
        print("Generating new dataset...")
        self.engine = chess.engine.SimpleEngine.popen_uci(
            STOCKFISH_PATH
        )

        # Generate random positions
        for _ in tqdm(range(num_positions), desc="Generating positions"):
            board = self._generate_random_position()
            evaluation = self._get_stockfish_evaluation(board)
            self.positions.append(board)
            self.evaluations.append(evaluation)

        self.engine.quit()
        return len(self.positions)

    def _generate_random_position(self) -> chess.Board:
        """Generate a random chess position."""
        board = chess.Board()
        num_moves = random.randint(0, 30)  # Random number of moves

        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            board.push(move)
        
        return board

    def _get_stockfish_evaluation(self, board: chess.Board) -> float:
        """Get Stockfish evaluation for a position."""
        result = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        score = result["score"].white().score(mate_score=10000)
        return score / 100.0  # Convert to centipawns

    def load_lichess_dataset(self, dataset_path, num_positions=None):
        """
        Load chess positions and evaluations from the Lichess dataset.
        
        Parameters:
        - dataset_path: Path to the Lichess dataset CSV file
        - num_positions: Maximum number of positions to load (None for all)
        
        Format expected:
        - CSV with columns for FEN, evaluation scores (could be in 'cp' or 'mate' columns)
        - FEN represents chess positions in Forsyth-Edwards Notation
        """
        import csv
        import pandas as pd
        
        print(f"Loading Lichess dataset from {dataset_path}...")
        
        try:
            # Try to load with pandas if available (faster for large files)
            df = pd.read_csv(dataset_path)
            
            # Check for expected columns in this dataset
            expected_columns = ['fen', 'line', 'depth', 'knodes', 'cp', 'mate']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Warning: Missing expected columns: {missing_columns}")
                
            # Ensure we have at least FEN and one evaluation column
            if 'fen' not in df.columns or ('cp' not in df.columns and 'mate' not in df.columns):
                print(f"Error: Required columns missing. Available columns: {df.columns.tolist()}")
                return 0
                
            # Limit the number of positions if specified
            if num_positions is not None:
                df = df.head(num_positions)
            
            # Process each position
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading positions"):
                try:
                    fen = row['fen']
                    
                    # Determine evaluation (either centipawns or mate)
                    if pd.notna(row.get('cp')):
                        evaluation = float(row['cp']) / 100.0  # Convert centipawns to pawns
                    elif pd.notna(row.get('mate')):
                        # For mate scores, use a large value with sign indicating which side mates
                        mate_score = row['mate']
                        if pd.notna(mate_score) and mate_score != '':
                            evaluation = 100.0 if float(mate_score) > 0 else -100.0
                        else:
                            evaluation = 0.0  # Default if mate score is missing/invalid
                    else:
                        print(f"Warning: No evaluation found for position {fen}, using 0.0")
                        evaluation = 0.0
                    
                    # Create a board from the FEN
                    board = chess.Board(fen)
                    
                    self.positions.append(board)
                    self.evaluations.append(evaluation)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
                
        except (ImportError, Exception) as e:
            print(f"Error using pandas: {e}. Trying CSV reader...")
            
            # Fallback to CSV reader
            with open(dataset_path, 'r') as f:
                reader = csv.DictReader(f)
                
                # Check column names
                header = reader.fieldnames
                if 'fen' not in header or ('cp' not in header and 'mate' not in header):
                    print(f"Error: Required columns missing. Available columns: {header}")
                    return 0
                
                # Process positions
                count = 0
                for row in tqdm(reader, desc="Loading positions"):
                    if num_positions is not None and count >= num_positions:
                        break
                        
                    try:
                        fen = row['fen']
                        
                        # Determine evaluation (either centipawns or mate)
                        if row.get('cp') and row['cp'].strip():
                            evaluation = float(row['cp']) / 100.0  # Convert centipawns to pawns
                        elif row.get('mate') and row['mate'].strip():
                            # For mate scores, use a large value with sign indicating which side mates
                            mate_score = row['mate']
                            evaluation = 100.0 if float(mate_score) > 0 else -100.0
                        else:
                            print(f"Warning: No evaluation found for position {fen}, using 0.0")
                            evaluation = 0.0
                        
                        # Create a board from the FEN
                        board = chess.Board(fen)
                        
                        self.positions.append(board)
                        self.evaluations.append(evaluation)
                        count += 1
                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue
        
        print(f"Successfully loaded {len(self.positions)} positions from Lichess dataset")
        return len(self.positions)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        board = self.positions[idx]
        evaluation = self.evaluations[idx]

        # Convert board to tensor
        tensor = board_to_tensor_1(board)

        return tensor, torch.tensor([evaluation], dtype=torch.float32)
    
    def save(self, save_path):
        """Save the dataset to a file."""
        save_data = {
            'positions': self.positions,
            'evaluations': self.evaluations
        }
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Dataset saved to {save_path}")

    def compute_channel_stats(self):
        """
        Compute channel-wise mean and standard deviation across all positions in the dataset.
        Returns two 1D Tensors of length C: (means, stds) where C is the number of channels in the board-tensor.
        """
        # Import board_to_tensor
        from chessnet import board_to_tensor
        
        # Take a sample board to determine number of channels
        if not self.positions:
            print("No positions available to compute statistics")
            return None, None
            
        print("Computing channel statistics...")
        
        sample_board = self.positions[0]
        sample_tensor = board_to_tensor(sample_board)
        num_channels = sample_tensor.shape[0]
        
        # Initialize accumulators
        cnt = 0
        ch_sum = torch.zeros(num_channels)
        ch_sq_sum = torch.zeros(num_channels)
        
        # Process each board
        for board in tqdm(self.positions, desc="Computing channel statistics"):
            tensor = board_to_tensor(board)  # This returns [C, 8, 8]
            
            # Accumulate statistics
            cnt += 64  # 8x8 board
            ch_sum += tensor.sum(dim=[1, 2])  # Sum across spatial dimensions for each channel
            ch_sq_sum += (tensor * tensor).sum(dim=[1, 2])
        
        # Compute mean and std per channel
        means = ch_sum / cnt
        variances = ch_sq_sum / cnt - means**2
        # Handle numerical issues (prevent negative variance due to precision)
        variances = torch.clamp(variances, min=1e-8)
        stds = torch.sqrt(variances)
        
        return means, stds
        
    def save_with_stats(self, save_path, stats_path="channel_stats.pth"):
        """Save the dataset to a file and compute/save channel statistics."""
        # Compute channel statistics
        means, stds = self.compute_channel_stats()
        
        # Save channel statistics
        if means is not None and stds is not None:
            torch.save({"mean": means, "std": stds}, stats_path)
            print(f"Channel statistics saved to {stats_path}")
        
        # Save dataset
        self.save(save_path)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate or load chess position datasets with evaluations')
    
    # Dataset source selection
    parser.add_argument('--method', type=str, choices=['stockfish', 'lichess'], default='stockfish',
                        help='Method to use for dataset creation: stockfish (generate new) or lichess (load existing)')
    
    # Common arguments
    parser.add_argument('--output', type=str, default='chess_dataset.pkl',
                        help='Output file path to save the dataset')
    parser.add_argument('--stats-output', type=str, default='channel_stats.pth',
                        help='Output file path to save the channel statistics')
    parser.add_argument('--num-positions', type=int, default=1000,
                        help='Number of positions to generate or load')
    
    # Stockfish-specific arguments
    parser.add_argument('--stockfish-path', type=str, default=STOCKFISH_PATH,
                        help='Path to Stockfish engine executable')
    
    # Lichess-specific arguments
    parser.add_argument('--lichess-path', type=str, default='lichess_db_eval.jsonl.zst',
                        help='Path to Lichess dataset file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create dataset object
    dataset = ChessDataset()
    
    # Process based on selected method
    if args.method == 'stockfish':
        print(f"Generating {args.num_positions} positions using Stockfish...")
        # Update stockfish path if provided
        
        # Generate data
        dataset.stockfish_generate_data(num_positions=args.num_positions)
        print(f"Generated Stockfish dataset with {len(dataset)} positions")
        
    elif args.method == 'lichess':
        print(f"Loading up to {args.num_positions} positions from Lichess dataset...")
        if not os.path.exists(args.lichess_path):
            print(f"Error: Lichess dataset file not found at {args.lichess_path}")
            print("Please download the Lichess dataset and provide the correct path.")
            return
            
        # Load data from Lichess dataset
        dataset.load_lichess_dataset(args.lichess_path, num_positions=args.num_positions)
        print(f"Loaded Lichess dataset with {len(dataset)} positions")
    
    # Save the dataset with channel statistics
    dataset.save_with_stats(args.output, args.stats_output)
    print(f"Dataset saved to {args.output} and channel statistics saved to {args.stats_output}")


if __name__ == "__main__":
    main()