# chessbot
CS4701 Project

How to use: 
Install stockfish '''brew install stockfish'''

cd to one of the model folders. You must download your data, and use the generate_dataset provided functions. Then run python train_chessnet_stockfish.py (You may need to install additional requirements)

Then run python test_chessnet.py --model {model name here} 
By default it will run MODEL_PATH in test_chessnet.py

You can look at the training functions to see other optional arguments you can pass (Depth, mode ... )
