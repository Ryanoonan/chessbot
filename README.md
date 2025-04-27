# chessbot
CS4701 Project

How to use: 
Install stockfish '''brew install stockfish'''

Create a file at the top of the repo named .env, and put your path in it
STOCKFISH_PATH=""

Run python train_chessnet_stockfish.py
Pip install all the things you need to install (Sorry i didnt make a requirements.txt)

Then run python test_chessnet.py --model {model name here} 
By default it will run MODEL_PATH in test_chessnet.py
