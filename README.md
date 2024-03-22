# TicTacToeAI
## Overview
This project implements a game AI for playing Tic-Tac-Toe using reinforcement learning. The reinforcement learning methods employed include training through matches between two AIs and RLHF (Reinforcement Learning with Human Feedback), where learning occurs through interactions with humans.
## Scripts
- trains/train.py: A script for training through matches between two AIs.
- trains/rlhf_aisennkou.py: A script for enhancing the AI through matches against humans, with the AI playing first.
- trains/rlhf_aikoukou.py: A script for enhancing the AI through matches against humans, with the AI playing second.
- battle_aisennkou.py: A script for playing matches against the AI, with the AI playing first.
- battle_aikoukou.py: A script for playing matches against the AI, with the AI playing second.
## How to Use
1. First, move to the trains directory and execute the train.py script.
```bash
python train.py
```
2. After training is complete, you can play against the trained AI by executing the battle_aisennkou.py and battle_aikoukou.py scripts.
```bash
python battle_aisennkou.py
```
```bash
python battle_aikoukou.py
```
## References
- https://arxiv.org/abs/1312.5602