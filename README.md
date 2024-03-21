# TicTacToeAI
## 概要
本プロジェクトは三目並べを行うゲームAIを強化学習を用いて実装するものである．強化学習にはAIとAIを対戦させながら学習させていく手法と人間とのインタラクションを通して学習していくRLHFを採用している．
## スクリプト
- trains/train.py：２つのAIが対戦しながら学習を行なっていくためのスクリプト
- battle_aisennkou.py：AIと人間が対戦するためのスクリプト．このスクリプトではAIが先攻．
- battle_aikoukou.py：AIと人間が対戦するためのスクリプト．このスクリプトではAIが後攻．
## 利用方法
1. まず，trainsディレクトリにカレントディレクトリを移動し，train.pyスクリプトを実行する．
```bash
python train.py
```
2. 訓練が終了するとbattle_aisennkou.pyとbattle_aikoukou.pyスクリプトを実行することで学習済みのAIと対戦できる．
```bash
python battle_aisennkou.py
```
```bash
python battle_aikoukou.py
```
## 参考文献
- https://arxiv.org/abs/1312.5602