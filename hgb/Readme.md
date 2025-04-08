# SeHGNN on the four medium-scale datasets

## Training

To reproduce the results of SeHGNN on four medium-scale datasets, please run following commands.

For **IMDB**:

```bash
python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
	--label-feats --hidden 512 --embed-size 512 --dropout 0.5 --input-drop 0. --amp --seeds 1 2 3 4 5
```

