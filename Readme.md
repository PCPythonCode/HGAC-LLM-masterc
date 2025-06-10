

## Training

To reproduce the results of HGAC-LLM on four medium-scale datasets, please run following commands.

For **IMDB**:

```bash
python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
	--label-feats --hidden 512 --embed-size 512 --dropout 0.3 --input-drop 0. --amp --seeds 1 2 3 4 5
```

## **Dataset**

All the processed datasets we used in the paper can be downloaded at [Baidu Yun](https://pan.baidu.com/s/1qpchYQqM_nsFSajxI_JaOQ) (password:hgac). Put datasets in the folder 'data/' to run experimments.
