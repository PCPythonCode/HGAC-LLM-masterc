

## Training

To reproduce the results of HGAC-LLM on six datasets, please run following commands.

For **IMDB**:

```bash
python main.py --epoch 200 --dataset IMDB --n-fp-layers 2 --n-task-layers 4 --num-hops 4 --num-label-hops 4 \
	--label-feats --hidden 512 --embed-size 512 --dropout 0.3 --input-drop 0. --amp --seeds 1 2 3 4 5
```
**LLM setting:**
temperature=0.3,       
max_tokens=200, 
top_p=1.0,
presence_penalty=0.1,
frequency_penalty=0,


