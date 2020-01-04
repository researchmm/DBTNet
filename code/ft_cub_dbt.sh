python3 ft_cub_dbt.py \
  --rec-train ../data/cub/train.rec --rec-train-idx ../data/cub/train.idx \
  --rec-val ../data/cub/val.rec --rec-val-idx ../data/cub/val.idx \
  --model resnet50 --mode hybrid \
  --lr 0.05 --lr-mode cosine --num-epochs 1000 --batch-size 48 --num-gpus 8 -j 60 --crop-ratio 0.875\
  --warmup-epochs 0 --dtype float16 \
  --use-rec --no-wd --label-smoothing --last-gamma \
  --save-dir ../model/params_cub_dbt \
  --logging-file ../model/log/cub_dbt.log 


