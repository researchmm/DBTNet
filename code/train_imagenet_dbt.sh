python3 train_imagenet_dbt.py \
  --rec-train ../data/imagenet/val.rec --rec-train-idx ../data/imagenet/val.idx \
  --rec-val ../data/imagenet/val.rec --rec-val-idx ../data/imagenet/val.idx \
  --model resnet50  --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --dtype float16 \
  --use-rec --no-wd --label-smoothing --last-gamma \
  --save-dir ../model/params_imagenet_dbt \
  --logging-file ../model/log/imagenet_dbt.log \
