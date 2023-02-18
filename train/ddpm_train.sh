#!/bash/bin

python3 train.py  --dataset_path /auto/data2/bguler/DDAN/breast_cancer/train \
                  --dataset_name ISIC \
                  --im_size 256 \
                  --tbs 4 \
                  --ebs 4 \
                  --training_steps 100 \
                  --lr 1e-4 \
                  --lr_warmup_steps 500 \
                  --save_image_steps 50 \
                  --save_model_steps 50 \
                  --train_timesteps 1000 \
                  --pretrained_model google/ddpm-ema-church-256 \
                  --gpu_id 0
