#!/bash/bin

python3 train.py  --bs 2 \
                  --save_dir xxxxx \
                  --dataset_path /auto/data2/bguler/DDAN/ISIC_x_ddpm/train \
                  --dataset_name ISIC \
                  --im_size 256 \
                  --tbs 2 \
                  --ebs 2 \
                  --training_steps 100000 \
                  --grad_acc_step 1 \
                  --lr 1e-4 \
                  --lr_warmup_steps 500 \
                  --save_image_steps 5 \
                  --save_model_steps 5 \
                  --train_timesteps 1000 \
                  --pretrained_model google/ddpm-ema-church-256
                  