#!/bash/bin

python3 train.py  --dataset_path /auto/data2/bguler/DDAN/breast_cancer/train \
                  --dataset_name breast_cancer \
                  --im_size 256 \
                  --tbs 8 \
                  --ebs 16 \
                  --training_steps 10000 \
                  --lr 1e-4 \
                  --lr_warmup_steps 500 \
                  --save_image_steps 1000 \
                  --save_model_steps 1000 \
                  --train_timesteps 1000 \
                  --gpu_id 0
