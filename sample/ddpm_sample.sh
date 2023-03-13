#!/bash/bin

python3 sample.py --bs 16 \
                  --dataset_path /auto/data2/bguler/DDAN/breast_cancer \
                  --gpu_id 0 \
                  --gen_to_orig_ratio 1 \
                  --checkpoint_name not_pretrained_breast_cancer \
                  --checkpoint_step model_10000 \
