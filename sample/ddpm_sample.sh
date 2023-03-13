#!/bash/bin

python3 sample.py --bs 5 \
                  --dataset_path /auto/data2/bguler/DDAN/breast_cancer \
                  --gpu_id 0 \
                  --gen_to_orig_ratio 0.01 \
                  --checkpoint_name pretrained_breast_cancer \
                  --checkpoint_step model_10000
