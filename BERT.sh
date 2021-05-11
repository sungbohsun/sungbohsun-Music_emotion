# CUDA_VISIBLE_DEVICES=1, python train_bert.py --model BERT --mode 4Q --fold 0 --CV ALL
# CUDA_VISIBLE_DEVICES=1, python train_bert.py --model BERT --mode 4Q --fold 1 --CV ALL
# CUDA_VISIBLE_DEVICES=1, python train_bert.py --model BERT --mode 4Q --fold 2 --CV ALL
# CUDA_VISIBLE_DEVICES=1, python train_bert.py --model BERT --mode 4Q --fold 3 --CV ALL
# CUDA_VISIBLE_DEVICES=1, python train_bert.py --model BERT --mode 4Q --fold 4 --CV ALL

# CUDA_VISIBLE_DEVICES=0, python train_bert.py --model BERT --mode Ar
# CUDA_VISIBLE_DEVICES=0, python train_bert.py --model BERT --mode Va

for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=1, python train_bert.py \
    --model BERT \
    --mode 4Q \
    --fold ${i} \
    --CV ALL

done

for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=1, python train_bert.py \
    --model ALBERT \
    --mode 4Q \
    --fold ${i} \
    --CV ALL

done