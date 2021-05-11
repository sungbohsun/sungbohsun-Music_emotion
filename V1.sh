# CUDA_VISIBLE_DEVICES=1, python train_cnn.py \
# --model Cnn10 \
# --mode Va \
# --fold 0 \
# --CV PME

# CUDA_VISIBLE_DEVICES=1, python train_bert.py \
# --model ALBERT \
# --mode Va \
# --fold 0 \
# --CV PME

# CUDA_VISIBLE_DEVICES=1, python train_mix.py \
# --mode Va \
# --path1 ./model/PME-Audio_Va_Cnn10_fold-0/best_net.pt \
# --path2 ./model/PME-Lyrics_Va_ALBERT_fold-0/best_net.pt

CUDA_VISIBLE_DEVICES=1, python train_mix.py \
--mode Va \
--path1 ./model/PME-Audio_Va_Cnn10_fold-0/best_net.pt \
--path2 ./model/PME-Lyrics_Va_BERT_fold-0/best_net.pt