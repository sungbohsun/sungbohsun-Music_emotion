CUDA_VISIBLE_DEVICES=1, python train_bert.py \
--model ALBERT \
--mode Ar \
--fold 0 \
--CV ALL

CUDA_VISIBLE_DEVICES=1, python train_cnn.py \
--model Cnn10 \
--mode Ar \
--fold 0 \
--CV ALL

CUDA_VISIBLE_DEVICES=0, python train_mix.py \
--mode Ar \
--path1 ./model/ALL-Audio_Ar_Cnn10_fold-0/best_net.pt \
--path2 ./model/ALL-Lyrics_Ar_BERT_fold-0/best_net.pt

CUDA_VISIBLE_DEVICES=0, python train_mix.py \
--mode Ar \
--path1 ./model/ALL-Audio_Ar_Cnn10_fold-0/best_net.pt \
--path2 ./model/ALL-Lyrics_Ar_ALBERT_fold-0/best_net.pt