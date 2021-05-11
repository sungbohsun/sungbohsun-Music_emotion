for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=0, python train_mix.py --mode 4Q \
    --path1 ./model/ALL-Audio_4Q_Cnn6_fold-${i}/best_net.pt \
    --path2 ./model/ALL-Lyrics_4Q_BERT_fold-${i}/best_net.pt
done

# for i in  $(seq 0 4);
# do
#     #echo $i;
#     CUDA_VISIBLE_DEVICES=1, python train_mix.py --mode 4Q \
#     --path1 ./model/PME-Audio_4Q_Cnn10_fold-${i}/best_net.pt \
#     --path2 ./model/PME-Lyrics_4Q_ALBERT_fold-${i}/best_net.pt
# done

# for i in  $(seq 0 4);
# do
#     #echo $i;
#     CUDA_VISIBLE_DEVICES=0, python train_mix.py --mode 4Q \
#     --path1 ./model/ALL-Audio_4Q_Cnn6_fold-${i}/best_net.pt \
#     --path2 ./model/ALL-Lyrics_4Q_ALBERT_fold-${i}/best_net.pt
# done

for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=1, python train_mix.py --mode 4Q \
    --path1 ./model/ALL-Audio_4Q_Cnn10_fold-${i}/best_net.pt \
    --path2 ./model/ALL-Lyrics_4Q_ALBERT_fold-${i}/best_net.pt
done

# CUDA_VISIBLE_DEVICES=0, python train_mix.py --mode 4Q \
# --path1 ./model/ALL-Audio_4Q_Cnn6_fold-0/best_net.pt \
# --path2 ./model/ALL-Lyrics_4Q_BERT_fold-0/best_net.pt