# for i in  $(seq 0 4);
# do
#     #echo $i;
#     CUDA_VISIBLE_DEVICES=0, python test_cnn.py --path ./model/PME-Audio_4Q_Cnn6_fold-${i}/best_net.pt
# done

# for i in  $(seq 3 4);
# do
#     #echo $i;
#     CUDA_VISIBLE_DEVICES=0, python test_cnn.py --path ./model/PME-Audio_4Q_Cnn10_fold-${i}/best_net.pt
# done

# CUDA_VISIBLE_DEVICES=2, python test_cnn.py  --path ./model/PME-Audio_Ar_Cnn6_fold-0/best_net.pt
# CUDA_VISIBLE_DEVICES=2, python test_cnn.py  --path ./model/PME-Audio_Ar_Cnn10_fold-0/best_net.pt
# CUDA_VISIBLE_DEVICES=2, python test_bert.py --path ./model/PME-Lyrics_Ar_BERT_fold-0/best_net.pt
# CUDA_VISIBLE_DEVICES=2, python test_bert.py --path ./model/PME-Lyrics_Ar_ALBERT_fold-0/best_net.pt
# CUDA_VISIBLE_DEVICES=2, python test_mix.py  --path ./model/PME-MIX_Ar_Cnn6_BERT_fold-0/best_net.pt
# CUDA_VISIBLE_DEVICES=2, python test_mix.py  --path ./model/PME-MIX_Ar_Cnn6_ALBERT_fold-0/best_net.pt
CUDA_VISIBLE_DEVICES=2, python test_mix.py  --path ./model/PME-MIX_Ar_Cnn10_BERT_fold-0/best_net.pt
CUDA_VISIBLE_DEVICES=2, python test_mix.py  --path ./model/PME-MIX_Ar_Cnn10_ALBERT_fold-0/best_net.pt