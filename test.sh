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


CUDA_VISIBLE_DEVICES=0, python test_bert.py --path ./model/ALL-Lyrics_4Q_ALBERT_fold-0/best_net.pt