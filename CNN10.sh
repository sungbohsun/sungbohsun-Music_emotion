# for i in  $(seq 0 4);
# do
#     #echo $i;
#     CUDA_VISIBLE_DEVICES=0, python train_cnn.py --model Cnn10 --mode Va --fold ${i} --CV PME
# done

CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn6 --mode Ar --fold 0 --CV PME
CUDA_VISIBLE_DEVICES=1, python train_cnn.py --model Cnn10 --mode Ar --fold 0 --CV PME