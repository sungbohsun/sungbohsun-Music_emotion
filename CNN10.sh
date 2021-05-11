for i in  $(seq 0 4);
do
    #echo $i;
    CUDA_VISIBLE_DEVICES=0, python train_cnn.py --model Cnn10 --mode Va --fold ${i} --CV PME
done