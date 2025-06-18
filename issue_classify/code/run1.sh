DEVICE=1
TRAIN_TIME=10

FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
python train.py --model models/BERTOverflow --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model --sequence
# python train.py --model models/bert-base-uncased/pytorch_model.bin --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model models/xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE --local_model
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE 
# python train.py --model models/roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model --sequence
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model models/albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model   
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE 

# python train.py --model textcnn --embed word2vec --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model bilstm --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model rcnn --embed glove --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE

# FILE='./my_data/train/contact_TRAIN_Aug/contact_TRAIN_Aug.txt'
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
