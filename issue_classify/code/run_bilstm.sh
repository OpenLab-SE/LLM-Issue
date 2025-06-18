DEVICE=0
TRAIN_TIME=1

# TRAIN_FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
# TEST_FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
# python train_cross.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE
# python train.py --model bert-base-uncased --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model xlnet-base-cased --embed none  --train_time $TRAIN_TIME --do_predict --device $DEVICE --sequence --file $FILE
# python train.py --model roberta-base --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE
# python train.py --model albert-base-v2 --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE


# TRAIN_FILE='./my_data/train/streamlit_new_clean_TRAIN_Aug/streamlit_new_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/streamlit_new_clean_TEST_Aug/streamlit_new_clean_TEST_Aug.txt'

TRAIN_FILE='./my_data/train/EasyOCR_newlabel_clean_TRAIN_Aug/EasyOCR_newlabel_clean_TRAIN_Aug.txt'
VALID_FILE='./my_data/valid/EasyOCR_newlabel_clean_VALID_Aug/EasyOCR_newlabel_clean_VALID_Aug.txt'
TEST_FILE='./my_data/test/EasyOCR_newlabel_clean_TEST_Aug/EasyOCR_newlabel_clean_TEST_Aug.txt'

# TRAIN_FILE='./my_data/train/faceswap_newlabel_clean_TRAIN_Aug/faceswap_newlabel_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/faceswap_newlabel_clean_TEST_Aug/faceswap_newlabel_clean_TEST_Aug.txt'

# TRAIN_FILE='./my_data/train/deepfacelab_newlabel_clean_TRAIN_Aug/deepfacelab_newlabel_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/deepfacelab_newlabel_clean_TEST_Aug/deepfacelab_newlabel_clean_TEST_Aug.txt'

# TRAIN_FILE='./my_data/train/Real-Time-Voice-Cloning_newlabel_clean_TRAIN_Aug/Real-Time-Voice-Cloning_newlabel_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/Real-Time-Voice-Cloning_newlabel_clean_TEST_Aug/Real-Time-Voice-Cloning_newlabel_clean_TEST_Aug.txt'

# TRAIN_FILE='./my_data/train/recommenders_newlabel_TRAIN_Aug/recommenders_newlabel_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/recommenders_newlabel_clean_TEST_Aug/recommenders_newlabel_clean_TEST_Aug.txt'

# TRAIN_FILE='./my_data/train/TTS_newlabel_clean_TRAIN_Aug/TTS_newlabel_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/TTS_newlabel_clean_TEST_Aug/TTS_newlabel_clean_TEST_Aug.txt'



# TRAIN_FILE='./my_data/train/streamlit_clean_TRAIN_Aug/streamlit_clean_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/streamlit_clean_TEST_Aug/streamlit_clean_TEST_Aug.txt'
# TRAIN_FILE='./my_data/train/streamlit_TRAIN_Aug/streamlit_TRAIN_Aug.txt'
# TEST_FILE='./my_data/test/streamlit_TEST_Aug/streamlit_TEST_Aug.txt'
# TEST_FILE='./my_data/train/recommenders1_TRAIN_Aug/recommenders1_TRAIN_Aug.txt'
# python train_cross.py --local_model --model models/bert-base-uncased --embed none --train_time $TRAIN_TIME --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE

python train_cross.py --local_model \
    --model bilstm \
    --trial easyocr_lr_1e-1_bs64_bilstm_newaug \
    --embed none \
    --train_time $TRAIN_TIME \
    --device $DEVICE \
    --train_file $TRAIN_FILE \
    --valid_file $VALID_FILE \
    --test_file $TEST_FILE \
    --do_predict \
    --sequence
    #  --disablefinetune

# python train_cross.py --local_model --model models/t5-large --embed none --train_time $TRAIN_TIME --device $DEVICE --train_file $TRAIN_FILE --test_file $TEST_FILE --sequence
# FILE='./my_data/train/streamlit1_TRAIN_Aug/streamlit1_TRAIN_Aug.txt'
# python train.py --model models/BERTOverflow --embed none --train_time $TRAIN_TIME --do_predict --device $DEVICE --file $FILE --local_model --sequence
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
