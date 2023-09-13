# nohup bash train_vfhm.sh > train_vfhm.out &

DEVICE='0'
SEED='42'

ARCH='resnet18'

EYE_TYPE='L'

TRAIN_CSV_FILE="./train/data.csv"
TRAIN_FUNDUS_DIR="./train/fundus"
TRAIN_VF_DIR="./train/vf"

TEST_CSV_FILE="./test/data.csv"
TEST_FUNDUS_DIR="./test/fundus"
TEST_VF_DIR="./test/vf"
LAM="0.1"

LOG_DIR="./runs/vfhm"
CUDA_VISIBLE_DEVICES=$DEVICE python train_vfhm.py --seed $SEED \
  --arch $ARCH \
  --train_csv_file $TRAIN_CSV_FILE \
  --train_fundus_dir $TRAIN_FUNDUS_DIR \
  --train_vf_dir $TRAIN_VF_DIR \
  --test_csv_file $TEST_CSV_FILE \
  --test_fundus_dir $TEST_FUNDUS_DIR \
  --test_vf_dir $TEST_VF_DIR \
  --eye_type $EYE_TYPE \
  --pretrained \
  --lam $LAM \
  --log_dir $LOG_DIR
