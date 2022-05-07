DATASET=webnlg
SYSTEM=DataTuner_No_FC
TEST_FILE="../data/${DATASET}/test.json"
BASE_MODEL_PATH="${HOME}/datatuner/trained_lms/${DATASET}_${SYSTEM}_$1"
DATE=$(date +%s)
LOG_OUTPUT_PATH="./outputs/lm_eval_${DATASET}_${1}_${DATE}.out"

echo "Evaluating classifier at $BASE_MODEL_PATH on $TEST_FILE"
echo "Output log saved at $LOG_OUTPUT_PATH"

bash evaluate_lm_simple.sh $TEST_FILE $BASE_MODEL_PATH &> $LOG_OUTPUT_PATH &
tail -f $LOG_OUTPUT_PATH
