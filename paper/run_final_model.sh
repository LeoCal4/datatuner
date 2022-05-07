DATASET=webnlg
TRAINING_DATA_FOLDER="../data/${DATASET}_consistency"
GENERATED_DATA_FOLDER="./eval_results/${DATASET}_$1_results"
MODEL_FOLDER="${HOME}/datatuner/trained_classifiers/${DATASET}"
# ===== VIGGO / E2E ======
#DATA_KEY=new_mr
#TEXT_KEY=ref
# ======== WEBNLG ========
DATA_KEY=modifiedtripleset
TEXT_KEY=text
DATE=$(date +%s)
LOG_OUTPUT_PATH="./outputs/final_model_${DATASET}_$1_${DATE}"

echo "Reading data from $TRAINING_DATA_FOLDER and $GENERATED_DATA_FOLDER, with model $MODEL_FOLDER"
echo "Saving log output at $LOG_OUTPUT_PATH"

bash eval_with_classifier.sh $TRAINING_DATA_FOLDER $GENERATED_DATA_FOLDER $MODEL_FOLDER $DATA_KEY $TEXT_KEY &> $LOG_OUTPUT_PATH &
tail -f $LOG_OUTPUT_PATH
