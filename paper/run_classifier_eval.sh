DATASET=viggo
TRAINING_DATA_FOLDER=../data/${DATASET}_consistency/
OUTPUT_FOLDER=${HOME}/datatuner/trained_classifiers/${dataset}
TRAINING_ARGS=./classifier_training_args/$dataset/${dataset}_model_eval_args.json
DATE=$(date +%s)
LOG_OUTPUT_PATH=./outputs/classifier_eval_${DATASET}_${DATE}.out

export CUDA_VISIBLE_DEVICES="1,2"

echo "Starting classifier evaluation, reading data from $TRAINING_DATA_FOLDER and using the model at $OUTPUT_FOLDER"
echo "Saving output log at $LOG_OUTPUT_PATH"

bash train_classifier.sh $TRAINING_DATA_FOLDER $OUTPUT_FOLDER $TRAINING_ARGS 2 &> $LOG_OUTPUT_PATH &
tail -f $LOG_OUTPUT_PATH
