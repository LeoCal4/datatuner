export CUDA_LAUNCH_BLOCKING=1
DATASET=viggo-small
SYSTEM=DataTuner_No_FC
DATE=$(date +%s)
MODEL_OUTPUT_PATH="${HOME}/datatuner/trained_lms/${DATASET}_${SYSTEM}_$1_${DATE}"
LOG_OUTPUT_PATH="./outputs/lm_train_${DATASET}_$1_${DATE}.out"

if [ "$1" == "" ]; then
    echo "Model name missing"
    return
fi

if [ "$2" == "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES="1,2"
fi


echo "Starting training for dataset $DATASET with GPUs $CUDA_VISIBLE_DEVICES"
echo "Saving model at             : $MODEL_OUTPUT_PATH"
echo "Saving execution log file at: $LOG_OUTPUT_PATH"


if [ "$2" == "1" ]; then
    echo "Running single GPU version"
    bash train_lm_single_gpu.sh $DATASET $SYSTEM $MODEL_OUTPUT_PATH &> $LOG_OUTPUT_PATH &
else
    echo "Running multi GPU version"
    bash train_lm.sh $DATASET $SYSTEM $MODEL_OUTPUT_PATH 2 &> $LOG_OUTPUT_PATH &
fi

tail -f $LOG_OUTPUT_PATH
