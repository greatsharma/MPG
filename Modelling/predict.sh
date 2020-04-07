export INPUT_PATH=input/
export META_PATH=meta/
export MODEL_PATH=models/

if [ -z "$pred_type" ]
    then
        pred_type=tpot
fi


if [ $pred_type == custom ]
    then
        MODEL=en_rf PIPE=pipe5 python -m src.predict
elif [ $pred_type == tpot ]
    then
        echo "predicting using TPOT"
        ESTIMATOR=tpot_pipe5_gen30_pop30 python -m src.automl.tpot_checkpoints.20200407_135014_pipe5_gen30_pop30.predict_tpot
else
    echo "invalid pred_type"
fi
