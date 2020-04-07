# run command `strategy=tpot bash automl.sh`
export INPUT_PATH=input/
export META_PATH=meta/


if [ -z "$strategy" ]
    then
        echo "strategy not provided"
fi

if [ ! -z "$pipe" ]
    then
        export PIPE=$pipe
fi


if [ $strategy == auto_sklearn ]
    then
        echo "executing automl ..."
        touch logs/auto_sklearn.log
        export LOGS_PATH=logs/auto_sklearn.log
        python -m src.automl.auto_sklearn
elif [ $strategy == tpot ]
    then
        echo "executing tpot ..."
        
        if [ -z "$gens" ]
            then
                export GENS=25
        else
            export GENS=$gens
        fi

        if [ -z "$pop_size" ]
            then
                export POP_SIZE=25
        else
            export POP_SIZE=$pop_size
        fi

        if [ ! -z "$max_time" ]
            then
                export MAX_TIME=$max_time
        fi

        if [ -z "$periodic_checkpt" ]
            then
                export PERIODIC_CHECKPT=false
        else
            export PERIODIC_CHECKPT=$periodic_checkpt
        fi

        CHECKPOINT_PATH=src/automl/tpot_checkpoints/$(date +%Y%m%d_%H%M%S)_"$PIPE"_"gen$GENS"_"pop$POP_SIZE"
        mkdir $CHECKPOINT_PATH
        export CHECKPOINT_PATH
        
        touch logs/tpot.log
        export LOGS_PATH=logs/tpot.log

        python -m src.automl.tpot
else
    echo "invalid strategy "$strategy", only automl and tpot are supported."
fi
