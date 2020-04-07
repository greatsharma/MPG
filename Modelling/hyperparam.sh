# run command `bash hyperparam.sh grid`

export INPUT_PATH=input/
export META_PATH=meta/
export MODEL_PATH=models/
export PIPE=pipe8

if [ -z "$1" ]
  then
    search_type=grid
else
    search_type=$1
fi


if [ $search_type == grid ]
    then
        echo "executing grid search ..."
        touch logs/gridsearch_$PIPE.log
        export LOGS_PATH=logs/gridsearch_$PIPE.log
        VERBOSE=1 python -m src.hyperparam_tunning.gridsearch
elif [ $search_type == random ]
    then
        echo "executing random search ..."
        touch logs/randomsearch_$PIPE.log
        export LOGS_PATH=logs/randomsearch_$PIPE.log
        VERBOSE=1 python -m src.hyperparam_tunning.randomsearch
else
    echo  "invalid search type "$search_type", only grid and random are supported."
fi
