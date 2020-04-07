# export IMG_PATH=img/
# export INPUT_PATH=input/
# export META_PATH=meta/
# export MODEL_PATH=models/

# python -m src.dataset --shuffle 1
# python -m src.pipelines
# python -m src.create_folds --shuffle 1

# for pipe in pipe1 pipe2 pipe3 pipe4 pipe5 pipe6 pipe7 pipe8 pipe9 pipe10 pipe11
#     do
#         touch logs/$pipe.log
#         export LOGS_PATH=logs/$pipe.log
#         python -m src.logger
        
#         export PIPE=$pipe
#         for model in lin_reg ridge lasso svr en_rf en_et en_adaboost en_gradboost
#             do
#                 MODEL=$model python -m src.train
#             done
#     done

# echo '\ngenerating plots...\n'
# runipy notebooks/plotting.ipynb -q