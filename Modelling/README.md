Run shell Command `sh train.sh` to begin training. You can add more pipelines and more estimators as you wish.<br><br>
Run shell Command `sh clear.sh` to clear all generated files from training. This is for a fresh start.<br><br>
Run shell Command `pred_type=custom/topt bash predict.sh` to predict for new data (test data) using either custom pipelines and estimators or best pipeline found by TPOT. This should be used at the end only when training is finished and you get your best pipeline and model.<br>
Run shell Command `bash hyperparam.sh grid/random` for hyper-parameter tunning using either
grid search or random search.<br>
Run shell Command `strategy=tpot bash automl.sh` to run TPOT. You can provide many other CL arguments as needed, but all are not compulsory there are defaults for them. For further you can visit `automl.sh`. There is a lot scope for changes in automl.<br>

You can change these script according to you as well.<br>

[Here](https://github.com/greatsharma/MPG/blob/master/Modelling/img/model_distmetric_by_pipes.png) is the violin plots for the `r2 scores` for various models bifurcated by pipelines. From these results I found `pipe 5` & `pipe 8` as the best pipelines and `ensemble models` are the best model.

[Here](https://github.com/greatsharma/MPG/blob/master/Modelling/img/model_validation_true_pred_plot.png) is the scatter plot for true v/s predicted values. The best performing models are `LinearRegression`, `RandomForestRegressor` and `GradientBoostingRegressor`.<br><br>

**Directory Structure**<br>

This directory structure is inspired from the [Applied Machine Learning Framework](https://www.youtube.co/watch?v=ArygUBY0QXwlist=PL98nY_tJQXZnKfgWIADbBG182nFUNIsxw) by kaggle grandmaster [abhishek thakur](https://www.kaggle.com/abhishek).
I modified the template provided by him to large extent. I highly recommend to watch his playlist.


```
img - All generated plots.
input - All given and generated csv files.
logs - All log files.
meta - All dumped pipelines.
models - All dumped models.
notebooks - All notebooks for exploration purpose.
src - Source code.
```
