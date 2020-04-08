Currently only `TPOT` is implemented. Although it is sufficient and good but for regression tasks but for classification tasks many recommends to use `auto-sklearn`. But auto-sklearn is having some dependency issues with TPOT, at the time of writing.<br>

`tpot_checkpoints` contains folders for different run. The naming is done elegantly, time_stamp appended by run's core parameters making the naming more readable, informatic and easy to search as we run more.

Each run contains three files by default. One `exported_tpot.py` the best pipeline for the run. and others contains all pipelines evaluated by TOPT during the run as dumped files.

The best run also has an extra file named `predict_tpot.py` for doing prediction using best TOPT ppeline.

I found that the best TOPT pipeline achieved an average r2 score around 88-89. Whereas my custom program achieved somewhat around 85. The reason for this is TPOT apply `genetic programming` approach for it's search but also that the resulted pipelines include stacked estimators. So if we had applied stacking in our custom program than we would also have achieved somewhat around 86-87.

This section has a lot scope for changes, and you can do so by passing the parameters while running those.