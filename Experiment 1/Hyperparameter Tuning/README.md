The three folders in these directory contain the logs of the HParams plugin used in this thesis. To view the tuning results, open a jupyter notebook and type:

%load_ext tensorboard

%tensorboard --logdir YOURFILEPATH/hparam_tuning1

%tensorboard --logdir YOURFILEPATH/hparam_tuning2

%tensorboard --logdir YOURFILEPATH/hparam_tuning3

Select the HPARAMS tab on top in the Tensorboard window to view the results.
