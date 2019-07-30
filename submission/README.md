## Train

*Please run train.py to train your model.*

If you want to train this nerual network, you can find the train parametes part from line 10 to line 21.

* *load_ckpt* : 

	means you want to load the saved model before which makes you can stop the train process and continued with last train outcome. This will record the best MSE score on val set, not the latest model you have trained.

* *train_ epoch_num* :

	For how many epoch you want to train.

* *lr* :

	learning_rate

* *base_dir* :

	You need to put all your train set and val set in the same dir.

* *ckpt_dir* :

	This dir contains the model you have trained. saved_model is the best model for val MSE scroe. Final model is the model while you finish your train process.

* *train_path* :

	Modify it with your train set name.

* *test_path* :

	Modify it with your validation set name.


## Inference

*Please use inference.py to predict your data set. Run calculate_CDF.py will help you to see your model's CDF*

If you want to run inference, just modify the test_path will be okay. 

You can change ckpt_path to use your own trained model.

*However, this just surpport the validation set. If you want to use as a test set. You can modify the code by deleting the ground truth part.*

## Tips

With the whole trainset. Maybe you cannot get the result as our trained model. We do some data processing before we trained.

