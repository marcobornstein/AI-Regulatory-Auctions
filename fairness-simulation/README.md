


To run the experiment, `python draft.py`


potential things to watch out for 
- I didn't check for label balance for training dataloader (maybe the more expensive/high safe cost runs have more label balance too?)
- We need some decision criteria related to how each "model builder" picks its best model over the epoch runs(should it be train accuracy? train accuracy and train fairness? which fairness criteria to use or some aggregate measure?)
- probably can do a better hyperparameter search and/or should hyper search be a part of the safety cost budget too?