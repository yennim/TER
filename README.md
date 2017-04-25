TER - Neurocomputational modeling in decision making
==


Installations needed
-
To use the file dopamine_cell_RL.ipynb, you will need to install anaconda
and then use Jupyter Notebook by writing jupyter notebook in your command.

To use the file dopamine_cell_RL_3D, you will need to install matplotlib
library by using pip install matplotlib (if you use IDLE),
or conda install matplotlib (if you use Spyder).  

Running the files
-
If you run the notebook you will get the 2D plot of dopamine cell's
activity in early training when alpha=0.05 , lambda=0 and a 3D plot
representing their activity when alpha=0.005 and lambda=0.9.  

If you run the file dopamine_cell_RL_2D, you will get the 2D plot
of dopamine cell's activity in early training when alpha=0.005, lambda=0.9,
and a graph representing their activity when the second cue is omitted.  

If you run the file dopamine_cell_RL_3D, you will get the 3D plot of 
dopamine cell's activity in late training when alpha=0.005, lambda=0.  
  
Changing parameters
-
To change the parameters, you must open the first file with jupyter
notebook or the two other files with an editor (IDLE or Spyder for example).  

- n_trials corresponds to the number of trials: n_trials=100 corresponds to
an early training, n_trials=400 for alpha=0.05. You may want n_trials=800
for a late training when alpha=0.005.  
- alpha (learning rate) is a value between 0 and 1, and so is 
lambda(eligibility trace parameter).  

You may want to change the trial into 1 stimulus, 1 reward. To do so, you
have to replace in either the 2D or 3D file : tdmodel.trial() by
tdmodel.trial(True, False).  
For the 2D model, if you don't want the comparison when omitting the second
cue, you should delete the line tdmodel.trial(True, False).