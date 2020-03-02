## Foundations of Data Privacy - Final project
Ella Neeman, Or honovitch

We submit all the files used by Alzantot et al. in their attack.
We only mention here new files added by us, or files from Alzantot et al. attack which we changed.
To run the code, please also use the instructions in the following link:
https://github.com/nesl/nlp_adversarial_examples
<ol>
<li>exponential mechanism.py - Used for the process of words' replacement using the Exponential Mechanism.</li>

<li>train_with_preprocess.py - Train a model with noise applied to the input layer using the Exponential Mechanism.
The parameters are set to be sensitivity, epsilon = 0.25, 0.1. To run with different parameters,
change the parameters given in "args" inside the main function. The code for the training process itself is taken
from Alzantot et al.</li>

<li>DPSentimentModel.py -  Instantiate a sentiment analysis model with an addition of a noise layer after the
embedding layer.</li>

<li>train_dp_model.py - Train a sentiment analysis classifier with an addition of a noise layer after the
embedding layer. The code for the training process itself is taken from Alzantot et al.</li>

<li>glove_utils.py - This is a file taken from the code of Alzantot et al. We changed only the function
pick_most_similar_words, which is used by the genetic attack, for efficiency reasons.</li>

<li>IMDB_attackdemo.py - Runs the attack of Alzantot et al. The code for running the attack was originally in a
Jupyter notebook. For convenience and readability, we organized the code inside a regular script file.</li>

<li>attacks.py - This is a file taken from the code of Alzantot et al. We only changed the call to the
pick_most_similar_words function.</li>
</ol>
