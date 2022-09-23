# medicalappointment

This is the github repo corresponding to paper:https://www.nature.com/articles/s41746-022-00594-w

Step to set up the experiments

1. install python 3+

2. pip install -r requirement.txt

3. Prepare your data (due to privacy constraint data used in this publication is not provided)

5. Set experiments you like to run in RunExperiment.py  , which is the master file used to train no-show prediction
models. 

6.choose your Experiment runner ,eg "RunExperiemnts__Keras_binary" for binary classificaiton. 

7.choose experimental type: eg. "data_missing_type_nomotheredu" means "data missing type" method wiht mother's educaitonal level removed to minize bias. 
(refer to the paper for details)

8.Set the number of epochs or use validaiton set for early stopping 

10. training your model 

11. check performance on TEST set. 


Notes:

1: model arhictecture( including number of layers, and number of units in each layer etc.)  need to be adjusted accoedin to data set
The rule of thumb is you don't underfit(check your train loss) or overfit(see your validation loss).

2. The reason vast majority of data types are converted into binary vectors is : artifical neural network does NOT handle different between real values very well.

Send me a message at dianbo@broadinstitue.org if you have any questions.
