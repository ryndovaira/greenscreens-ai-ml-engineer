# Greenscreens.ai homework
You should predict rate per mile using your model, features etc.

Example code returns average rate as prediction, it gives 34.85% accuracy.

Rate quality prediction is measured by MAPE (mean absolute percentage error):


$$ MAPE =  \frac{1}{N}  \sum_{i=1}^{N} {\bigg | {1 - \frac{Rate_{predicted}^{i}}{Rate_{real}^{i}}}\bigg | }  \times 100\%  $$

There are three files with data: train.csv, validation.csv (validation set infront of train in term of date) and test.csv (for your prediction)
We have divided all US territory into KMA (Key Market Regions). These regions grouped by similar market conditions that are inside each market. Try to enhance the current Rate Engine by pushing knowledge about origin and destination KMA into model. 
The dataset contains the following features: the number of miles of the route, the type of the transport (there are three main types of transport), used for transporting the cargo, the weight of the cargo, the date when the cargo was picked up, the KMA origin point and the KMA destination point.

Try to beat our prediction accuracy (MAPE), we are expecting less than 9%.

Please send estimate your validation MAPE, the predictions for the test.csv and don't forget to attach your code.
