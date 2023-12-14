# Pic-16B-Project
1. When you clone this Repository to your local computer, Please UNZIP ALL the data files in the LA_crime_predictor module before running the Project
2. Do NOT change the location of any file unzipped or generated throughout this project.
3. This Project serves as a predictor for crime. In particular, it answers two important questions: What is the probability of a crime in the user's neighborhood; What is the type of crime likely to be?
4. This Project consists of several packages: A Package for Establishing the Database, A Package for Visualization; A Package for Mathematical Modeling; A Package for Machine Learning, and A Package for Neural Network.
   
- Establishing the Database: Load Crime Data from 2010 to 2023 into a database, write SQL queries to fetch data from the database
- Mathematical Modeling: Use Exponential and Poisson Distribution to analyze the likelihood of crime instances in the neighborhood
- Machine Learning: Use Random Forest of Depth 12 to predict the type of crime(Light, Medium, or Serious) based on the user's information(Sex, Age Group, Current Location(LAT, LON))
- Neural Network: Use Neural Network(Convolution -> Maxpooling -> LSTM -> Dense) to predict the type of crime(Light, Medium, or Serious) based on the user's information(Sex, Age Group, Current Location(LAT, LON)). It serves as a comparison to the performance of the machine learning model.

5. For detailed instructions on how to use the functionalities of this project, please read "Final Report.ipynb"
6. We hope that this project keeps people aware of the potential danger surrounding them and helps everyone stay safe.
