# <u>Hit Maker- The data driven guide to craft songs</u>
### Team 6, Lab group A139
## About
While composing songs, artists need to focus on several parameters like- lyrics, tempo, duration,danceability, etc. to name a few. Often, the artists are confused about which feature of the song they should focus more on. Additionally, despite the hardwork, many songs fail to gain sufficient popularity among the audience.Keeping this problem in mind, the goal of our SC1015 Mini-project is to find out which parameters of a song influence its popularity more- its lyrics or its other statistical features like tempo, danceability, etc. Analysing this influence using data science can assist artists in focusing more on relevant features and thus, in releasing more chartbuster songs.

To achieve this goal, we followed the following steps-:

1. Extract suitable datasets- one for lyrics and other for features like tempo,key,etc
2. Clean the datasets
3. Use suitable models for predicting popularity index of song
4. Compare the accuracy of prediction done using lyrics and prediction done using the other statistical features to decide which gives a better prediction    and thus, influences popularity more.

## Presentation
Presentation Video-: [Click Here](https://youtu.be/8SpMJKR04Vw)

Presentation Slides-: [Click Here](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/SC1015%20Mini_Project%20Slides.pdf)

For detailed walkthrough, please view the source code in order from:

  1. [Data Extraction and Cleaning](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/Data%20Extraction%20%26%20Cleaning-4.ipynb)
  2. [Data Visualisation and EDA](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/Exploratory%20Data%20Analysis-3-4%20(1).ipynb)
  3. [Train-Test Split](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/Train-Test-Split-2%20(1).ipynb)
  4. [Sequential Neural Network](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/Neural%20Networks.ipynb)
  5. [Random Forest](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/randomForest.ipynb)
  6. [Linear Regression (For Statistical Data)](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/Regression-2%20(1).ipynb)
  7. [LSTM Network](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/NLP-LSTM.ipynb)
  8. [Linear Regression (For Lyrics Data)](https://github.com/MoBanerjee/SC1015_DSAI_MiniProject_Team6/blob/main/NLP-baseline.ipynb)
  
## Requirements
There were a lot of open-source python libraries which were used for this project. They can all be installed on the virtual environment using: 
```
pip install -r requirements.txt
```

## Problem Definition
What predicts Song Popularity better - Lyrics or Other Statistical Features like tempo, loudness, etc?

Features Analysed-:
 1. Numerical: tempo, danceability, loudness etc.
 2. Categorical: explicit (true/false)
 3. Natural Language - lyrics

## Datasets Used
We extracted these two datasets from Kaggle.
  1. Dataset for statistical features like tempo, key, etc-: [Click Here](https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets/code?select=tracks.csv)
  2. Dataset for lyrics data-: [Click Here](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv)
  
  The dataset used was combined from these 2 datasets. In order to maintain uniformity, we merged the two datasets and found the common songs between the 2 datasets and only used those. 
 
## Data Extraction and Cleaning: 
These were the steps taken to perform data extraction and cleaning: 
  1. Merged the 2 datasets based on the common songs. All songs which were not common were dropped. 
  2. Dropped all duplicate values
  3. Dropped all null values
  4. Performed min-max normalisation. 

## Models Used
Models used for predicting popularity using features like tempo, key, loudness, etc :
  1. Sequential Neural Network
  2. Random Forest
  3. Linear Regression
  
Models used for predicting popularity using lyrics :
  1. Long Short-Term Memory (LSTM) Network
  2. Linear Regression
  
## File structure: 
The file structure of repository is as follows: 
  1. Root Directory: 
    Contains all the notebooks
  2. Final Datasets: 
    a. tracks.csv - The non-edited dataset downloaded from Kaggle. Contains the statistical data such as danceability, explicit, etc. 
    b. lyrics.csv - The non-edited dataset downloaded from Kaggle. Contains the lyrics to all the songs used. 
    c. dataCleaned.csv - The merged and cleaned file. This file is created in 'Data Extraction and Cleaning.ipynb'. 
  3. Train Test Data: 
    This file contains the train and test data for each model. We split the data in a seperate notebook called 'Train Test Split.ipynb'. This was to ensure that the same data is fed to each model for better accuracy. 
    It has the files: 
      a. train.csv - Training data
      b. test.csv - Test data

## Conclusion
  1. For statistical data, we obtained the best results with neural networks.It yielded a Mean Squared Error of only 0.026 which is lesser than that given      by both the baseline models(Random Forest and Linear Regression).
  
  2. For this data, Linear Regression performed better than Random Forest Model as the former gave a lower MSE of 0.027 while the latter had an MSE of 0.16. This could be due to overfitting in the random forest model. Overfitting happens when a model is too complex and fits noise instead of patterns in training data, leading to poor generalization on new data. Random forest models tend to have more flexibility than linear regression models, which makes them more susceptible to overfitting.

  3. For lyrics data, we obtained the best results with LSTM.It yielded a Mean Squared Error of only 0.045 which is lesser than that given by the baseline      model(Linear Regression) used.
  
  4. Since the mean squared error of statistical data(0.026) is lesser than that of lyrics data(0.045), hence we conclude that the statistical features          like liveness, tempo, danceability, etc are better predictors of the popularity of a song than its lyrics. Thus, our inference would be that artists        should focus more on these statistical features than on lyrics in order to compose more chartbuster songs.
 
 ## Future Improvements
Our current model has some limitations which can be improved in the future. Some of them are-:

* Limited input features: Although we have included several features, there may be some additional song features that could impact song popularity but are not included in our current model. We can try finding and adding more relevant features to our datasets to improve prediction accuracy.

* Lack of temporal data: Song popularity is not a static feature and can change over time. We can consider incorporating temporal data, such as streaming counts, charts positions, and reviews, to improve our model's accuracy.

* Fine-tune pre-trained models: We can try fine-tuning pre-trained models, such as GPT-3, or ResNet, on our dataset to see if they can improve our model's accuracy.

* Ensemble learning: We can combine multiple models into an ensemble to improve our model's accuracy instead of depending on a single model.

* Incorporate domain knowledge: We can try incorporating domain knowledge, such as music theory, industry trends, or expert opinions, into our model to improve its accuracy.
 
  
## Takeaways
* Data Pre-processing (Text Processing for Natural Language Processing)
  1. Tokenization
  2. Stemming
  3. Stopwords Removal
* Machine Learning
  1. Implementing Sequential Neural Network
  2. Implementing LSTM Network

## References
1. https://www.kaggle.com/datasets/lehaknarnauli/spotify-datasets/code?select=tracks.csv
2. https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv
3. https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
4. https://www.activestate.com/resources/quick-reads/how-to-create-a-neural-network-in-python-with-and-without-keras/
5. https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

## Contributors
  1. Ananya Agarwal - Data extraction, Data cleaning and EDA 
  2. Mohor Banerjee - Models for Statistical Data
  3. Shrivardhan Goenka - Models for Lyrics Data















