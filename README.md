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
Presentation Video-: [Click Here](https://youtu.be/3MMJGD-TlSA)

Presentation Slides-: [Click Here](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/SLIDES%20FOR%20DSAI.pdf)

For detailed walkthrough, please view the source code in order from:

  1. [Data Extraction and Cleaning](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Data%20Extraction%20%26%20Cleaning.ipynb)
  2. [Data Visualisation and EDA](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Exploratory%20Data%20Analysis%20.ipynb)
  3. [Train-Test Split](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Train-Test-Split.ipynb)
  4. [Linear Regression (For Statistical Data)](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Linear%20Regression%20for%20Statistical%20Data.ipynb)
  5. [Random Forest](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Random%20forest%20for%20Statistical%20Data.ipynb)
  6. [Sequential Neural Network](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/Sequential%20Neural%20Networks.ipynb)
  7. [Linear Regression (For Lyrics Data)](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/NLP-baseline-Linear%20Regression.ipynb)
  8. [LSTM Network](https://github.com/MoBanerjee/SC1015_DSAI-MiniProject_Team6/blob/main/NLP-LSTM.ipynb)
  
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
 
 Information about each column in the first dataset can be found below. (Note: the values stated are before Min-Max Normalization)
* ID:
  * It is the unique index assigned to each song
  * Datatype: Categorical - it is actually a string so the data type in the output specifies it as object

* Name:
  * It is the title of the song
  * Datatype: It is actually a string so the data type in the output specifies it as object

* Popularity:
  * It is the popularity score which ranges from 0 to 100, with 100 being the most popular. It is calculated based on the number of streams and how recent those streams are
  * Datatype: Categorical

* Duration_ms:
  * It is the duration of the song given in milliseconds
  * Datatype: Numerical

* Explicit:
  * It indicates whether a track contains explicit content (lyrics or themes that may be unsuitable for children or sensitive listeners). The value of this column is binary, with 1 indicating that the track contains explicit content, and 0 indicating that it does not.
  * Datatype: Categorical

* Artists:
  * It is the name of the artists of the song
  * Datatype: Categorical - it is actually a string so the data type specifies it as object

* Id_artists:
  * It is the unique id of each artist
  * Datatype: Categorical-it is actually a string so the data type specifies it as object

* Release_date:
  * It gives the song release date
  * Datatype: It is a timestamp, so it has been given as object data type in the output

* Danceability:
  * It represents a score indicating how suitable a track is for dancing based on a combination of musical elements, such as tempo, rhythm stability, beat strength, and overall regularity. The values range from 0.0 to 1.0, with higher values indicating that a track is more danceable
  * Datatype: Numerical

* Energy:
  * It is a measure of how powerful and energetic a song sounds.The energy value ranges from 0 to 1, with higher values indicating more energetic songs
  * Datatype: Numerical

* Key:
  * It is assigned based on the standard pitch class notation system, which assigns a number between 0 and 11 to represent the 12 different pitch classes in western music. In the dataset, this column contains integer values ranging from 0 to 11, where each value corresponds to a specific pitch class
  * Datatype: Numerical

* Loudness:
  * It represents the overall loudness of a track in decibels (dB). The loudness values are represented as floating-point numbers ranging from -60 dB to 0 dB
  * Datatype: Numerical

* Mode:
  * It represents the modality of a track, i.e., whether a track is in a major or minor key. It is indicated by a binary value, where 1 represents a major key and 0 represents a minor key
  * Datatype: Numerical

* Speechiness:
  * It ranges from 0.0 to 1.0, with higher values indicating that a track is more likely to be mostly spoken words, while lower values indicate that the track is likely to be more instrumental or music-focused
  * Datatype: Numerical

* Acousticness:
  * It is represented as a value between 0.0 and 1.0, where 0.0 indicates a high degree of electronic sounds and 1.0 indicates a high degree of acoustic sounds
  * Datatype: Numerical

* Instrumentalness:
  * It is represented as a value between 0.0 and 1.0, where a higher value indicates that the track is more instrumental and contains fewer vocals
  * Datatype: Numerical

* Liveness:
  * It refers to the probability that a track was performed live. The value is represented as a float between 0.0 and 1.0, where a higher value indicates that the track is more likely to have been performed live
  * Datatype: Numerical

* Valence:
  * It is a numeric indicator of the musical positiveness conveyed by a track. The values range from 0.0 to 1.0, where tracks with a higher valence value sound more positive or happy, and those with a lower valence value sound more negative or sad
  * Datatype: Numerical

* Tempo:
  * It ranges from around 40 BPM to over 200 BPM, with an average of around 120 BPM.A higher tempo generally indicates a faster-paced song, while a lower tempo suggests a slower, more relaxed song
  * Datatype: Numerical

* Time_signature:
  * It is represented as integers, where the first number in the time signature is represented by the integer value in the column. For example, a track with a time signature of 4/4 would have a value of 4 in the time signature column
  * Datatype: Numerical

Information about each column in the second dataset can be found below.
* ALink:
  * It refers to the link to the webpage where the lyrics were scraped from. It is irrelevent to our project so we do not use it in analysis
  * Datatype: It is a URL so the output specifies it as object

* SName:
  * It refers to the title of the song
  * Datatype: It is a string so the output specifies it as object

* Slink:
  * It refers to link of the song. It is irrelevent to our project so we do not use it in analysis
  * Datatype: It is a URL so it has been specified as object in the ouput

* Lyric:
  * It refers to the lyrics of the song
  * Datatype: It is a string so it has been specified as object in the output

* Language:
  * The values are two-letter codes representing the language, such as "en" for English, "es" for Spanish, "fr" for French, and so on
  * Datatype: It is actually a string so the data type in the output specifies it as object



## Exploratory Data Analysis
EDA carried out in our first dataset showed the following-:
1. We plotted scatterplots between popularity index (to be predicted) and each numerical feature variable like loudness, danceability, etc. All of them indicated very poor correlation coefficient values.This led us to the hypothesis that a non-linear relationship exists between the numerical variables and popularity index.

2. We plotted categorically divided boxplots for the categorical variables like key, mode, etc. Even they indicated very poor correlation values.

3. Same conclusion was drawn from a heatmap plotted between all the variables and popularity.

4. These poor correlations are understandable as song making is a complicated process and a single feature can never guarantee success of the song.

For the second dataset, before performing EDA, we performed some text-cleaning functions (Natural Language Pre-processing) on the lyrics data.The steps taken involve-:
1. Conversion of all alphabets to lowercase for consistency.

2. Stopwords Removal-: Stopwords are common words that are usually removed from text data during preprocessing because they do not carry much meaning.

3. Stemming-: Process of reducing a word to its base form (or stem) by removing affixes (prefixes, suffixes, etc.).

4. Tokenization-: Splitting the text string into individual words or tokens.

After all this text pre-processing, we recombined the cleaned text string and returned it. 

EDA carried out in our second dataset showed the following-:
1. We created a new column in the dataset called lenth with the length of each element x of the 'Lyric Altered' column. After creating the column, we have performed EDA by creating a box plot for the length values.

2. We also plotted the histogram plot along with the kernel density estimate of popularity index.

3. We also extracted the top 20 common words in the lyrics and plotted them in a barplot.


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
     1. tracks.csv - The non-edited dataset downloaded from Kaggle. Contains the statistical data such as danceability, explicit, etc. 
     2. lyrics.csv - The non-edited dataset downloaded from Kaggle. Contains the lyrics to all the songs used. 
     3. dataCleaned.csv - The merged and cleaned file. This file is created in 'Data Extraction and Cleaning.ipynb'.
      
  3. Train Test Data: 
    This file contains the train and test data for each model. We split the data in a seperate notebook called 'Train Test Split.ipynb'. This was to ensure that the same data is fed to each model for better accuracy. 
    It has the files: 
      1. train.csv - Training data
      2. test.csv - Test data

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















