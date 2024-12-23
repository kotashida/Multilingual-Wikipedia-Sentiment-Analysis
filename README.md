# National Favorability and Sentiment Bias: Insights from Wikipedia Articles About U.S. Presidents

This project analyzes sentiment across Wikipedia articles about various presidents in multiple languages, examining correlations between sentiment scores and global favorability perceptions.

The repository contains the following files and folders:

## 1. code

This folder contains Python scripts used for sentiment and correlation analysis.

### Multilingual Wikipedia Sentiment Analysis BERT AND XLM-RoBERTa.py
- Analyzes sentiment scores for articles in multiple languages using bert-base-multilingual-uncased-sentiment and twitter-XLM-roBERTa-base.
- Outputs: The CSV files in the sentiment_statistics folder.

### Multilingual Wikipedia Sentiment Analysis LaBSE.py
- Analyzes sentiment scores using the LaBSE model.
- Outputs: The CSV files in the sentiment_statistics folder.

### Sentiment_Favorable_Correlation.py
-   Analyzes sentiment scores for each language.
-   Outputs: The CSV files in the article_favorable_correlation folder.

### Article_Sentiment_Favorable_Correlation.py
-   Analyzes the correlation between favorability scores and sentiment scores.
-   Outputs: The CSV files in the favorable_correlation folder.

## 2. sentiment_statistics

This folder includes CSV files summarizing sentiment scores for articles in different languages.

Column Definitions:

-   Article_Title: The title of the article, corresponding to the president's name.
-   Language: Two-character ISO 639 language abbreviation.
-   Mean_Sentiment_Score: The average sentiment score of 512-character text chunks from the article.
-   Median_Sentiment_Score: The median sentiment score of the analyzed text chunks.
-   Standard_Deviation: The standard deviation of sentiment scores for the analyzed chunks.

## 3. favorable_correlation

This folder contains CSV files analyzing the correlation between favorability ratings and sentiment scores by language.

Column Definitions:

-   Language_Code: ISO 639 two-character language abbreviation.
-   Language_Name: Full name of the language.
-   Favorable_View: Percentage of people with a favorable view of the United States in the country where the language is predominantly spoken (Pew Research Center, Spring 2024 Global Attitudes Survey).
-   Mean_Sentiment: Average sentiment score for all articles in the language (range: 0–1).
-   Median_Sentiment: Median sentiment score for all articles in the language (range: 0–1).
-   Standard_Deviation: Average standard deviation of sentiment scores for articles analyzed in the language.

## 4. article_favorable_correlation

This folder contains CSV files analyzing correlations between sentiment scores and favorability ratings for individual articles.

Column Definitions:

-   Article_Title: Title of the article, corresponding to the president's name.
-   Mean_Sentiment_Correlation: Correlation coefficient (r) between favorability ratings and mean sentiment scores.
-   Median_Sentiment_Correlation: Correlation coefficient (r) between favorability ratings and median sentiment scores.
-   Total_Languages: Total number of languages analyzed for the article. While 17 languages were used, some articles are missing data for specific languages, resulting in lower counts.

***

## Sentiment Analysis Models

CSV filenames include the model used for analysis:

-   bert: bert-base-multilingual-uncased-sentiment
-   labse: LaBSE
-   xlm_roberta: twitter-XLM-roBERTa-base

Each file reflects results specific to the respective model.

***

## Results

The overall correlations between favorability ratings and mean/median sentiment scores, along with the mean standard deviation across languages, were calculated using Sentiment_Favorable_Correlation.py and are summarized below.

### Model 1: bert-base-multilingual-uncased-sentiment
- Correlations with US Favorable Views:
  - Mean_Sentiment: -0.010
  - Median_Sentiment: -0.042
- Mean Standard Deviation Across Languages: 0.058

### Model 2: LaBSE
- Correlations with US Favorable Views:
  - Mean_Sentiment: 0.008
  - Median_Sentiment: 0.020
- Mean Standard Deviation Across Languages: 0.065

### Model 3: twitter-XLM-roBERTa-base
- Correlations with US Favorable Views:
  - Mean_Sentiment: -0.393
  - Median_Sentiment: -0.386
- Mean Standard Deviation Across Languages: 0.090
