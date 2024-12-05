# Multilingual Wikipedia Sentiment Analysis

This project explores sentiment analysis across Wikipedia articles about various presidents in multiple languages, correlating sentiment scores with global favorability perceptions. 

Below is a detailed explanation of the files and folders included in this repository.

## 1. code

This folder contains Python scripts used for sentiment and correlation analysis.

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
