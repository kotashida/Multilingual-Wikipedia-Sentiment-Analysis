import pandas as pd
import numpy as np
from scipy import stats

# Favorable view percentages for different countries/languages
favorable_views = {
    'pl': (86, 'Polish'),      # Poland
    'it': (56, 'Italian'),     # Italy
    'hu': (52, 'Hungarian'),   # Hungary
    'de': (49, 'German'),      # Germany
    'nl': (48, 'Dutch'),       # Netherlands
    'el': (48, 'Greek'),       # Greece
    'sv': (47, 'Swedish'),     # Sweden
    'ko': (77, 'Korean'),      # South Korea
    'th': (77, 'Thai'),        # Thailand
    'tl': (74, 'Filipino'),    # Philippines
    'ja': (70, 'Japanese'),    # Japan
    'si': (57, 'Sinhala'),     # Sri Lanka
    'bn': (52, 'Bengali'),     # Bangladesh
    'hi': (51, 'Hindi'),       # India
    'ms': (35, 'Malay'),       # Malaysia
    'he': (77, 'Hebrew'),      # Israel
    'tr': (18, 'Turkish'),     # Turkey
}

# Path (for BERT)
file = "sentiment_statistics_bert.csv"

# Read the CSV file
df = pd.read_csv(file)

# Remove duplicates while preserving order
unique_articles = df['Article_Title'].unique()

# Add favorability column based on language
df['Favorable_View'] = df['Language'].map(lambda x: favorable_views.get(x, (np.nan, ''))[0])

# Function to calculate correlations for each article group
def calculate_correlations(group):
    # Prepare the data, removing any infinite or NaN values
    clean_group = group.dropna(subset=['Mean_Sentiment_Score', 'Median_Sentiment_Score', 'Favorable_View'])
    clean_group = clean_group[
        np.isfinite(clean_group['Mean_Sentiment_Score']) & 
        np.isfinite(clean_group['Median_Sentiment_Score']) & 
        np.isfinite(clean_group['Favorable_View'])
    ]
    
    # Ensure we have enough data for meaningful correlation
    if len(clean_group) > 2:
        try:
            # Calculate Pearson correlations between favorable views and sentiment metrics
            correlations = {
                'Mean_Sentiment_Correlation': round(stats.pearsonr(clean_group['Favorable_View'], 
                                                             clean_group['Mean_Sentiment_Score'])[0], 3),
                'Median_Sentiment_Correlation': round(stats.pearsonr(clean_group['Favorable_View'], 
                                                               clean_group['Median_Sentiment_Score'])[0], 3),
                'Total_Languages': len(clean_group)
            }
            return pd.Series(correlations)
        except Exception as e:
            # Handle potential correlation calculation errors
            print(f"Error calculating correlation: {e}")
            return pd.Series({
                'Mean_Sentiment_Correlation': np.nan,
                'Median_Sentiment_Correlation': np.nan,
                'Total_Languages': len(clean_group)
            })
    else:
        # Return NaN if insufficient data
        return pd.Series({
            'Mean_Sentiment_Correlation': np.nan,
            'Median_Sentiment_Correlation': np.nan,
            'Total_Languages': len(clean_group)
        })
        
# Group by Article_Title and calculate correlations
correlation_results = df.groupby('Article_Title').apply(calculate_correlations).reset_index()

# Reorder results to match original order of unique articles
correlation_results = correlation_results.set_index('Article_Title').loc[unique_articles].reset_index()

# Print correlation results
print("Correlation between Sentiment Scores and Favorability by Article Title:")
print(correlation_results.to_string(index=False))

# To CSV file
correlation_results.to_csv("article_favorable_correlation_bert.csv", index=False)
