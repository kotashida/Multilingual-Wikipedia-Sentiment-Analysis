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

# Analyze a filtered datset of 10 most recent presidents:
# unique_articles = df['Article_Title'].unique()[:10]
# df = df[df['Article_Title'].isin(unique_articles)]

# Analyze a filtered datset of the first 10 presidents:
# unique_articles = df['Article_Title'].unique()[-10:]
# df = df[df['Article_Title'].isin(unique_articles)]

# Calculate averages by language
language_averages = df.groupby('Language').agg({
    'Mean_Sentiment_Score': 'mean',
    'Median_Sentiment_Score': 'mean',
    'Standard_Deviation': 'mean'
}).round(3)

# Combine the data
combined_data = []
for lang_code in favorable_views.keys():
    if lang_code in language_averages.index:
        row = {
            'Language_Code': lang_code,
            'Language_Name': favorable_views[lang_code][1],
            'Favorable_View': favorable_views[lang_code][0],
            'Mean_Sentiment': language_averages.loc[lang_code, 'Mean_Sentiment_Score'],
            'Median_Sentiment': language_averages.loc[lang_code, 'Median_Sentiment_Score'],
            'Standard_Deviation': language_averages.loc[lang_code, 'Standard_Deviation']
        }
        combined_data.append(row)

result_df = pd.DataFrame(combined_data)

# Calculate correlations
correlations = {
    'Mean_Sentiment': stats.pearsonr(result_df['Favorable_View'], result_df['Mean_Sentiment'])[0],
    'Median_Sentiment': stats.pearsonr(result_df['Favorable_View'], result_df['Median_Sentiment'])[0],
}

# Format and print the results
print("\nCorrelations with US Favorable Views:")
for metric, corr in correlations.items():
    print(f"{metric}: {corr:.3f}")

print(f"\nMean Standard Deviation Across Languages: {result_df['Standard_Deviation'].mean():.3f}")

print("\nDetailed Data:")
result_df = result_df.sort_values('Favorable_View', ascending=False)
print(result_df.to_string(index=False))

# To CSV file
result_df.to_csv("favorable_correlation_bert.csv", index=False)
