import wikipediaapi
from transformers import pipeline, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from functools import partial
import logging
from sklearn.metrics.pairwise import cosine_similarity

"""
Libraries Used:
- wikipediaapi: Retrieve Wikipedia articles
- transformers: Sentiment analysis with pre-trained models
- pandas: Data management and CSV handling
- numpy: Statistical calculations
- torch: Machine learning backend
- concurrent.futures: Parallel processing

Key Features:
- Supports multiple languages
- Uses GPU or Apple Silicon if available
- Parallel text processing
- Robust error handling
"""

class WikiAnalyzer:
    def __init__(self, user_agent: str):
        """
        Setup the Wikipedia sentiment analyzer:
        - Initialize Wikipedia API connection
        - Select best available computational device (GPU, Apple Silicon, or CPU)
        - Configure sentiment analysis model
        - Set up logging
        """
        self.wiki = wikipediaapi.Wikipedia(user_agent, 'en')
        
        # Choose the best available computational device
        if torch.cuda.is_available():
            device = 0  # Use first GPU
        elif torch.backends.mps.is_available():
            device = "mps"  # Use Apple M1/M2 GPU
        else:
            device = -1  # Fall back to CPU

        # BERT:
        # model="nlptown/bert-base-multilingual-uncased-sentiment"
        # tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"

        # XLM-RoBERTa:
        # model=AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
        # tokenizer=AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")
            
        # Configure sentiment analysis model
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device
        )
        
        # List of supported language codes
        self.languages = ['en', 'pl', 'it', 'hu', 'de', 'nl', 'el', 'sv', 'ko', 'th', 'tl', 'ja', 'si', 'bn', 'hi', 'ms', 'he', 'tr']
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the computational device in use
        device_name = "GPU" if device == 0 else "Apple Silicon" if device == "mps" else "CPU"
        self.logger.info(f"Using device: {device_name}")

    def fetch_article_texts(self, article_name: str) -> Dict[str, Optional[str]]:
        """
        Retrieve Wikipedia article content in multiple languages:
        - Starts with English article
        - Attempts to find translations for other supported languages
        - Returns dictionary of language codes and their article texts
        
        Handles potential errors:
        - Logs if an article is not found
        - Returns None for unavailable translations
        """
        texts = {}
        
        # Retrieve English article
        page = self.wiki.page(article_name)
        if not page.exists():
            self.logger.error(f"en article '{article_name}' not found")
            texts['en'] = None
        else:
            texts['en'] = page.text
            self.logger.info(f"Found en article: {article_name}")
        
        # Retrieve translations for other languages
        for lang in self.languages[1:]:  # Skip English
            try:
                lang_page = page.langlinks.get(lang)
                if lang_page:
                    texts[lang] = lang_page.text
                    self.logger.info(f"Found {lang} article: {lang_page.title}")
                else:
                    self.logger.warning(f"No {lang} translation available")
                    texts[lang] = None
            except Exception as e:
                self.logger.error(f"Error fetching {lang} article: {str(e)}")
                texts[lang] = None
                
        return texts

    def analyze_chunk(self, chunk: str) -> Optional[float]:
        """
        Process a text segment for sentiment:
        - Runs sentiment analysis on a single chunk of text
        - Returns sentiment score
        - Handles empty or problematic chunks gracefully
        """
        try:
            if not chunk.strip():
                return None
            return self.analyzer(chunk)[0]['score']
        except Exception as e:
            self.logger.error(f"Error analyzing chunk: {str(e)}")
            return None

    def analyze_sentiment(self, text: str, chunk_size: int = 512) -> Optional[Dict[str, float]]:
        """
        Perform comprehensive sentiment analysis:
        - Splits text into smaller chunks
        - Processes chunks in parallel
        - Calculates statistical sentiment measures:
          * Average sentiment score
          * Median sentiment score
          * Standard deviation of scores
        
        Efficiently handles texts of varying lengths
        """
        if not text:
            return None
            
        # Split text into manageable chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Process chunks concurrently
        with ThreadPoolExecutor() as executor:
            scores = list(filter(None, executor.map(self.analyze_chunk, chunks)))
            
        if not scores:
            return None
            
        return {
            'average_score': float(np.mean(scores)),
            'median_score': float(np.median(scores)),
            'standard_deviation': float(np.std(scores))
        }

    def create_report(self, article_name: str, results: Dict[str, Optional[Dict[str, float]]], output_file: str = 'sentiment_statistics_bert.csv') -> pd.DataFrame:
        """
        Generate comprehensive sentiment analysis report:
        - Creates structured data from analysis results
        - Handles both successful and failed analyses
        - Manages CSV file operations:
          * Creates new file if needed
          * Updates existing file
          * Prevents duplicate entries
        
        Ensures clean and consistent data representation
        """
        report_data = []
        
        for language, data in results.items():
            if data:
                report_data.append({
                    'Article_Title': article_name,
                    'Language': language,
                    'Mean_Sentiment_Score': f"{data['average_score']:.3f}",
                    'Median_Sentiment_Score': f"{data['median_score']:.3f}",
                    'Standard_Deviation': f"{data['standard_deviation']:.3f}"
                })
            else:
                report_data.append({
                    'Article_Title': article_name,
                    'Language': language,
                    'Mean_Sentiment_Score': 'NA',
                    'Median_Sentiment_Score': 'NA',
                    'Standard_Deviation': 'NA'
                })
                
        stats_df = pd.DataFrame(report_data)
        
        try:
            existing_df = pd.read_csv(output_file)
            # Remove any existing entries for this article
            existing_df = existing_df[existing_df.Article_Title != article_name]
            updated_df = pd.concat([existing_df, stats_df], ignore_index=True)
        except FileNotFoundError:
            updated_df = stats_df
            
        updated_df.to_csv(output_file, index=False)
        return updated_df

    def analyze_article(self, article_name: str, chunk_size: int = 512) -> pd.DataFrame:
        """
        Main method to analyze a Wikipedia article:
        - Retrieves article texts in multiple languages
        - Performs sentiment analysis
        - Generates comprehensive report
        
        Single entry point for full multilingual sentiment analysis
        """
        self.logger.info(f"Starting analysis of article: {article_name}")
        
        # Fetch article texts in different languages
        texts = self.fetch_article_texts(article_name)
        
        # Analyze sentiments for each language
        results = {}
        for lang, text in texts.items():
            self.logger.info(f"Analyzing sentiment for language: {lang}")
            results[lang] = self.analyze_sentiment(text, chunk_size) if text else None
            
        # Create and save report
        report = self.create_report(article_name, results)
        self.logger.info("Analysis completed successfully")
        return report

# Usage example
if __name__ == "__main__":
    analyzer = WikiAnalyzer('WikiSentimentAnalysis/1.0 (username@email.com)')
    
    presidents = ['Joe Biden', 'Donald Trump', 'Barack Obama', 'George W. Bush', 'Bill Clinton', 'George H. W. Bush', 'Ronald Reagan', 
                  'Jimmy Carter', 'Gerald Ford', 'Richard Nixon', 'Lyndon B. Johnson', 'John F. Kennedy', 'Dwight D. Eisenhower', 'Harry S. Truman', 
                  'Franklin D. Roosevelt', 'Herbert Hoover', 'Calvin Coolidge', 'Warren G. Harding', 'Woodrow Wilson', 'William Howard Taft', 
                  'Theodore Roosevelt', 'William McKinley', 'Grover Cleveland', 'Benjamin Harrison', 'Chester A. Arthur', 'James A. Garfield', 
                  'Rutherford B. Hayes', 'Ulysses S. Grant', 'Andrew Johnson', 'Abraham Lincoln', 'James Buchanan', 'Franklin Pierce', 
                  'Millard Fillmore', 'Zachary Taylor', 'James K. Polk', 'John Tyler', 'William Henry Harrison', 'Martin Van Buren', 'Andrew Jackson', 
                  'John Quincy Adams', 'James Monroe', 'James Madison', 'Thomas Jefferson', 'John Adams', 'George Washington']
    for president in presidents:
        report = analyzer.analyze_article(president)
    
    print(report)
