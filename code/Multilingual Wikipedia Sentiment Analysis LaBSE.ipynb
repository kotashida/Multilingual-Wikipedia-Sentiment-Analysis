from typing import Dict, Optional, List
import wikipediaapi
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import torch
from sklearn.metrics.pairwise import cosine_similarity

class WikiLaBSEAnalyzer:
    def __init__(self, user_agent: str):
        """
        Set up the Wikipedia LaBSE analyzer:
        - Initialize Wikipedia API connection
        - Select best available computational device
        - Load multilingual embedding model
        - Configure logging
        """
        self.wiki = wikipediaapi.Wikipedia(user_agent, 'en')
        
        # Choose the best available computational device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        self.device = device
        
        # Load Language-agnostic BERT Sentence Embedding (LaBSE) model
        self.model = AutoModel.from_pretrained('setu4993/LaBSE').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('setu4993/LaBSE')
        
        # Supported language codes
        self.languages = ['en', 'pl', 'it', 'hu', 'de', 'nl', 'el', 'sv', 'ko', 'th', 'tl', 'ja', 'si', 'bn', 'hi', 'ms', 'he', 'tr']
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the computational device in use
        device_name = "GPU" if device.type == "cuda" else "Apple Silicon" if device.type == "mps" else "CPU"
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

    def get_embedding(self, text: str, chunk_size: int = 512) -> Optional[np.ndarray]:
        """
        Generate embedding for a text chunk:
        - Tokenizes input text
        - Uses LaBSE model to create language-agnostic embeddings
        - Returns embedding vector or None if processing fails
        """
        try:
            if not text.strip():
                return None
                
            # Tokenize and prepare input
            inputs = self.tokenizer(text[:chunk_size], return_tensors="pt", 
                                  padding=True, truncation=True, max_length=chunk_size)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token embedding
                
            return embeddings[0]  # Return the embedding vector
            
        except Exception as e:
            self.logger.error(f"Error getting embedding: {str(e)}")
            return None

    def process_chunks(self, chunks: List[str], chunk_size: int) -> List[np.ndarray]:
        """
        Process text chunks and extract valid embeddings:
        - Iterates through text chunks
        - Generates embedding for each chunk
        - Filters out any failed embedding generations
        """
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk, chunk_size)
            if embedding is not None:  # Only append if we got a valid embedding
                embeddings.append(embedding)
        return embeddings

    def compute_similarity_metrics(self, text: str, chunk_size: int = 512) -> Optional[Dict[str, float]]:
        """
        Compute text chunk similarity metrics:
        - Splits text into manageable chunks
        - Generates embeddings for each chunk
        - Calculates pairwise cosine similarities
        - Computes statistical measures of chunk similarities
        """
        if not text:
            return None
            
        # Split text into chunks
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        # Get embeddings for all chunks
        embeddings = self.process_chunks(chunks, chunk_size)
            
        if not embeddings:
            return None
            
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle indices for unique comparisons
        upper_tri_indices = np.triu_indices_from(similarities, k=1)
        
        # If we have valid similarities to compare
        if len(similarities[upper_tri_indices]) > 0:
            return {
                'average_score': float(np.mean(similarities[upper_tri_indices])),
                'median_score': float(np.median(similarities[upper_tri_indices])),
                'standard_deviation': float(np.std(similarities[upper_tri_indices]))
            }
        else:
            # If we only have one chunk or no valid comparisons
            return {
                'average_score': 1.0,  # Perfect similarity with itself
                'median_score': 1.0,
                'standard_deviation': 0.0
            }

    def create_report(self, article_name: str, results: Dict[str, Optional[Dict[str, float]]], output_file: str = 'sentiment_statistics_labse.csv') -> pd.DataFrame:
        """
        Generate comprehensive analysis report:
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
        - Performs embedding and similarity analysis
        - Generates comprehensive report
        
        Single entry point for full multilingual embedding analysis
        """
        self.logger.info(f"Starting analysis of article: {article_name}")
        
        # Fetch article texts in different languages
        texts = self.fetch_article_texts(article_name)
        
        # Analyze text embeddings for each language
        results = {}
        for lang, text in texts.items():
            self.logger.info(f"Analyzing text for language: {lang}")
            results[lang] = self.compute_similarity_metrics(text, chunk_size) if text else None
            
        # Create and save report
        report = self.create_report(article_name, results)
        self.logger.info("Analysis completed successfully")
        return report

# Usage example
if __name__ == "__main__":
    analyzer = WikiLaBSEAnalyzer('WikiLaBSEAnalysis/1.0 (username@email.com)')
    
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
