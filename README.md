# Cross-lingual-plagiarism-detector
A web-based plagiarism checker built with BERT. Allows user to compare between texts and also works like a basic search engine which provides hypertext links through similarity from web.

# Features

* Text comparison : compares two texts using BERT for similarity to find plagiarism.
* Web checker : By web scraping, works like a simple search engine to find web links for search.
* Simple API : Uses web API for web access for wide search

# Requirements

Ensure the requirements provided in "requirements.txt" to be installed before running this.

# Working

1. The app uses **Sentence-BERT** and **BERT** models to generate embeddings for the text.
2. It compares sentence similarity using **cosine similarity** between the embeddings.
3. For web checker, it uses the **Google Custom Search API** to find matching content online.
4. Through web scraping it compares the text in web and input texts to find similarity and generate links directly to access.
   


* The Google API has a daily limit for free usage.
* Sentence-BERT makes the comparison more accurate than keyword-based methods.
* Works best for text comparison, research checks, and writing originality testing.

---

Would you like me to make it slightly more polished (like GitHub formatting, bold keywords, etc.) while keeping it simple?
