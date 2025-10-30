from flask import Flask, request, render_template_string, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from googleapiclient.discovery import build

# Replace with your actual API key and Custom Search Engine ID
API_KEY = "api-key"  # Replace with your API key
CX = "cx-id"  # Replace with your Custom Search Engine ID

# Initialize models
model_sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

app = Flask(__name__)

# Functions (BERT, scrape, compare) same as before

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()

def cosine_similarity_bert(text1, text2):
    embedding1 = get_bert_embedding(text1)
    embedding2 = get_bert_embedding(text2)
    return cosine_similarity(embedding1, embedding2)[0][0]

def scrape_website(url):
    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return f"Error scraping {url}: {e}"

def search_google(query):
    try:
        service = build("customsearch", "v1", developerKey=API_KEY)
        res = service.cse().list(q=query, cx=CX).execute()
        if 'items' in res:
            return [item['link'] for item in res['items']]
        else:
            return []  # Return empty list if no results
    except Exception as e:
        print(f"Error in Google Custom Search: {e}")
        return []

def check_similarity(text):
    results = []
    sentences = text.split(".")
    for sentence in sentences:
        if len(sentence) < 5:
            continue
        urls = search_google(f'"{sentence.strip()}"')
        for url in urls:
            web_content = scrape_website(url)
            if not web_content:
                continue
            results.append(url)
    return results

def compare_texts_advanced(text1, text2):
    embedding1 = model_sbert.encode(text1, convert_to_tensor=True)
    embedding2 = model_sbert.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    sentences1 = text1.split('.')
    sentences2 = text2.split('.')
    embeddings1 = model_sbert.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model_sbert.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    highlighted_text1 = ""
    highlighted_text2 = ""
    threshold = 0.75
    for i, sentence1 in enumerate(sentences1):
        max_score = max(cosine_scores[i])
        if max_score >= threshold:
            highlighted_text1 += f'<span class="highlight-yellow">{sentence1}.</span>'
        else:
            highlighted_text1 += sentence1 + "."
    for i, sentence2 in enumerate(sentences2):
        max_score = max(cosine_scores[:, i])
        if max_score >= threshold:
            highlighted_text2 += f'<span class="highlight-blue">{sentence2}.</span>'
        else:
            highlighted_text2 += sentence2 + "."
    similarity_percentage = similarity * 100
    return highlighted_text1, highlighted_text2, similarity_percentage

# Routes
@app.route('/')
def home():
    return render_template_string(home_template)

@app.route('/web', methods=['GET', 'POST'])
def web():
    if request.method == 'POST':
        text = request.form['text']
        results = check_similarity(text)
        return render_template_string(web_template, text=text, results=results)
    return render_template_string(web_template)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        highlighted_text1, highlighted_text2, similarity = compare_texts_advanced(text1, text2)
        return render_template_string(compare_template, text1=text1, text2=text2, highlighted_text1=highlighted_text1, highlighted_text2=highlighted_text2, similarity=similarity)
    return render_template_string(compare_template)

# API endpoint
@app.route('/api/web_search', methods=['POST'])
def api_web_search():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({"error": "Text is required"}), 400
        results = check_similarity(text)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
