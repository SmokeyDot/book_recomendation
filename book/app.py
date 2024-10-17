from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and data files
model = pickle.load(open('model.pkl', 'rb'))
book_names = pickle.load(open('book_names.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('book_pivot.pkl', 'rb'))

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])

    for name in book_name[0]: 
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)
    
    for i in range(len(suggestion)):
        books = book_pivot.index[suggestion[i]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

@app.route('/')
def home():
    return render_template('index.html', book_names=book_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    book_name = request.form['book_name']
    recommended_books, poster_url = recommend_book(book_name)
    return render_template('recommend.html', book_name=book_name, recommended_books=recommended_books, poster_url=poster_url, zip=zip)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').lower()
    matched_books = [book for book in book_names if query in book.lower()]
    return jsonify(matched_books)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
