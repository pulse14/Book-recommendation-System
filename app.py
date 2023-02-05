from flask import Flask, render_template ,request
import pickle
import numpy as np
import pandas
from sklearn.neighbors import NearestNeighbors
model=NearestNeighbors(algorithm='brute')



popular_df=pickle.load(open('popular.pkl','rb'))
pt= pickle.load(open('book_pivot.pkl','rb'))
dist= pickle.load(open('distances.pkl','rb'))
sug= pickle.load(open('suggestions.pkl','rb'))
book= pickle.load(open('book.pkl','rb'))
app = Flask(__name__)

from scipy.sparse import csr_matrix
book_sparse=csr_matrix(pt)
model.fit(book_sparse)

@app.route('/')
def index():
    return render_template('index.html',
                           book_name= list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           ratings=list(popular_df['avg_ratings'].values),
    )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input=request.form.get('user_input')
    book_id = np.where(pt.index == user_input)[0][0]
    distances, suggestions = model.kneighbors(pt.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)
    data = []
    for i in range(5):
        item = []
        temp_df = book[book['Book-Title'] == pt.index[suggestions[0][i]]]

        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)

    return render_template('recommend.html',data=data)
if __name__ == '__main__':
    app.run(debug=True)