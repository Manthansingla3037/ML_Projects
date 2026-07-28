import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


API_KEY = st.secrets["TMDB_API_KEY"]

def fetch_poster(movie_ID):
    response=requests.get(f'https://api.themoviedb.org/3/movie/{movie_ID}?api_key={API_KEY}&language=en-US')
    data=response.json()
    return 'https://image.tmdb.org/t/p/w500/'+data['poster_path']

def recommend(movie):
    movie_index=movies_list[movies_list['title']==movie].index[0]
    distance=similarity[movie_index]
    movies_similar_index=sorted(list(enumerate(distance)), reverse=True, key=lambda x:x[1])[1:6]

    recommended_movies=[]
    recommended_movies_poster=[]
    for i in movies_similar_index:
        movie_ID=movies_list.iloc[i[0]].movie_id
        recommended_movies.append(movies_list.iloc[i[0]].title)
        #fetch poster from api
        recommended_movies_poster.append(fetch_poster(movie_ID))
    return recommended_movies,recommended_movies_poster

movies_list=pickle.load(open('ML_Project\MovieRecommender\movies.pkl','rb'))

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies_list["tags"]).toarray()
similarity = cosine_similarity(vectors)

st.title('Movie Recommender System')

selected_movie_name=st.selectbox("Select your Favourite Movie",movies_list['title'].values)

if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.caption(names[0])
        st.image(posters[0])
    with col2:
        st.caption(names[1])
        st.image(posters[1])
    with col3:
        st.caption(names[2])
        st.image(posters[2])
    with col4:
        st.caption(names[3])
        st.image(posters[3])
    with col5:
        st.caption(names[4])
        st.image(posters[4])