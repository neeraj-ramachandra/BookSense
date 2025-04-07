import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load your pivot table

pt = pd.read_csv("processed_data.csv", index_col=0)  # Replace this with actual pivot table
similarities = cosine_similarity(pt)

books = pd.read_csv("datasets/books.csv")
# Calculate average rating
book_avg_rating = pt.replace(0, np.nan).mean(axis=1).reset_index()
book_avg_rating.columns = ['Book-Title', 'Average-Rating']

# Merge with metadata
books = books.merge(book_avg_rating, on='Book-Title', how='left')

st.title("📚 Book Recommendation System")

option = st.radio("Choose how to get recommendations:", ["By Book Title", "By User ID"])

def get_book_info(title):
    info = books[books['Book-Title'] == title].drop_duplicates('Book-Title')
    if not info.empty:
        row = info.iloc[0]
        return row['Book-Author'], row['Image-URL-M'], round(row['Average-Rating'], 2)
    return "Unknown", "", 0

def recommend(book_name=None, user_id=None):
    if book_name:
        if book_name in pt.index:
            index = np.where(pt.index == book_name)[0][0]
            similar_books_list = sorted(
                list(enumerate(similarities[index])), key=lambda x: x[1], reverse=True
            )[1:11]
            return [pt.index[book[0]] for book in similar_books_list]
        else:
            return ["❌ Book Not Found"]

    elif user_id:
        user_id = str(user_id)
        if user_id in pt.columns:
            user_ratings = pt[user_id]
            liked_books = user_ratings[user_ratings > 0].sort_values(ascending=False)

            if liked_books.empty:
                return ["⚠️ User has not rated any books yet."]

            recommendation_scores = {}
            for book in liked_books.index:
                index = np.where(pt.index == book)[0][0]
                similar_books = list(enumerate(similarities[index]))
                for sim_index, score in similar_books:
                    similar_book = pt.index[sim_index]
                    if similar_book not in liked_books.index:
                        recommendation_scores[similar_book] = recommendation_scores.get(similar_book, 0) + score

            recommended = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            return [title for title, score in recommended]
        else:
            return ["❌ User ID Not Found"]

if option == "By Book Title":
    book_name = st.selectbox("Select a Book", pt.index.tolist())
    if st.button("Get Recommendations"):
        results = recommend(book_name=book_name)
        st.subheader("📖 Top Recommendations:")
        for i, r in enumerate(results, 1):
            # st.write(f"{i}. {r}")
            author, img, rating = get_book_info(r)
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image(img, width=80)
            with col2:
                st.markdown(f"**{r}**  \n*by {author}*  \n*Rated {rating}/10*")

elif option == "By User ID":
    user_id_input = st.text_input("Enter User ID", value="")
    if st.button("Get Recommendations"):
        try:
            user_id = int(user_id_input)
            results = recommend(user_id=user_id)
            st.subheader("👤 Top Recommendations:")
            for i, r in enumerate(results, 1):
                # st.write(f"{i}. {r}")
                author, img, rating = get_book_info(r)
                col1, col2 = st.columns([1, 5])
                with col1:
                    if (img == ""):
                        st.image("https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg", width=80)
                    else:
                        st.image(img, width=80)
                with col2:
                    st.markdown(f"**{r}**  \n*by {author}*  \n*Rated {rating}/10*")
        except ValueError:
            st.error("Please enter a valid numeric User ID.")
