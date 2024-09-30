import requests
import tkinter as tk
from tkinter import messagebox, simpledialog

API_KEY = '595e2a513528def41d589997d0cdc80d'
BASE_URL = 'https://api.themoviedb.org/3'

"""
Fetches the movie ID for a given movie title from the movie database API.

Args:
  movie_title (str): The title of the movie to search for.

Returns:
  int or None: The ID of the movie if found, otherwise None.
"""
def get_movie_id(movie_title):
    url = f"{BASE_URL}/search/movie"
    params = {
        'api_key': API_KEY,
        'query': movie_title
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get('results')
    if results:
        return results[0]['id']
    else:
        return None

"""
Fetches movie recommendations based on a given movie ID from the movie database API.

Args:
  movie_id (int): The ID of the movie to get recommendations for.

Returns:
  list: A list of recommended movie titles.
"""
def get_recommendations(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params = {
        'api_key': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get('results', [])
    recommendations = [movie['title'] for movie in results[:10]]
    return recommendations

"""
Starts the recommendation process by prompting the user for a movie title,
fetching the movie ID, and displaying the recommendations.

Args:
  None

Returns:
  None
"""
def start_recommendation():
    movie_title = simpledialog.askstring("Input", "Enter your favorite movie:", parent=root)
    if movie_title:
        movie_id = get_movie_id(movie_title)
        if movie_id:
            recommendations = get_recommendations(movie_id)
            if recommendations:
                rec_text = "\n".join(recommendations)
                messagebox.showinfo("Recommendations", f"Top 10 Movie Recommendations:\n\n{rec_text}")
            else:
                messagebox.showinfo("Recommendations", "No recommendations found.")
        else:
            messagebox.showerror("Error", f"Movie '{movie_title}' not found.")
    else:
        messagebox.showerror("Error", "Please enter a valid movie title.")

# Create GUI
root = tk.Tk()
root.title("Movie Recommendation System")

canvas = tk.Canvas(root, height=200, width=400)
canvas.pack()

frame = tk.Frame(root)
frame.place(relwidth=1, relheight=1)

welcome_label = tk.Label(frame, text="Welcome to the Movie Recommendation System!", font=('Helvetica', 14))
welcome_label.pack(pady=20)

start_button = tk.Button(frame, text="Start", padx=10, pady=5, command=start_recommendation)
start_button.pack()

root.mainloop()
