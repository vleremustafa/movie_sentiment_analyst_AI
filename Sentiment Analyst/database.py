import sqlite3

class DatabaseManager:
    def __init__(self, db_path='movie_diary.db'):
        self.db_path = db_path
        self.init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users 
                         (username TEXT PRIMARY KEY, hashed_password TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS reviews 
                         (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          owner TEXT, date TEXT, movie TEXT, sentiment TEXT, confidence TEXT)''')
            conn.commit()

    def create_user(self, username, hashed_password):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", 
                         (username, hashed_password))

    def get_user(self, username):
        with self._get_connection() as conn:
            return conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    def save_review(self, owner, date, movie, sentiment, confidence):
        with self._get_connection() as conn:
            conn.execute("""INSERT INTO reviews (owner, date, movie, sentiment, confidence) 
                            VALUES (?, ?, ?, ?, ?)""", (owner, date, movie, sentiment, confidence))

    def get_history(self, username):
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM reviews WHERE owner = ? ORDER BY id DESC", (username,)).fetchall()
            return [dict(row) for row in rows]

    def delete_review(self, review_id):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM reviews WHERE id = ?", (review_id,))

    def update_review(self, review_id, movie, sentiment, confidence):
        with self._get_connection() as conn:
            conn.execute("UPDATE reviews SET movie = ?, sentiment = ?, confidence = ? WHERE id = ?", 
                         (movie, sentiment, confidence, review_id))