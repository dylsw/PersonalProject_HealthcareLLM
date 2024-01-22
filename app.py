# app.py


from flask import Flask, request, jsonify
from flask_cors import CORS

import preprocess_embeddings
import process_user  # This is the script you created earlier for handling user queries

app = Flask(__name__)
CORS(app)#, resources={r"/get_answer": {"origins": "http://localhost:3000"}})

@app.route('/reload_embeddings', methods=['POST'])
def reload_embeddings():
    preprocess_embeddings.handle_reload()
    return jsonify({'status': 'Embeddings reloaded successfully'})

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json['question']
    answer = process_user.handle_query(user_question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
