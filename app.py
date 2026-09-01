from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime
import PyPDF2
from transformers import pipeline
import nltk
import torch
import tempfile

app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('punkt')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'ppt', 'pptx'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
MAX_PAGES = 50  # Maximum number of pages for summarization

# Initialize the summarization model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
print("Loading AI Model directly (Bypassing Pipeline)...")

# 1. Load model and tokenizer directly
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Define a class that acts exactly like the summarizer pipeline
class CustomSummarizer:
    def __call__(self, text, max_length=150, min_length=30, do_sample=False):
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate the summary
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            min_length=min_length, 
            do_sample=do_sample
        )
        
        # Decode back to text
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Return list-of-dict format to match what your existing functions expect
        return [{"summary_text": summary_text}]

# 3. Create the object that the rest of your script uses
summarizer = CustomSummarizer()
print("AI Model loaded successfully!")


# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Database initialization
def init_db():
    conn = sqlite3.connect('notes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            subject TEXT NOT NULL,
            subject_code TEXT,
            description TEXT,
            author_type TEXT NOT NULL,
            filename TEXT NOT NULL,
            upload_date DATETIME NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            sender_type TEXT NOT NULL,
            reply_to INTEGER,
            timestamp DATETIME NOT NULL,
            is_answered BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/api/auth', methods=['POST'])
def authenticate():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Hard-coded credentials as requested
    if username == 'test' and password == '1234':
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check authentication
    username = request.form.get('username')
    password = request.form.get('password')
    
    if username != 'test' or password != '1234':
        return jsonify({'success': False, 'message': 'Authentication required'}), 401

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'File type not allowed'}), 400

    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        unique_filename = timestamp + filename
        file.save(os.path.join(UPLOAD_FOLDER, unique_filename))

        # Save metadata to database
        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO notes (title, subject, subject_code, description, author_type, filename, upload_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.form.get('title'),
            request.form.get('subject'),
            request.form.get('subject_code'),
            request.form.get('description'),
            request.form.get('author_type'),
            unique_filename,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'filename': unique_filename
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def file_exists(filename):
    return os.path.exists(os.path.join(UPLOAD_FOLDER, filename))

def clean_deleted_notes():
    try:
        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        
        # Get all notes
        c.execute('SELECT id, filename FROM notes')
        notes = c.fetchall()
        
        # Check each note's file existence
        for note_id, filename in notes:
            if not file_exists(filename):
                c.execute('DELETE FROM notes WHERE id = ?', (note_id,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error cleaning deleted notes: {str(e)}")

@app.route('/api/notes', methods=['GET'])
def get_notes():
    try:
        # Clean up deleted notes first
        clean_deleted_notes()
        
        subject_code = request.args.get('subject_code')
        sort_by = request.args.get('sort', 'date-desc')

        conn = sqlite3.connect('notes.db')
        c = conn.cursor()

        # Base query
        query = 'SELECT * FROM notes'
        params = []

        # Add subject code filter if provided
        if subject_code:
            query += ' WHERE subject_code = ?'
            params.append(subject_code)

        # Add sorting
        if sort_by == 'date-asc':
            query += ' ORDER BY upload_date ASC'
        elif sort_by == 'date-desc':
            query += ' ORDER BY upload_date DESC'
        elif sort_by == 'title':
            query += ' ORDER BY title ASC'

        c.execute(query, params)
        notes = c.fetchall()
        conn.close()

        notes_list = []
        for note in notes:
            # Only include notes whose files exist
            if file_exists(note[6]):  # note[6] is the filename
                notes_list.append({
                    'id': note[0],
                    'title': note[1],
                    'subject': note[2],
                    'subject_code': note[3],
                    'description': note[4],
                    'author_type': note[5],
                    'filename': note[6],
                    'upload_date': note[7]
                })

        return jsonify({'success': True, 'notes': notes_list})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/chat/messages', methods=['GET'])
def get_chat_messages():
    try:
        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        c.execute('SELECT * FROM chat_messages ORDER BY timestamp ASC')
        messages = c.fetchall()
        conn.close()

        messages_list = []
        for msg in messages:
            messages_list.append({
                'id': msg[0],
                'message': msg[1],
                'sender_type': msg[2],
                'reply_to': msg[3],
                'timestamp': msg[4],
                'is_answered': bool(msg[5])
            })

        return jsonify({'success': True, 'messages': messages_list})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    try:
        data = request.json
        message = data.get('message')
        sender_type = data.get('sender_type')
        reply_to = data.get('reply_to')

        if not message or not sender_type:
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400

        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO chat_messages (message, sender_type, reply_to, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (message, sender_type, reply_to, datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/chat/mark_answered', methods=['POST'])
def mark_message_answered():
    try:
        data = request.json
        message_id = data.get('message_id')

        if not message_id:
            return jsonify({'success': False, 'message': 'Missing message ID'}), 400

        conn = sqlite3.connect('notes.db')
        c = conn.cursor()
        c.execute('UPDATE chat_messages SET is_answered = 1 WHERE id = ?', (message_id,))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if len(pdf_reader.pages) > MAX_PAGES:
                raise ValueError(f"PDF exceeds maximum page limit of {MAX_PAGES} pages")
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            if not text.strip():
                raise ValueError("No text content found in PDF")
                
            return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def generate_summary(text):
    try:
        # Split text into chunks if it's too long
        max_chunk_length = 1024
        chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        raise Exception(f"Error generating summary: {str(e)}")

@app.route('/api/summarize/<filename>', methods=['GET'])
def summarize_file(filename):
    try:
        # Check if file exists
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': 'File not found'}), 404

        # Extract text from PDF
        text = extract_text_from_pdf(file_path)
        
        # Generate summary
        summary = generate_summary(text)
        
        # Save summary to a temporary file
        summary_filename = f"summary_{filename.rsplit('.', 1)[0]}.txt"
        summary_path = os.path.join(UPLOAD_FOLDER, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return jsonify({
            'success': True,
            'message': 'Summary generated successfully',
            'summary_filename': summary_filename
        })

    except ValueError as ve:
        return jsonify({'success': False, 'message': str(ve)}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True) 