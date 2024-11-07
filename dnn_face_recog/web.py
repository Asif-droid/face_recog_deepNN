from flask import Flask, jsonify, render_template
import random

app = Flask(__name__)

# Sample data simulating the list from another system
people = [
]

@app.route('/')
def index():
    return render_template('indx.html')

@app.route('/get_status')
def get_status():
    # Simulate dynamic updates
    for person in people:
        person['status'] = random.choice(['present', 'absent'])
    return jsonify(people)

if __name__ == '__main__':
    app.run(debug=True)
