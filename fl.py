
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
app = Flask(__name__)
import os

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/execute/')
def my_link():
  print ('EXECUTE EMOTION DETECTION')
  os.system('python im.py')
  return 'INITIATED'

  
if __name__ == '__main__':
  app.run(debug=True)
