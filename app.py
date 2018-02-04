from flask import Flask, request, render_template
import pickle
import numpy

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/getdelay',methods=['POST', 'GET'])
def get_delay():
    if not request.method == 'POST':
        return render_template('index.html')

    result=request.form
    categories = pickle.load(open('categories.pkl', 'rb'))
    features = numpy.zeros(len(categories))

    try:
        features[categories['DAY_OF_WEEK_' + str(result['weekday'])]] = 1
        features[categories['UNIQUE_CARRIER_' + str(result['carrier'])]] = 1
        features[categories['ORIGIN_' + str(result['origin'])]] = 1
        features[categories['DEST_' + str(result['dest'])]] = 1
        features[categories['DEP_HOUR_' + str(result['dep_hour'])]] = 1
    except:
        pass

    logmodel = pickle.load(open('logmodel.pkl', 'rb'))
    prediction = logmodel.predict([features])

    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
  app.run()
