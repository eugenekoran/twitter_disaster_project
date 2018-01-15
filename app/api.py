# encoding: utf-8
from flask import Flask
from flask_restful import Resource, Api
from flask_restful import reqparse
import cPickle as pickle

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from utils import load_production_model

app = Flask(__name__)
api = Api(app)

with open('tokenizer.pkl', 'r') as f:
    tokenizer = pickle.load(f)

vocab_size, seq_len, model = load_production_model()

model.load_weights('model_weigths.krs')

def predict(twit, treshold=0.5):
    '''
    Predict
    '''
    x = tokenizer.texts_to_sequences([twit])
    x = [[vocab_size - 1 if i >= vocab_size else i for i in line] for line in x]
    x = sequence.pad_sequences(x, maxlen=seq_len)
    prob = model.predict_proba(x)[0][0]
    return 'Relevant' if prob > treshold else 'Not Relevant', prob

class Prediction(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('twit', help='slength cannot be converted')
        args = parser.parse_args()

        prediction, prob = predict(args['twit'])

        print "PREDICTION: {} PROB_of_RELEVANT: {}".format(prediction, prob)

        return {
                'twit': args['twit'],
                'prediction': prediction,
                'prob_of_relevant': str(prob)
               }

api.add_resource(Prediction, '/prediction')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
