from fastapi import FastAPI,HTTPException 
from pydantic import BaseModel

#here we are going to make a api of the sentiment analysis 
#and going to upload on the render.com so we can access it 
app=FastAPI()
class Sentiment(BaseModel):
    text: str 
    
    


#lets load the vectorizer and model for the sentiment analysis 
import joblib 

model=joblib.load('log_reg_model.pkl')
vectorizer=joblib.load('tfidf_vectorizer.pkl')


def get_sentiment(text):
    #here we have to get the confidence and the sentiment from the text 
    vectors=vectorizer.transform([text])
    label=model.predict(vectors)
    #0 for the negative class and 1 for the positive class 
    negative,positive=model.predict_proba(vectors)[0]
    
    
    sentiment=''
    if negative>positive:
        confidence=negative 
        sentiment='Negative'
    elif positive>negative:
        confidence=positive
        sentiment='Positive'
    else:
        sentiment='Neutral'
        confidence=positive+negative
        
    return sentiment,confidence
    


@app.get('/')
def home():
        return {'message': 'Welcome to the sentiment analysis API'}
    
@app.post('/sentiment')
def sentiment_analysis(sentiment: Sentiment):
    text = sentiment.text
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    #let's handle the text input and predict the sentiment 
    
    sentiment,confidence=get_sentiment(text)
    return {'sentiment':sentiment,'confidence':confidence
            }