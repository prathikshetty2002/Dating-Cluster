from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'

# Load the dataset
data = pd.read_csv('matcher.csv')

feature_names=['score','is_habit_drink','is_habit_smoke','is_subscribed','A','B','C','D','E']
# Create the model
model = NearestNeighbors(metric='cosine', algorithm='brute')
# model.set_params(**{'feature_names': feature_names})


@app.route('/recommend',methods=['GET'])
def recommend():
    # Parse the input parameters from the URL query string
    uid = int(request.args.get('uid'))
    whoToDate = request.args.get('whoToDate')
    # gender = data.loc[data['uid'] == uid, 'whoToDate'].iloc[0]
    gender = data[data['gender'] == whoToDate]
    le = LabelEncoder()
    gender['gender'] = le.fit_transform(gender['gender'])
    model.fit(gender.drop('uid', axis=1))
    print(f"The preffered gender of user {uid} is {gender}.")
    
    k = int(request.args.get('k', 5))
    
    # Find the index of the user with the given uid
    user_index = data[data['uid'] == uid].index[0]
    
    # Calculate the cosine similarities between the user and all other users
    distances, indices = model.kneighbors(gender.drop('uid', axis=1).iloc[user_index].values.reshape(1,-1), n_neighbors=k+1)
    
    # Get the indices of the k most similar users (excluding the user itself)
    similar_indices = indices.squeeze()[1:]
    
    # Get the uids of the k most similar users
    similar_uids = data.iloc[similar_indices]['uid'].tolist()
    similar_genders = data.iloc[similar_indices]['gender'].tolist()
    similarity_scores = (1 - distances.squeeze()[1:]).tolist()
    print(similar_uids)
    recommendations = []
    for i in range(k):
        recommendations.append({
            'uid': similar_uids[i],
            'gender': [similar_genders[i]][0],
            'confidence_score': similarity_scores[i]
        })
    recommendated_users=[]
    if whoToDate=='F':
        for rec in recommendations:
            if rec["gender"] == "F":
                recommendated_users.append(rec)
    elif whoToDate=='M':
        for rec in recommendations:
            if rec["gender"] == "M":
                recommendated_users.append(rec)
    else:
        pass
            
        
    # Return the recommendations as a JSON response
    return jsonify({'recommendations': recommendated_users})
    

if __name__ =='__main__':
    app.run(debug=True, host="0.0.0.0")