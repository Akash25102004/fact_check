import requests 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
 
# Function to collect training data 
def collect_training_data(): 
    data = [] 
    while True: 
        statement = input("Enter statement (or type 'done' to finish): ") 
        if statement.lower() == 'done': 
            break 
        label = input("Enter label (1 for True, 0 for False): ") 
        if label not in ['0', '1']: 
            print("Invalid label. Enter 1 for True or 0 for False.") 
            continue 
        data.append({"statement": statement, "label": int(label)}) 
    return data 
 
# Function to preprocess and split the data 
def preprocess_data(data): 
    statements = [item['statement'] for item in data] 
    labels = [item['label'] for item in data] 
     
    # Convert text to features using TF-IDF Vectorizer 
vectorizer = TfidfVectorizer(stop_words='english') 
X = vectorizer.fit_transform(statements) 
y = labels 
return X, y, vectorizer 
# Function to train the model 
def train_model(X, y): 
model = LogisticRegression(max_iter=1000) 
model.fit(X, y) 
return model 
# Function to make a prediction using the trained model 
def predict_statement(model, vectorizer, statement): 
statement_tfidf = vectorizer.transform([statement]) 
prediction = model.predict(statement_tfidf)[0] 
return "True" if prediction == 1 else "False" 
# Function to get fact-check results from an external API (e.g., ClaimBuster) 
def get_fact_check(statement, api_key): 
api_endpoint = f"https://idir.uta.edu/claimbuster/api/v2/score/text/{statement}" 
headers = {"x-api-key": api_key} 
try: 
response = requests.get(url=api_endpoint, headers=headers) 
response.raise_for_status() 
return response.json() 
except requests.exceptions.RequestException as e: 
        print(f"Error fetching fact-check results: {e}") 
        return {"error": f"Error fetching fact-check results: {e}"} 
 
# Function to combine AI prediction and API fact-check results 
def check_fact(statement, model, vectorizer, api_key): 
    ai_result = predict_statement(model, vectorizer, statement) 
    print(f"AI Prediction: {ai_result}") 
 
    fact_check_result = get_fact_check(statement, api_key) 
     
    return ai_result, fact_check_result 
 
# Main function to collect data, train model, and check facts 
def main(): 
    # Collect training data 
    data = collect_training_data() 
     
    if len(data) < 2: 
        print("Not enough data to train the model. At least 2 samples are required.") 
        return 
     
    # Preprocess and train the model 
    X, y, vectorizer = preprocess_data(data) 
    model = train_model(X, y) 
     
    # Ask user to check facts 
    while True: 
        statement = input("\nEnter a statement to check (or type 'exit' to quit): ") 
        if statement.lower() == 'exit': 
            print("Exiting the program.") 
            break 
         
        # Get API key and check fact 
        api_key = input("Enter your API key for fact-checking (e.g., ClaimBuster API key): ") 
        ai_result, fact_check_result = check_fact(statement, model, vectorizer, api_key) 
 
        # Display the external fact-check results 
        print("\nExternal Fact-Check Results:") 
        if 'error' in fact_check_result: 
            print(fact_check_result['error']) 
        else: 
            articles = fact_check_result.get('value', []) 
            if articles: 
                for article in articles: 
                    print(f"Title: {article['title']}") 
                    print(f"Description: {article['description']}") 
                    print(f"URL: {article['url']}\n") 
            else: 
                print("No fact-checking results found.") 
 
if __name__ == '__main__': 
    main() 