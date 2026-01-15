from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv("../.env")

def get_models(client):
    # The models.list() method returns an object containing all active models
    models = client.models.list()
    # You can iterate over models.data to access individual model details
    for model in models.data:
        print(model.id)

def main():
    client = Groq() 
    get_models(client)

if __name__ == "__main__":
    main()    
