import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
GRAPHDB_URL = os.getenv('GRAPHDB_URL', 'http://localhost:7200')
REPO_ID = "oncograph"  # Your repository ID
ONTOLOGY_FILE = r"d:\Projects\OncoGraphRAG\data\OncoGraphRAG_ont.rdf"

def upload_ontology():
    url = f"{GRAPHDB_URL}/repositories/{REPO_ID}/statements"
    headers = {
        "Content-Type": "application/rdf+xml"
    }
    
    # Read the RDF file
    try:
        with open(ONTOLOGY_FILE, 'rb') as f:
            data = f.read()
            
        print(f"Uploading schema from {ONTOLOGY_FILE}...")
        
        # POST the data to the default graph
        response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 204:
            print("✅ Ontology Schema uploaded successfully!")
        else:
            print(f"❌ Failed to upload. Status: {response.status_code}")
            print(response.text)
            
    except FileNotFoundError:
        print(f"Error: Ontology file not found at {ONTOLOGY_FILE}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    upload_ontology()