from flask import Flask, request, jsonify, make_response, send_file
from enrich_kg import return_relations, add_relations_to_kg
from disambiguation.reason import reason_and_update
from search_kg import query_knowledge_graph, delete_relation_kg
from bulk_relations import extract_relations, add_bulk_relations_to_kg
from flask_cors import CORS
import requests
import tempfile
import json
import os
import logging

app = Flask(__name__)
CORS(app)


GRAPHDB_URL = os.getenv('GRAPHDB_URL', 'http://localhost:7200')
# Default to LM Studio (local)
AVAILABLE_MODELS = ["lmstudio"] 
DEFAULT_REPO = "oncograph"

@app.route('/api/showRelations', methods=['POST'])
def show_relations():
    try:
        text = request.data.decode('utf-8')
        relations = return_relations(text)
        response = make_response(jsonify(relations))
        return response
    except Exception as e:
        logging.error(f"Error showing relations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/addRelations', methods=['POST'])
def add_relations():
    try:
        json_file = request.get_json()
        repo_id = json_file.get("repo_id", DEFAULT_REPO)
        add_relations_to_kg(json_file["relations"], repo_id)

        response = make_response(jsonify({"status": "Successfully completed", "repo_id": repo_id}))
        return response
    except Exception as e:
        logging.error(f"Error adding relations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/reason', methods=['GET'])
def reason_kg():
    repo_id = request.args.get('repo_id', DEFAULT_REPO)
    try:
        reason_and_update(repo_id)
        response = make_response(jsonify({"status": "Successfully completed reasoning", "repo_id": repo_id}))
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search():
    query_text = request.args.get('q', '')
    filter_type = request.args.get('type', '')
    repo_id = request.args.get('repo_id', DEFAULT_REPO)

    try:
        data = query_knowledge_graph(query_text, filter_type, repo_id)
        response = make_response(jsonify(data))
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/deleteRelation', methods=['POST'])
def delete_relation():
    data = request.json

    subject = data.get('subject')
    predicate = data.get('predicate')
    object_ = data.get('object')
    repo_id = data.get("repo_id", DEFAULT_REPO)

    try:
        delete_relation_kg(subject, predicate, object_, repo_id)
        response = make_response(jsonify({"status": "Successfully completed"}))
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/available_repositories", methods=["GET"])
def available_repositories():
    try:
        headers = {"Accept": "application/json, */*;q=0.9"}
        auth = None
        if os.getenv("GRAPHDB_USERNAME") and os.getenv("GRAPHDB_PASSWORD"):
            auth = (os.getenv("GRAPHDB_USERNAME"), os.getenv("GRAPHDB_PASSWORD"))

        response = requests.get(
            f"{GRAPHDB_URL}/rest/repositories",
            headers=headers,
            auth=auth,
            timeout=10,
        )

        if response.status_code == 200:
            repos = response.json()
            available_repos = [{"id": repo["id"]} for repo in repos]
            return jsonify(available_repos)
        else:
            return jsonify({"error": "Failed to fetch repositories", "status": response.status_code}), response.status_code
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/upload_abstract", methods=["POST"])
def upload_abstract():
    # Model optional, default to the local one
    if 'file' not in request.files:
        return jsonify({"error": "File is required"}), 400

    file = request.files['file']
    model = request.form.get("model", "lmstudio")

    try:
        content = json.loads(file.read().decode('utf-8'))
        # If content is a list of abstracts, this works. If it's a single item wrap it.
        if not isinstance(content, list):
            content = [content]
            
        # Ensure format for bulk_relations (needs id and text)
        formatted_abstracts = []
        for item in content:
            txt = item.get("abstract") or item.get("text") or item.get("AbstractText") or ""
            pid = item.get("source_id") or item.get("pmid") or item.get("id") or "unknown"
            if txt:
                formatted_abstracts.append({"id": pid, "text": txt})

        relations = extract_relations(formatted_abstracts, model)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
        json.dump(relations, temp_file, ensure_ascii=False, indent=2)
        temp_file.close()

        return send_file(temp_file.name, as_attachment=True, download_name="extracted_relations.json",
                         mimetype="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_relations", methods=["POST"])
def upload_relations():
    if 'file' not in request.files:
        return jsonify({"error": "File is required"}), 400

    file = request.files['file']
    repo_id = request.form.get("repo_id", DEFAULT_REPO)

    try:
        relations = json.load(file)
        add_bulk_relations_to_kg(relations, repo_id)
        response = make_response(jsonify({"status": "Successfully completed", "count": len(relations)}))
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
