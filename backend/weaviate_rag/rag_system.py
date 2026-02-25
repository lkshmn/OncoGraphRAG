import types
import weaviate
from weaviate.classes.query import Filter, QueryReference
import itertools
import tempfile
import atexit
import shutil
import os
import re
from openai import OpenAI
from weaviate_rag.query_analyzer import analyze_query
from dotenv import load_dotenv

load_dotenv()

# LM Studio Configuration
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL", "local-model")

class GraphRAGSystem:

    def __init__(self, question, model_config=None):
        """Initialize the analyzer with a question and set up Weaviate and LM Studio clients."""
        self.question = question
        # Default to LM Studio configuration
        self.model_config = model_config or {
            'provider': 'lmstudio', 
            'model_name': LMSTUDIO_MODEL
        }
        
        # Initialize OpenAI client pointing to LM Studio
        self.api_client = OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key=LMSTUDIO_API_KEY
        )

        # Oncology Acronyms Mapping
        self.acronyms = {
            "nsclc": "non-small cell lung cancer",
            "sclc": "small cell lung cancer",
            "tnbc": "triple-negative breast cancer",
            "crc": "colorectal cancer",
            "pdl1": "programmed death-ligand 1",
            "tki": "tyrosine kinase inhibitor",
            "os": "overall survival",
            "pfs": "progression-free survival",
            "ae": "adverse event",
            "gbm": "glioblastoma",
            "aml": "acute myeloid leukemia",
            "all": "acute lymphoblastic leukemia",
            "cll": "chronic lymphocytic leukemia",
            "cml": "chronic myeloid leukemia",
            "mm": "multiple myeloma",
            "nhl": "non-hodgkin lymphoma",
            "hl": "hodgkin lymphoma",
            "pdac": "pancreatic ductal adenocarcinoma",
            "hr+": "hormone receptor-positive",
            "her2": "human epidermal growth factor receptor 2",
            "brca": "breast cancer gene",
        }

        self.temp_dir = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, self.temp_dir, ignore_errors=True)
        
        # Connect to Weaviate (ensure Weaviate is running)
        self.client = weaviate.connect_to_local()
        
        # Define collections
        self.entity_collection = self.client.collections.get("Entity")
        self.relation_collection = self.client.collections.get("Relation")
        self.entity_id_to_name = {}

    def expand_terms(self, terms):
        """Replace acronyms with their full forms in the list of terms or noun phrases, normalizing case."""
        expanded_terms = []
        for term in terms:
            words = term.split()
            expanded_words = [self.acronyms.get(word.lower(), word) for word in words]
            expanded_terms.append(" ".join(expanded_words))
        return expanded_terms

    def retrieve_entities(self):
        """Retrieve entities based on noun phrases AND keywords from the question."""
        query_analysis = analyze_query(self.question)

        terms_to_search = []

        if query_analysis.get('noun_phrases', []):
            terms_to_search.extend(query_analysis['noun_phrases'])

        if query_analysis.get('keywords', []):
            for keyword in query_analysis['keywords']:
                if not any(keyword.lower() in phrase.lower() for phrase in terms_to_search):
                    terms_to_search.append(keyword)

        if not terms_to_search:
            terms_to_search = [self.question]

        expanded_terms = self.expand_terms(terms_to_search)

        entity_results = {}
        for term in expanded_terms:
            try:
                # Using hybrid search or bm25 depending on Weaviate configuration
                entity_search_res = self.entity_collection.query.bm25(
                    query=term,
                    limit=5,
                    return_properties=["name", "type"]
                )
                if entity_search_res.objects:
                    entity_results[term] = entity_search_res.objects
            except Exception as e:
                print(f"Error searching for entity term '{term}': {e}")

        return entity_results

    def get_entity_ids_and_names(self, entity_results):
        """Extract entity IDs and names from search results."""
        entity_ids = []
        entity_names = []
        for noun_phrase, entity_objects in entity_results.items():
            for entity_obj in entity_objects:
                entity_ids.append(entity_obj.uuid)
                name = entity_obj.properties.get('name', 'Unknown')
                entity_names.append(name)
                self.entity_id_to_name[entity_obj.uuid] = name
        return entity_ids, entity_names

    def retrieve_pairwise_relations(self, entity_ids):
        """Retrieve pairwise relationships between entities."""
        if not entity_ids:
            return [], set()

        entity_pairs = set(itertools.permutations(entity_ids, 2))
        
        # Note: Weaviate filtering syntax
        combined_filter = (
                Filter.by_ref("relation_subject").by_id().contains_any(entity_ids) &
                Filter.by_ref("relation_object").by_id().contains_any(entity_ids)
        )
        
        references = [
            QueryReference(link_on="relation_subject", return_properties=["name", "type"]),
            QueryReference(link_on="relation_object", return_properties=["name", "type"]),
            QueryReference(link_on="hasPublication", return_properties=["name", "content"])
        ]
        
        try:
            relations = self.relation_collection.query.fetch_objects(
                filters=combined_filter,
                return_properties=["relation_predicate"],
                return_references=references,
                limit=100
            )
        except Exception as e:
            print(f"Error executing pairwise query: {e}")
            return [], set()

        seen = set()
        pairwise_relations = []
        involved_entities = set()
        
        for o in relations.objects:
            # Safely get references
            if not o.references.get("relation_subject") or not o.references.get("relation_object"):
                continue
                
            subject_obj = o.references["relation_subject"].objects[0]
            object_obj = o.references["relation_object"].objects[0]
            
            subject_id = subject_obj.uuid
            object_id = object_obj.uuid
            
            if (subject_id, object_id) in entity_pairs:
                predicate = o.properties.get("relation_predicate", "related_to")
                triple = (subject_id, predicate, object_id)
                
                if triple not in seen:
                    seen.add(triple)
                    subject_name = subject_obj.properties.get("name", "Unknown")
                    object_name = object_obj.properties.get("name", "Unknown")
                    pairwise_relations.append((subject_name, predicate, object_name))
                    involved_entities.add(subject_id)
                    involved_entities.add(object_id)
                    
        return pairwise_relations, involved_entities

    def retrieve_individual_relations(self, entity_ids, involved_entities):
        """Retrieve individual relations for entities not in pairwise relations."""
        entities_to_check = [eid for eid in entity_ids if eid not in involved_entities]
        
        # If all entities are involved in pairs, we might still want some individual context, 
        # but the original logic returns early. Keeping logic similar.
        if not entities_to_check:
            return {}

        entity_relations = {eid: [] for eid in entities_to_check} 
        seen = set()
        
        for eid in entities_to_check:
            entity_filter = (
                    Filter.by_ref("relation_subject").by_id().equal(eid) |
                    Filter.by_ref("relation_object").by_id().equal(eid)
            )
            try:
                relations = self.relation_collection.query.fetch_objects(
                    filters=entity_filter,
                    return_properties=["relation_predicate"],
                    return_references=[
                        QueryReference(link_on="relation_subject", return_properties=["name", "type"]),
                        QueryReference(link_on="relation_object", return_properties=["name", "type"]),
                    ],
                    limit=20 # Reduced limit for individual context to avoid noise
                )
                for o in relations.objects:
                    if not o.references.get("relation_subject") or not o.references.get("relation_object"):
                        continue

                    subject_obj = o.references["relation_subject"].objects[0]
                    object_obj = o.references["relation_object"].objects[0]
                    subject_id = subject_obj.uuid
                    object_id = object_obj.uuid
                    
                    predicate = o.properties.get("relation_predicate", "related_to")
                    triple = (subject_id, predicate, object_id)
                    
                    if triple not in seen:
                        seen.add(triple)
                        subject_name = subject_obj.properties.get("name", "Unknown")
                        object_name = object_obj.properties.get("name", "Unknown")
                        # Add to the entity that triggered this search
                        if eid in entity_relations:
                             entity_relations[eid].append((subject_name, predicate, object_name))
            except Exception as e:
                print(f"Error querying relations for entity {self.entity_id_to_name.get(eid, 'unknown')}: {e}")
                
        return entity_relations

    def build_relation_string(self, pairwise_relations, entity_relations):
        """Construct a string of all relations for summary generation."""
        relation_string = ""
        relation_counter = 1

        for subj, pred, obj in pairwise_relations:
            relation_string += f"{relation_counter}. {subj} {pred} {obj}\n"
            relation_counter += 1

        for eid, rels in entity_relations.items():
            if rels:
                for subj, pred, obj in rels:
                    relation_string += f"{relation_counter}. {subj} {pred} {obj}\n"
                    relation_counter += 1
        return relation_string


    def remove_think_tags(self, response):
        """Removes <think> tags often present in reasoning models like DeepSeek."""
        cleaned_content = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned_content.strip()


    def generate_summary(self, relation_string):
        """Generate a summary using the configured Local LLM model."""
        if not relation_string:
            return "No specific relationships found in the knowledge graph for this query."

        # Prompt tuned for Oncology
        prompt = (
            f"You are an expert Oncology Assistant. \n"
            f"Question: '{self.question}'\n\n"
            f"Relevant Knowledge Graph Facts:\n{relation_string}\n\n"
            f"Based STRICTLY on the facts provided above, synthesize a concise answer to the question. "
            f"If the facts suggest a clinical trial result, mention findings (e.g., 'improved OS', 'reduced toxicity'). "
            f"Do not hallucinate external information. If the facts are insufficient, state what is known."
        )
        
        try:
            response = self.api_client.chat.completions.create(
                model=self.model_config['model_name'],
                messages=[
                    {"role": "system", "content": "You are a precise oncology knowledge assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            return self.remove_think_tags(content)

        except Exception as e:
            return f"Error processing response with Local LLM: {e}"

    def analyze(self):
        """Orchestrate the analysis process and return the summary."""
        try:
            entity_results = self.retrieve_entities()
            entity_ids, entity_names = self.get_entity_ids_and_names(entity_results)
            
            if not entity_ids:
                return "No relevant entities found in the Knowledge Graph for your query."

            pairwise_relations, involved_entities = self.retrieve_pairwise_relations(entity_ids)
            entity_relations = self.retrieve_individual_relations(entity_ids, involved_entities)
            
            relation_string = self.build_relation_string(pairwise_relations, entity_relations)
            summary = self.generate_summary(relation_string)
            return summary
        finally:
            self.client.close()


# Example usage
if __name__ == "__main__":
    question = "Does immunotherapy improve survival in NSCLC?"
    analyzer = GraphRAGSystem(question)
    print("Analyzing...")
    summary = analyzer.analyze()
    print("\nSummary of relevant relationships:\n", summary)