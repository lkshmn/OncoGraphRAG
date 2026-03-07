import re
import logging
from collections import defaultdict

synonyms_map = {
    # Breast Cancer
    "breast ca": "breast cancer",
    "breast carcinoma": "breast cancer",
    "tnbc": "triple-negative breast cancer",
    "her2+ breast cancer": "her2-positive breast cancer",
    "dcis": "breast cancer",

    # Lung Cancer
    "nsclc": "non-small cell lung cancer",
    "sclc": "small cell lung cancer",
    "lung ca": "lung cancer",
    "lung carcinoma": "lung cancer",
    "bronchogenic carcinoma": "lung cancer",

    # Colorectal Cancer
    "crc": "colorectal cancer",
    "colon cancer": "colorectal cancer",
    "rectal cancer": "colorectal cancer",
    "colon ca": "colorectal cancer",
    
    # Prostate Cancer
    "prostate ca": "prostate cancer",
    "prostate carcinoma": "prostate cancer",
    "pca": "prostate cancer",
    "castration-resistant prostate cancer": "prostate cancer", 
    "crpc": "prostate cancer",

    # Melanoma
    "malignant melanoma": "melanoma",
    "mm": "melanoma",
    "cutaneous melanoma": "melanoma",

    # Oral Cancer
    "oral cavity cancer": "oral cancer",
    "mouth cancer": "oral cancer",
    "oral squamous cell carcinoma": "oral cancer",
    "oscc": "oral cancer",
    "tongue cancer": "oral cancer",

    # Leukemia
    "all": "acute lymphoblastic leukemia",
    "aml": "acute myeloid leukemia",
    "cll": "chronic lymphocytic leukemia",
    "cml": "chronic myeloid leukemia",
    "leukaemia": "leukemia",
    
    # Cervical Cancer
    "cervical ca": "cervical cancer",
    "cervix cancer": "cervical cancer",
    "cc": "cervical cancer",
    "cervical intraepithelial neoplasia": "cervical cancer", # often precursor but grouped for retrieval context

    # General terms
    "ca": "cancer",
    "carcinoma": "cancer",
    "neoplasm": "tumor",
    "metastasis": "metastatic",
}


def get_most_frequent_type(type_counts):
    type_priority = {
        'disease': 12,
        'body_part': 11,
        'treatment': 10,
        'symptom': 9,
        'complication': 8,
        'comorbidity': 7,
        'risk_factor': 6,
        'biomarker': 5,
        'progression': 4,
        'test': 3,
        'gene': 2,
        'prognosis': 1
    }

    if not type_counts:
        return None

    max_count = max(type_counts.values())

    candidates = [type_label for type_label, count in type_counts.items() if count == max_count]

    if len(candidates) == 1:
        return candidates[0]
    else:
        candidates.sort(key=lambda x: type_priority.get(x, 0), reverse=True)
        return candidates[0]


def resolve_entity_types(relations):
    entity_type_counts = defaultdict(lambda: defaultdict(int))

    for rel in relations:
        e1_name = rel['entity1_name'].lower()
        e1_type = rel['entity1_type']
        e2_name = rel['entity2_name'].lower()
        e2_type = rel['entity2_type']

        entity_type_counts[e1_name][e1_type] += 1
        entity_type_counts[e2_name][e2_type] += 1

    resolved_entity_types = {}
    for entity, types in entity_type_counts.items():
        if len(types) > 1:
            resolved_type = get_most_frequent_type(types)
            resolved_entity_types[entity] = resolved_type
            logging.info(f"Resolved entity '{entity}' to type '{resolved_type}' based on counts: {types}")

    for rel in relations:
        e1_name = rel['entity1_name'].lower()
        e2_name = rel['entity2_name'].lower()

        if e1_name in resolved_entity_types:
            old_type = rel['entity1_type']
            rel['entity1_type'] = resolved_entity_types[e1_name]
            logging.debug(f"Updated entity1 '{e1_name}' type from '{old_type}' to '{resolved_entity_types[e1_name]}'")

        if e2_name in resolved_entity_types:
            old_type = rel['entity2_type']
            rel['entity2_type'] = resolved_entity_types[e2_name]
            logging.debug(f"Updated entity2 '{e2_name}' type from '{old_type}' to '{resolved_entity_types[e2_name]}'")

    return relations


def normalize_entity_name(name):
    if not isinstance(name, str):
        return ""

    # Lowercase and trim whitespace
    name = name.lower().strip()

    # Replace multiple spaces with a single space
    name = ' '.join(name.split())

    # Split into words
    words = name.split()
    
    # Apply domain-specific synonyms mapping
    if name in synonyms_map:
        name = synonyms_map[name]

    return name


def refine_relations(relations):
    """
    Processes a list of raw relations to:
    1. Sanitize entity names
    2. Resolve synonyms (simple distinct)
    3. Deduplicate exact relationship matches
    4. PRESERVE Metadata (efficacy, phase, scores)
    """
    
    unique_rels = {}
    refined_list = []

    for rel in relations:
        s_name = sanitize_entity_name(rel.get('entity1_name'))
        o_name = sanitize_entity_name(rel.get('entity2_name'))
        pred = sanitize_entity_name(rel.get('relation_type'))
        
        # Create a unique key for deduplication
        # Now including efficacy/phase in key to distinguish different findings for same entity pair
        # If you want to merge findings, key would just be s_name+pred+o_name
        # Here we treat unique findings (even same entities) as distinct if the evidence differs
        
        # However, GraphDB needs distinct Reified Relation nodes. 
        # We will pass through all valid relations, sanitized.
        
        new_rel = {
            "entity1_name": s_name,
            "entity1_type": rel.get('entity1_type'),
            "relation_type": pred,
            "entity2_name": o_name,
            "entity2_type": rel.get('entity2_type'),
            # Preserving Metadata (Updated to reliability_score)
            "reliability_score": rel.get('reliability_score', 50),
            "trial_phase": rel.get('trial_phase', 'Unknown'),
            "efficacy": rel.get('efficacy', 'Not stated'),
            "pub_id": rel.get('pub_id')
        }
        
        refined_list.append(new_rel)

    return refined_list


def sanitize_entity_name(name):
    """
    Cleans up entity names for URI generation.
    - Replaces spaces with underscores
    - Removes special characters
    - Lowercase
    """
    if not name:
        return "unknown"
    
    # Remove special characters except alphanumeric, hyphen, underscore
    clean = re.sub(r'[^a-zA-Z0-9\-\_ ]', '', str(name))
    clean = clean.strip().replace(" ", "_").lower()
    return clean
