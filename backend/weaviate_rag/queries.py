ONCOLOGY_ONTOLOGY_URI = "http://www.semanticweb.org/lkshmn/ontologies/2026/1/OncoGraphRAG#"

SPARQL_QUERY_ENTITIES = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ont: <{ONCOLOGY_ONTOLOGY_URI}>

SELECT DISTINCT ?entity ?label ?comment ?entityType
WHERE {{
  ?entity a ?entityType .
  
  FILTER NOT EXISTS {{
    ?subClass rdfs:subClassOf+ ?entityType .
    ?entity a ?subClass .
  }}
  
  # Ensure we capture things that are subclasses of the top-level concepts
  # Adjust 'Entity' if your top-level class is named differently (e.g., owl:Thing or specific root)
  # For now, fetching everything that has valid types in our domain
  ?entityType rdfs:subClassOf* ont:Entity .

  OPTIONAL {{ ?entity rdfs:label ?label }}
  OPTIONAL {{ ?entity rdfs:comment ?comment }}
}}
"""

SPARQL_QUERY_PUBLICATIONS = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ont: <{ONCOLOGY_ONTOLOGY_URI}>
SELECT ?pub ?label ?comment
WHERE {{
  ?pub a ont:PUBLICATION .
  OPTIONAL {{ ?pub rdfs:label ?label . }}
  OPTIONAL {{ ?pub rdfs:comment ?comment . }}
}}
"""

SPARQL_QUERY_RELATIONS = f"""
PREFIX ont: <{ONCOLOGY_ONTOLOGY_URI}>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX prov: <http://www.w3.org/ns/prov#>

SELECT ?subj ?pred ?obj ?publication ?subjLabel ?objLabel ?relLabel ?relComment
WHERE {{
  {{
    SELECT ?subj ?pred ?obj ?publication
    WHERE {{
      VALUES ?pred {{ 
        ont:cause ont:affect ont:prevent ont:aggravate ont:diagnose ont:improve 
        ont:present ont:progression ont:test ont:treat
      }}
      ?subj ?pred ?obj .
      OPTIONAL {{ ?subj prov:wasDerivedFrom ?publication . }}
    }}
  }}
  UNION
  {{
    ?rel a ont:RELATION .
    OPTIONAL {{ ?rel ont:relation_subject   ?subj . }}
    OPTIONAL {{ ?rel ont:relation_predicate ?pred . }}
    OPTIONAL {{ ?rel ont:relation_object    ?obj . }}
    OPTIONAL {{ ?rel prov:wasDerivedFrom ?publication . }}
  }}

  OPTIONAL {{ ?subj rdfs:label ?subjLabel . }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabel . }}
  OPTIONAL {{
    ?pred rdfs:label ?relLabel ;
          rdfs:comment ?relComment .
  }}
}}
"""