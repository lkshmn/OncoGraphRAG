import time
import json
from Bio import Entrez

# NCBI requires an email address for their APIpip
Entrez.email = "lakshman12102003@gmail.com" 

def fetch_pubmed_findings(cancer_types, studies_per_type=500, output_file="pubmed_cancer_findings.json"):
    """
    Fetches published medical abstracts from PubMed, specifically filtering 
    for Clinical Trials to ensure the text contains actual findings and results.
    """
    all_studies = []
    
    print(f"Targeting {studies_per_type} published trials for {len(cancer_types)} cancer types...\n")

    for cancer in cancer_types:
        print(f"--- Fetching published findings for: {cancer} ---")
        
        # SEARCH FILTER: This restricts results to the specific cancer AND forces the 
        # publication type to be a 'Clinical Trial', guaranteeing it contains findings.
        search_query = f'"{cancer}"[Title/Abstract] AND "Clinical Trial"[Publication Type]'
        
        try:
            # 1. Search PubMed to get the IDs (PMIDs)
            search_handle = Entrez.esearch(db="pubmed", term=search_query, retmax=studies_per_type)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results.get("IdList", [])
            print(f"Found {len(id_list)} publications. Downloading abstracts...")
            
            if not id_list:
                continue
                
            # 2. Fetch the actual text (Batched by 100 to avoid API timeouts)
            for i in range(0, len(id_list), 100):
                batch_ids = id_list[i:i+100]
                fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, retmode="xml")
                papers = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                # 3. Parse the XML response
                for paper in papers.get('PubmedArticle', []):
                    medline = paper['MedlineCitation']
                    article = medline['Article']
                    
                    pmid = str(medline['PMID'])
                    title = article.get('ArticleTitle', '')
                    
                    # Extract the Abstract
                    abstract_parts = article.get('Abstract', {}).get('AbstractText', [])
                    if not abstract_parts:
                        continue # Skip if no abstract is available
                        
                    # PubMed abstracts are often split into sections (Background, Methods, RESULTS)
                    # We join them into a single block of text for your LLM.
                    abstract_text = " ".join([str(part) for part in abstract_parts])
                    
                    all_studies.append({
                        "search_category": cancer,
                        "source_id": f"PMID:{pmid}", # This is your verifiable source ID!
                        "title": title,
                        "abstract": abstract_text
                    })
                    
                # Rate limiting: NCBI requires a slight delay between requests
                time.sleep(1) 
                
        except Exception as e:
            print(f"❌ Error fetching {cancer}: {e}")
            
    # Save the data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_studies, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 Success! Saved {len(all_studies)} abstracts with findings to '{output_file}'")

if __name__ == "__main__":
    cancer_categories = [
        "Non-Small Cell Lung Cancer",
        "Breast Cancer",
        "Colorectal Cancer",
        "Prostate Cancer",
        "Melanoma",
        "Leukemia",
        "Lymphoma",
        "Pancreatic Cancer",
        "Ovarian Cancer",
        "Glioblastoma"
    ]
    
    # Run the fetcher
    fetch_pubmed_findings(cancer_categories, studies_per_type=500)