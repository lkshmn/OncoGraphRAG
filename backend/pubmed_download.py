import time
import json
import os
from Bio import Entrez
from pathlib import Path

# NCBI requires an email address for their API
Entrez.email = "lakshman12102003@gmail.com" 

def fetch_pubmed_findings(cancer_types, studies_per_type=500, output_folder="data/abstracts"):
    """
    Fetches published medical abstracts from PubMed.
    Now filters specifically for Phased Clinical Trials to ensure high-quality metadata.
    Saves each cancer type to a separate JSONL file in the specified folder.
    """
    
    # Ensure output directory exists
    folder_path = Path(output_folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    print(f"Targeting {studies_per_type} high-quality trials for {len(cancer_types)} cancer types...\n")

    for cancer in cancer_types:
        print(f"--- Fetching published findings for: {cancer} ---")
        
        # Define output file path for this specific cancer type
        safe_filename = cancer.replace(" ", "_").replace("/", "-") + ".jsonl"
        output_file = folder_path / safe_filename
        
        # Clear existing file if needed (optional, currently overwrites/starts fresh per run)
        with open(output_file, 'w', encoding='utf-8') as f:
            pass

        # IMPROVED SEARCH FILTER:
        # 1. Explicitly asks for Phase 1-4 trials OR Randomized Controlled Trials
        # 2. Excludes Reviews and Protocols to get actual results
        search_query = (
            f'"{cancer}"[Title/Abstract] AND '
            '("Clinical Trial, Phase III"[Publication Type] OR '
            '"Clinical Trial, Phase II"[Publication Type] OR '
            '"Clinical Trial, Phase I"[Publication Type] OR '
            '"Clinical Trial, Phase IV"[Publication Type] OR '
            '"Randomized Controlled Trial"[Publication Type])'
        )
        
        try:
            # 1. Search PubMed to get the IDs (PMIDs)
            search_handle = Entrez.esearch(db="pubmed", term=search_query, retmax=studies_per_type, sort="relevance")
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            id_list = search_results["IdList"]
            print(f"Found {len(id_list)} publications. Downloading abstracts...")
            
            if not id_list:
                continue
            
            # 2. Fetch the actual text (Batched by 100 to avoid API timeouts)
            for i in range(0, len(id_list), 100):
                batch_ids = id_list[i:i+100]
                fetch_handle = Entrez.efetch(db="pubmed", id=batch_ids, retmode="xml")
                papers = Entrez.read(fetch_handle)
                fetch_handle.close()
                
                new_records = []
                
                # 3. Parse the XML response
                # Note: 'PubmedArticle' might not exist if only 'PubmedBookArticle' returned, using .get()
                for paper in papers.get('PubmedArticle', []):
                    medline = paper['MedlineCitation']
                    article = medline['Article']
                    
                    pmid = str(medline['PMID'])
                    title = article.get('ArticleTitle', '')

                    # --- ENHANCED: Extract Trial Phase from Metadata ---
                    pub_type_list = article.get('PublicationTypeList', [])
                    # Flatten the list of strings (PubType can be a string or StringElement)
                    pub_types = [str(pt).lower() for pt in pub_type_list]
                    
                    trial_phase = "Unknown"
                    if any("phase 3" in pt or "phase iii" in pt for pt in pub_types):
                        trial_phase = "Phase III"
                    elif any("phase 2" in pt or "phase ii" in pt for pt in pub_types):
                        trial_phase = "Phase II"
                    elif any("phase 1" in pt or "phase i" in pt for pt in pub_types):
                        trial_phase = "Phase I"
                    elif any("phase 4" in pt or "phase iv" in pt for pt in pub_types):
                        trial_phase = "Phase IV"
                    elif "randomized controlled trial" in pub_types:
                        trial_phase = "Randomized Controlled Trial" 
                    # ----------------------------------------------

                    # Extract the Abstract
                    abstract_obj = article.get('Abstract', {}).get('AbstractText', [])
                    if not abstract_obj:
                        continue 
                    
                    # Handle different abstract formats (sometimes list of strings, sometimes list of objects)
                    abstract_text = ""
                    if isinstance(abstract_obj, list):
                        parts = []
                        for part in abstract_obj:
                            if hasattr(part, 'attributes') and 'Label' in part.attributes:
                                parts.append(f"{part.attributes['Label']}: {part}")
                            else:
                                parts.append(str(part))
                        abstract_text = " ".join(parts)
                    else:
                        abstract_text = str(abstract_obj)
                    
                    record = {
                        "search_category": cancer,
                        "id": f"PMID:{pmid}", 
                        "trial_phase_metadata": trial_phase,
                        "title": title,
                        "text": abstract_text 
                    }
                    new_records.append(record)

                # Append batch to file immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    for record in new_records:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

                # Rate limiting
                time.sleep(1) 
            
            print(f"  -> Saved records to {output_file}")
                
        except Exception as e:
            print(f"❌ Error fetching {cancer}: {e}")
            
    print(f"\n🎉 Download Complete. Check '{output_folder}' folder.")

if __name__ == "__main__":
    cancer_categories = [
        "Breast Cancer",
        "Lung Cancer",
        "Colorectal Cancer",
        "Prostate Cancer",
        "Melanoma",
        "Oral Cancer",
        "Leukemia",
        "Cervical Cancer"
    ]
    
    # Run the fetcher
    # Use absolute path or relative path from project root
    output_dir = r"D:\Projects\OncoGraphRAG\data\abstracts"
    
    fetch_pubmed_findings(cancer_categories, studies_per_type=500, output_folder=output_dir)