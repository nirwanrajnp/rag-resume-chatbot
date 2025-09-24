import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from universal_parser import UniversalResumeParser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_universal_chunks(parsed_data):
    """Convert universal parser data into optimized chunks for RAG"""
    chunks = []

    personal = parsed_data.get('personal', {})
    name = personal.get('name', 'Unknown')

    # 1. Personal Information Chunk
    if personal.get('name'):
        personal_info = []
        if personal.get('name'):
            personal_info.append(f"Name: {personal['name']}")
        if personal.get('email'):
            personal_info.append(f"Email: {personal['email']}")
        if personal.get('phone'):
            personal_info.append(f"Phone: {personal['phone']}")
        if personal.get('location'):
            personal_info.append(f"Location: {personal['location']}")

        personal_chunk = {
            'text': f"Personal Information:\n" + "\n".join(personal_info) + f"\n\n{name} is available for contact via the above information.",
            'metadata': {'section': 'personal', 'type': 'contact_info', 'person': name}
        }
        chunks.append(personal_chunk)

    # 2. Company Experience Chunks (one per company with details)
    companies = parsed_data.get('experience', {}).get('companies', [])
    for company_data in companies:
        if isinstance(company_data, dict):
            company_name = company_data.get('name', '').strip()
            position = company_data.get('position', 'Software Professional')
            dates = company_data.get('dates', 'Date not specified')

            # Clean company name from extra text
            if '\n' in company_name:
                company_name = company_name.split('\n')[-1].strip()

            if company_name and not any(skip in company_name.lower() for skip in ['experience', 'work', 'achievements']):
                # Create detailed work experience text with dates
                experience_text = f"Work Experience:\nCompany: {company_name}\nPosition: {position}\nDates: {dates}\nEmployee: {name}\n\n{name} worked at {company_name} as a {position}"
                if dates != "Date not specified":
                    experience_text += f" from {dates}"
                experience_text += ", contributing to development projects, technical solutions, and software engineering tasks."

                company_chunk = {
                    'text': experience_text,
                    'metadata': {
                        'section': 'experience',
                        'type': 'company',
                        'company_name': company_name,
                        'position': position,
                        'dates': dates,
                        'person': name
                    }
                }
                chunks.append(company_chunk)

    # 3. Skills Chunk (filtered and organized)
    skills = parsed_data.get('skills', [])
    if skills:
        # Filter out obvious false positives
        filtered_skills = []
        false_positives = [
            'EXPERIENCE', 'EDUCATION', 'PROJECTS', 'PERSONAL', 'CERTIFICATIONS',
            'SKILLS', 'INTERESTS', 'WORK', 'McPhail', 'ACT', 'PWA', 'NFTs', 'SPAs'
        ]

        for skill in skills:
            if (len(skill) > 1 and
                skill not in false_positives and
                not skill.isupper() or len(skill) <= 4):  # Allow short acronyms
                filtered_skills.append(skill)

        if filtered_skills:
            # Group by category for better organization
            skills_text = f"Technical Skills and Expertise:\n{name} is proficient in: {', '.join(filtered_skills[:15])}"  # Limit to top 15
            if len(filtered_skills) > 15:
                skills_text += f" and {len(filtered_skills) - 15} other technologies."

            skills_chunk = {
                'text': skills_text,
                'metadata': {'section': 'skills', 'type': 'technical_skills', 'person': name}
            }
            chunks.append(skills_chunk)

    # 4. Education Chunks
    education = parsed_data.get('education', [])
    for edu in education:
        if isinstance(edu, dict):
            edu_parts = []
            if edu.get('degree'):
                edu_parts.append(f"Degree: {edu['degree']}")

            if edu.get('institution'):
                # Clean institution name
                institution = edu['institution']
                if '\n' in institution:
                    # Take the cleanest part
                    institution_parts = institution.split('\n')
                    for part in institution_parts:
                        if 'university' in part.lower() or 'college' in part.lower():
                            institution = part.strip()
                            break
                    else:
                        institution = institution_parts[-1].strip()
                edu_parts.append(f"Institution: {institution}")

            if edu.get('dates'):
                edu_parts.append(f"Dates: {edu['dates']}")

            if edu.get('location'):
                edu_parts.append(f"Location: {edu['location']}")

            if edu_parts:
                education_text = f"Education Background:\n" + "\n".join(edu_parts) + f"\n\n{name} studied at {edu.get('institution', 'this institution')}"
                if edu.get('dates'):
                    education_text += f" from {edu['dates']}"
                education_text += f" and earned a {edu.get('degree', 'degree')}."

                education_chunk = {
                    'text': education_text,
                    'metadata': {
                        'section': 'education',
                        'type': 'academic',
                        'institution': edu.get('institution', ''),
                        'degree': edu.get('degree', ''),
                        'dates': edu.get('dates', ''),
                        'person': name
                    }
                }
                chunks.append(education_chunk)

    # 5. Certifications Chunk (filtered)
    certifications = parsed_data.get('certifications', [])
    valid_certs = []
    for cert in certifications:
        # Filter out section headers and false positives
        if (len(cert) > 10 and
            not cert.isupper() and
            'certified' in cert.lower() or 'certificate' in cert.lower()):
            valid_certs.append(cert)

    if valid_certs:
        cert_text = f"Certifications:\n{name} holds the following certifications:\n" + "\n".join([f"• {cert}" for cert in valid_certs])
        cert_chunk = {
            'text': cert_text,
            'metadata': {'section': 'certifications', 'type': 'credentials', 'person': name}
        }
        chunks.append(cert_chunk)

    # 6. References Chunk
    references = parsed_data.get('references', [])
    if references:
        ref_parts = []
        for ref in references:
            if isinstance(ref, dict) and ref.get('name'):
                ref_info = f"• {ref['name']}"
                if ref.get('phone'):
                    ref_info += f" - Phone: {ref['phone']}"
                if ref.get('email'):
                    ref_info += f" - Email: {ref['email']}"
                ref_parts.append(ref_info)

        if ref_parts:
            ref_text = f"Professional References:\n{name} has provided the following professional references:\n" + "\n".join(ref_parts)
            ref_chunk = {
                'text': ref_text,
                'metadata': {'section': 'references', 'type': 'contacts', 'person': name}
            }
            chunks.append(ref_chunk)

    # 7. Interests Chunk
    interests = parsed_data.get('interests', [])
    if interests:
        interests_text = f"Personal Interests:\n{name} enjoys: {', '.join(interests)}"
        interests_chunk = {
            'text': interests_text,
            'metadata': {'section': 'interests', 'type': 'hobbies', 'person': name}
        }
        chunks.append(interests_chunk)

    return chunks

def ingest_universal_resume(pdf_path: str, chroma_db_path: str = "./data/chroma_db"):
    """Ingest resume using universal parser"""

    # Parse resume with universal parser
    logger.info(f"Parsing resume with Universal Parser: {pdf_path}")
    parser = UniversalResumeParser()
    parsed_data = parser.parse_resume(pdf_path)

    if not parsed_data:
        logger.error("Failed to parse resume with Universal Parser")
        return False

    person_name = parsed_data.get('personal', {}).get('name', 'Unknown')
    logger.info(f"Successfully parsed resume for: {person_name}")

    # Create optimized chunks
    chunks = create_universal_chunks(parsed_data)
    logger.info(f"Created {len(chunks)} optimized chunks")

    # Display chunk summary
    for i, chunk in enumerate(chunks):
        section = chunk['metadata'].get('section', 'unknown')
        chunk_type = chunk['metadata'].get('type', 'unknown')
        preview = chunk['text'][:100].replace('\n', ' ')
        logger.info(f"  Chunk {i+1}: {section}/{chunk_type} - {preview}...")

    # Initialize embedding model
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize ChromaDB
    os.makedirs(chroma_db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    # Create or reset collection
    try:
        collection = chroma_client.delete_collection("resume_knowledge")
        logger.info("Deleted existing collection")
    except:
        pass

    collection = chroma_client.create_collection("resume_knowledge")
    logger.info("Created new collection")

    # Generate embeddings and store
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = [f"universal_chunk_{i}" for i in range(len(chunks))]

    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = embedding_model.encode(documents)

    # Store in ChromaDB
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Successfully ingested {len(chunks)} universal chunks into ChromaDB")

    # Save parsed data for debugging
    debug_file = f"{chroma_db_path}/universal_parsed_resume.json"
    with open(debug_file, 'w') as f:
        json.dump(parsed_data, f, indent=2, default=str)

    logger.info(f"Saved parsed resume data to: {debug_file}")

    return True

def main():
    """Main function to test universal ingestion"""
    pdf_path = "Nirwan-Resume-1.pdf"

    if not os.path.exists(pdf_path):
        logger.error(f"Resume file not found: {pdf_path}")
        return

    success = ingest_universal_resume(pdf_path)

    if success:
        print("✅ Universal resume ingestion completed successfully!")

        # Test retrieval with various queries
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = chroma_client.get_collection("resume_knowledge")

        print(f"\n📊 Collection stats: {collection.count()} documents")

        # Test queries
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        test_queries = [
            "What companies has Nirwan worked for?",
            "What are Nirwan's technical skills?",
            "What is Nirwan's contact information?",
            "Tell me about Nirwan's education background"
        ]

        for query in test_queries:
            print(f"\n🔍 Test query: '{query}'")
            query_embedding = embedding_model.encode([query])

            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=2
            )

            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                section = meta.get('section', 'unknown')
                chunk_type = meta.get('type', 'unknown')
                preview = doc[:150].replace('\n', ' ')
                print(f"  Result {i+1} ({section}/{chunk_type}): {preview}...")

    else:
        print("❌ Universal resume ingestion failed!")

if __name__ == "__main__":
    main()