import os
import json
import chromadb
from ollama_embeddings import get_ollama_embeddings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_json_chunks(resume_data):
    """Convert structured JSON resume data into optimized chunks for RAG"""
    chunks = []

    personal = resume_data.get('personal_info', {})
    name = personal.get('name', 'Unknown')

    # 1. Personal Information Chunk
    if personal.get('name'):
        personal_info = []
        if personal.get('name'):
            personal_info.append(f"Name: {personal['name']}")
        if personal.get('title'):
            personal_info.append(f"Title: {personal['title']}")
        if personal.get('email'):
            personal_info.append(f"Email: {personal['email']}")
        if personal.get('location'):
            personal_info.append(f"Location: {personal['location']}")
        if personal.get('website'):
            personal_info.append(f"Website: {personal['website']}")
        if personal.get('linkedin'):
            personal_info.append(f"LinkedIn: {personal['linkedin']}")
        if personal.get('github'):
            personal_info.append(f"GitHub: {personal['github']}")

        personal_chunk = {
            'text': f"Personal Information:\n" + "\n".join(personal_info) + f"\n\n{name} is an experienced software engineer based in {personal.get('location', 'Australia')}.",
            'metadata': {'section': 'personal', 'type': 'contact_info', 'person': name, 'source': 'personal_info'}
        }
        chunks.append(personal_chunk)

    # 2. Professional Summary Chunk
    if resume_data.get('professional_summary'):
        summary = resume_data['professional_summary']
        summary_text = f"Professional Summary:\nTitle: {summary.get('title', '')}\n\n{summary.get('description', '')}"

        summary_chunk = {
            'text': summary_text,
            'metadata': {'section': 'summary', 'type': 'overview', 'person': name, 'source': 'professional_summary'}
        }
        chunks.append(summary_chunk)

    # 3. Work Experience Chunks (one per company)
    work_experience = resume_data.get('work_experience', [])
    for exp in work_experience:
        company_name = exp.get('company', 'Unknown Company')
        position = exp.get('position', 'Unknown Position')
        duration = exp.get('duration', 'Unknown Duration')
        location = exp.get('location', '')
        company_desc = exp.get('company_description', '')

        # Build experience text (make it more searchable with company keywords)
        experience_text = f"Work Experience and Employment History:\nCompany: {company_name}\nEmployer: {company_name}\nPosition: {position}\nJob Title: {position}\nDuration: {duration}\nEmployee: {name}"
        if location:
            experience_text += f"\nLocation: {location}"
        if company_desc:
            experience_text += f"\n\nCompany Description: {company_desc}"

        # Add current employment indicators
        is_current = "Present" in duration or "present" in duration.lower()
        if is_current:
            experience_text += f"\n\nCURRENT EMPLOYMENT STATUS: {name} is currently working at {company_name} as a {position}. This is his current job and present employer."

        # Add achievements
        achievements = exp.get('achievements', [])
        if achievements:
            experience_text += "\n\nKey Achievements and Responsibilities:"
            for achievement in achievements:
                experience_text += f"\n• {achievement}"

        # Add technologies
        technologies = exp.get('technologies', [])
        if technologies:
            experience_text += f"\n\nTechnologies Used: {', '.join(technologies)}"

        if is_current:
            experience_text += f"\n\n{name} is currently employed at {company_name} company as a {position} since {duration.split(' - ')[0]}. This is his current position and present job."
        else:
            experience_text += f"\n\n{name} worked at {company_name} company as a {position} {duration}. This employment experience shows {name} has professional work experience at {company_name}."

        company_chunk = {
            'text': experience_text,
            'metadata': {
                'section': 'experience',
                'type': 'company',
                'company_name': company_name,
                'position': position,
                'duration': duration,
                'person': name,
                'source': 'work_experience'
            }
        }
        chunks.append(company_chunk)

    # 4. Education Chunks
    education = resume_data.get('education', [])
    for edu in education:
        institution = edu.get('institution', 'Unknown Institution')
        degree = edu.get('degree', 'Unknown Degree')
        duration = edu.get('duration', 'Unknown Duration')
        location = edu.get('location', '')
        major = edu.get('major', '')
        details = edu.get('details', '')

        education_text = f"Education Background:\nInstitution: {institution}\nDegree: {degree}"
        if major:
            education_text += f"\nMajor: {major}"
        education_text += f"\nDuration: {duration}"
        if location:
            education_text += f"\nLocation: {location}"
        if details:
            education_text += f"\n\nDetails: {details}"

        education_text += f"\n\n{name} studied at {institution} and earned a {degree}."

        education_chunk = {
            'text': education_text,
            'metadata': {
                'section': 'education',
                'type': 'academic',
                'institution': institution,
                'degree': degree,
                'duration': duration,
                'person': name,
                'source': 'education'
            }
        }
        chunks.append(education_chunk)

    # 5. Technical Skills Chunks (organized by category)
    technical_skills = resume_data.get('technical_skills', {})
    for category_name, category_data in technical_skills.items():
        if isinstance(category_data, list):
            for skill_group in category_data:
                if isinstance(skill_group, dict):
                    group_name = skill_group.get('category', category_name)
                    skills = skill_group.get('skills', [])

                    if skills:
                        skills_text = f"Technical Skills - {group_name}:\n{name} is proficient in: {', '.join(skills)}"

                        skills_chunk = {
                            'text': skills_text,
                            'metadata': {
                                'section': 'skills',
                                'type': 'technical_skills',
                                'category': group_name,
                                'person': name,
                                'source': 'technical_skills'
                            }
                        }
                        chunks.append(skills_chunk)

    # 6. Projects Chunks
    projects = resume_data.get('projects', [])
    for project in projects:
        project_name = project.get('name', 'Unknown Project')
        duration = project.get('duration', 'Unknown Duration')
        project_type = project.get('type', 'Project')
        description = project.get('description', '')

        project_text = f"Project: {project_name}\nType: {project_type}\nDuration: {duration}\n\nDescription: {description}"

        # Add key features if available
        key_features = project.get('key_features', [])
        if key_features:
            project_text += "\n\nKey Features:"
            for feature in key_features:
                project_text += f"\n• {feature}"

        # Add technologies
        technologies = project.get('technologies', [])
        if technologies:
            project_text += f"\n\nTechnologies: {', '.join(technologies)}"

        # Add achievement/impact
        achievement = project.get('achievement', '')
        impact = project.get('impact', '')
        if achievement:
            project_text += f"\n\nAchievement: {achievement}"
        if impact:
            project_text += f"\n\nImpact: {impact}"

        project_chunk = {
            'text': project_text,
            'metadata': {
                'section': 'projects',
                'type': 'project',
                'project_name': project_name,
                'project_type': project_type,
                'person': name,
                'source': 'projects'
            }
        }
        chunks.append(project_chunk)

    # 7. Certifications Chunk
    certifications = resume_data.get('certifications', [])
    if certifications:
        cert_text = f"Certifications:\n{name} holds the following certifications:\n"
        for cert in certifications:
            cert_name = cert.get('name', 'Unknown Certification')
            issuer = cert.get('issuer', 'Unknown Issuer')
            validity = cert.get('validity', '')
            status = cert.get('status', '')
            description = cert.get('description', '')
            credlyUrl = cert.get('credlyUrl', '')

            cert_text += f"\n• {cert_name} - {issuer}"
            if validity:
                cert_text += f" (Valid: {validity})"
            if status:
                cert_text += f" - Status: {status}"
            if description:
                cert_text += f"\n  {description}"
            if credlyUrl:
                cert_text += f"\n Credly Badge URL: {credlyUrl}"

        cert_chunk = {
            'text': cert_text,
            'metadata': {'section': 'certifications', 'type': 'credentials', 'person': name, 'source': 'certifications'}
        }
        chunks.append(cert_chunk)

    # 8. Interests Chunk
    interests = resume_data.get('interests', [])
    if interests:
        interests_text = f"Personal Interests:\n{name} is interested in: {', '.join(interests)}"
        interests_chunk = {
            'text': interests_text,
            'metadata': {'section': 'interests', 'type': 'hobbies', 'person': name, 'source': 'interests'}
        }
        chunks.append(interests_chunk)

    return chunks

def ingest_json_resume(json_path: str, chroma_db_path: str = "./data/chroma_db"):
    """Ingest resume from structured JSON file"""

    # Load JSON data
    logger.info(f"Loading resume data from: {json_path}")
    try:
        with open(json_path, 'r') as f:
            resume_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return False

    person_name = resume_data.get('personal_info', {}).get('name', 'Unknown')
    logger.info(f"Processing resume for: {person_name}")

    # Create optimized chunks
    chunks = create_json_chunks(resume_data)
    logger.info(f"Created {len(chunks)} optimized chunks")

    # Display chunk summary
    for i, chunk in enumerate(chunks):
        section = chunk['metadata'].get('section', 'unknown')
        chunk_type = chunk['metadata'].get('type', 'unknown')
        preview = chunk['text'][:100].replace('\n', ' ')
        logger.info(f"  Chunk {i+1}: {section}/{chunk_type} - {preview}...")

    # Initialize Ollama embedding model
    logger.info("Initializing Ollama bge-m3 embeddings...")
    embedding_model = get_ollama_embeddings("bge-m3:latest")

    # Test connection
    if not embedding_model.test_connection():
        logger.error("Failed to connect to Ollama bge-m3 model. Make sure Ollama is running and bge-m3:latest is available.")
        return False

    # Initialize ChromaDB (match universal_ingest.py exactly)
    os.makedirs(chroma_db_path, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)

    # Create or reset collection (match universal_ingest.py exactly)
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
    ids = [f"json_chunk_{i}" for i in range(len(chunks))]

    # Generate embeddings using Ollama bge-m3
    logger.info(f"Generating embeddings for {len(documents)} documents using bge-m3...")
    embeddings = embedding_model.encode(documents)

    # Store in ChromaDB
    collection.add(
        embeddings=embeddings,  # Ollama embeddings are already lists
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    logger.info(f"Successfully ingested {len(chunks)} JSON chunks into ChromaDB")
    return True

def main():
    """Main function to ingest JSON resume"""
    json_path = "nirwan_resume_data.json"

    if not os.path.exists(json_path):
        logger.error(f"JSON file not found: {json_path}")
        return

    success = ingest_json_resume(json_path)

    if success:
        print("✅ JSON resume ingestion completed successfully!")

        # Test retrieval with various queries
        chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = chroma_client.get_collection("resume_knowledge")

        print(f"\n📊 Collection stats: {collection.count()} documents")

        # Test queries with Ollama embeddings
        embedding_model = get_ollama_embeddings("bge-m3:latest")

        test_queries = [
            "What companies has Nirwan worked for?",
            "What are Nirwan's technical skills?",
            "What is Nirwan's contact information?",
            "Tell me about Nirwan's education background",
            "What certifications does Nirwan have?",
            "What projects has Nirwan worked on?"
        ]

        for query in test_queries:
            print(f"\n🔍 Test query: '{query}'")
            query_embedding = embedding_model.encode([query])

            results = collection.query(
                query_embeddings=query_embedding,  # Already a list from Ollama
                n_results=2
            )

            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                section = meta.get('section', 'unknown')
                chunk_type = meta.get('type', 'unknown')
                source = meta.get('source', 'unknown')
                preview = doc[:150].replace('\n', ' ')
                print(f"  Result {i+1} ({section}/{chunk_type} from {source}): {preview}...")

    else:
        print("❌ JSON resume ingestion failed!")

if __name__ == "__main__":
    main()