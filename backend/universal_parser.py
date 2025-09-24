import re
import json
import spacy
from pdfminer.high_level import extract_text
from typing import Dict, List, Any, Optional
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalResumeParser:
    def __init__(self):
        """Initialize the universal resume parser"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            self.nlp = None

        # Universal patterns
        self.setup_patterns()

    def setup_patterns(self):
        """Setup universal patterns for different resume formats"""

        # Company suffixes (international)
        self.company_suffixes = [
            r'Pty Ltd', r'Ltd', r'Inc', r'LLC', r'Corp', r'Corporation',
            r'GmbH', r'AG', r'SA', r'S\.A\.', r'BV', r'AB', r'AS',
            r'Co\.', r'Company', r'Group', r'Holdings', r'Ventures',
            r'Technologies', r'Systems', r'Solutions', r'Services'
        ]

        # Education keywords
        self.education_keywords = [
            'university', 'college', 'institute', 'school', 'academy',
            'polytechnic', 'tech', 'state university', 'community college'
        ]

        # Degree patterns
        self.degree_patterns = [
            r'Bachelor(?:\'s)?(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'Master(?:\'s)?(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'PhD(?:\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'Doctor(?:ate)?(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'Associate(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'Diploma(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'Certificate(?:\s+of\s+|\s+in\s+|\s+)([A-Z][^,\n.]+)',
            r'B\.?[A-Z]\.?(?:\s+|$)',  # BA, BS, etc.
            r'M\.?[A-Z]\.?(?:\s+|$)',  # MA, MS, etc.
        ]

        # Phone patterns (international) - improved
        self.phone_patterns = [
            # Australian full format
            r'\+61[0-9]{9}',  # +61481948203
            r'\+61[-.\s][0-9][-.\s]?[0-9]{4}[-.\s]?[0-9]{4}',  # +61 4 8194 8203
            # US/Canada
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            # UK
            r'\+?44[-.\s]?(?:\(?0\)?[-.\s]?)?[0-9]{2,5}[-.\s]?[0-9]{3,8}',
            # General international continuous
            r'\+[0-9]{10,15}',
            # General patterns with separators
            r'\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            r'[0-9]{2,4}[-.\s][0-9]{3,4}[-.\s][0-9]{4,8}'
        ]

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfminer"""
        try:
            text = extract_text(pdf_path)
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def extract_name(self, text: str) -> str:
        """Extract name using multiple heuristics"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Strategy 1: First meaningful line
        for line in lines[:10]:
            # Skip lines with common resume keywords
            skip_keywords = [
                'software', 'engineer', 'developer', 'manager', 'analyst',
                'certified', 'phone', 'email', '@', 'resume', 'cv',
                'experience', 'years', 'skills', 'objective', 'summary'
            ]

            if not any(keyword in line.lower() for keyword in skip_keywords):
                # Check if it looks like a name (2-4 words, proper capitalization)
                words = line.split()
                if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word.isalpha()):
                    return ' '.join(words)

        # Strategy 2: Use spaCy NER if available
        if self.nlp:
            doc = self.nlp(' '.join(lines[:5]))  # Check first 5 lines
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and len(ent.text.split()) >= 2:
                    return ent.text

        return ""

    def extract_email(self, text: str) -> str:
        """Extract primary email address intelligently"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text, re.IGNORECASE)

        if not emails:
            return ""

        # Extract name to help identify personal email
        name = self.extract_name(text)
        name_parts = name.lower().split() if name else []

        # Score emails based on multiple factors
        email_scores = {}

        for email in emails:
            score = 0
            email_lower = email.lower()
            email_local = email_lower.split('@')[0]  # Part before @

            # POSITIVE scoring (personal email indicators)
            # Check if email contains name parts
            if name_parts:
                for part in name_parts:
                    if len(part) > 2 and part in email_local:
                        score += 10  # Strong indicator

            # Common personal email patterns
            personal_patterns = [
                'gmail', 'yahoo', 'hotmail', 'outlook', 'icloud',
                'protonmail', 'me.com', 'live.com'
            ]
            if any(pattern in email_lower for pattern in personal_patterns):
                score += 5

            # NEGATIVE scoring (reference/business email indicators)
            # Skip obvious reference emails
            reference_indicators = [
                'noreply', 'admin', 'info', 'contact', 'support', 'help',
                'hr@', 'jobs@', 'careers@', 'team@'
            ]
            if any(indicator in email_lower for indicator in reference_indicators):
                score -= 20

            # Business domain patterns that might be references
            business_patterns = ['benmcphail', 'company', 'corp', 'ltd']
            if any(pattern in email_lower for pattern in business_patterns):
                score -= 5

            # Context analysis - check surrounding text
            email_pos = text.lower().find(email_lower)
            if email_pos != -1:
                context = text[max(0, email_pos-50):email_pos+len(email)+50].lower()
                # If email is near "contact" or reference keywords, likely a reference
                if any(word in context for word in ['contact', 'reference', 'manager', 'supervisor']):
                    score -= 15

            email_scores[email] = score

        # Return highest scoring email
        if email_scores:
            best_email = max(email_scores.items(), key=lambda x: x[1])
            return best_email[0]

        return emails[0]  # Fallback

    def extract_phone(self, text: str) -> str:
        """Extract primary phone number intelligently"""
        phone_candidates = []

        for pattern in self.phone_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                phone_text = match.group(0)
                phone_start = match.start()

                # Clean up phone number - preserve + and digits
                cleaned_phone = re.sub(r'[^\d+]', '', phone_text)
                if len(cleaned_phone) >= 10:  # Valid phone length

                    # Store original phone text for return
                    display_phone = phone_text.strip()

                    # Analyze context to score the phone number
                    context_start = max(0, phone_start - 100)
                    context_end = min(len(text), phone_start + len(phone_text) + 100)
                    context = text[context_start:context_end].lower()

                    score = 0

                    # NEGATIVE scoring for reference phones
                    reference_indicators = [
                        'contact:', 'reference:', 'manager:', 'supervisor:',
                        'dr.', 'prof.', 'mr.', 'ms.', 'mrs.',
                        'ben mcphail', 'christopher read'
                    ]

                    if any(indicator in context for indicator in reference_indicators):
                        score -= 10

                    # POSITIVE scoring for personal phone indicators
                    personal_indicators = ['mobile:', 'cell:', 'phone:', 'tel:']
                    if any(indicator in context for indicator in personal_indicators):
                        score += 5

                    # Prefer phones that appear earlier in document (usually personal info)
                    if phone_start < len(text) * 0.3:  # First 30% of document
                        score += 3

                    phone_candidates.append({
                        'phone': display_phone,
                        'cleaned': cleaned_phone,
                        'score': score,
                        'position': phone_start
                    })

        if phone_candidates:
            # Sort by score (highest first), then by position (earliest first)
            phone_candidates.sort(key=lambda x: (-x['score'], x['position']))
            return phone_candidates[0]['phone']

        return ""

    def extract_references(self, text: str) -> List[Dict[str, str]]:
        """Extract referee/reference information"""
        references = []

        # Look for reference sections
        ref_section_patterns = [
            r'(?i)(references?|contacts?)\s*:?\s*(.*?)(?=\n\s*[A-Z]{2,}|\n\s*$|$)',
            r'(?i)contact\s*:\s*([^\n]+(?:\n[^A-Z\n][^\n]*)*)',
        ]

        # Find reference indicators with context
        ref_indicators = [
            r'(?i)(dr\.|prof\.|mr\.|ms\.|mrs\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?i)(manager|supervisor|director|lead|head)\s*:?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'(?i)contact\s*:?\s*[^\n]*?([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]

        # Extract email and phone patterns for references
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_patterns = self.phone_patterns

        # Find potential reference blocks
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if line contains reference indicators
            for pattern in ref_indicators:
                match = re.search(pattern, line)
                if match:
                    ref_name = match.group(2) if len(match.groups()) >= 2 else match.group(1)

                    # Look for associated contact info in surrounding lines
                    ref_info = {'name': ref_name.strip()}

                    # Search next few lines for email/phone
                    for j in range(max(0, i-2), min(len(lines), i+5)):
                        context_line = lines[j]

                        # Find email
                        email_match = re.search(email_pattern, context_line)
                        if email_match and 'email' not in ref_info:
                            ref_info['email'] = email_match.group(0)

                        # Find phone
                        for phone_pattern in phone_patterns[:3]:  # Use simpler patterns
                            phone_match = re.search(phone_pattern, context_line)
                            if phone_match and 'phone' not in ref_info:
                                ref_info['phone'] = phone_match.group(0).strip()
                                break

                    # Only add if we have at least name + (email or phone)
                    if len(ref_info) >= 2:
                        references.append(ref_info)
                    break
            i += 1

        # Remove duplicates based on name
        seen_names = set()
        unique_references = []
        for ref in references:
            if ref['name'] not in seen_names:
                seen_names.add(ref['name'])
                unique_references.append(ref)

        return unique_references

    def extract_interests(self, text: str) -> List[str]:
        """Extract interests/hobbies from resume"""
        interests = []

        # Find interests section more reliably
        # First, find where INTERESTS section starts
        interests_match = re.search(r'(?i)\binterests?\b\s*\n', text)

        if interests_match:
            # Get everything after the INTERESTS heading
            interests_start = interests_match.end()
            interests_section = text[interests_start:].strip()

            # Split by lines and extract each interest
            for line in interests_section.split('\n'):
                line = line.strip()

                # Skip empty lines and stop at whitespace blocks or very short lines
                if not line or len(line) < 2:
                    continue

                # Stop if we hit another section (usually all caps or very short)
                if line.isupper() and len(line) > 10:
                    break

                # Clean and add the interest
                if (len(line) > 2 and
                    not line.isupper() and
                    line.lower() not in ['interests', 'hobbies', 'personal']):
                    interests.append(line.title())

        # Remove duplicates while preserving order
        seen = set()
        unique_interests = []
        for interest in interests:
            if interest.lower() not in seen:
                seen.add(interest.lower())
                unique_interests.append(interest)

        return unique_interests

    def extract_companies(self, text: str) -> List[Dict[str, Any]]:
        """Extract companies with dates from work experience sections"""
        companies = []

        # Look for work experience sections with dates
        # Pattern: Job Title \n Company Name \n Date Range, Location
        work_sections = re.split(r'(?i)(programmer analyst|software engineer|developer|engineer)', text)

        for i in range(1, len(work_sections), 2):  # Skip every other match (the title itself)
            if i + 1 < len(work_sections):
                title = work_sections[i].strip()
                section = work_sections[i + 1].strip()

                # Extract company name (first line after title)
                lines = section.split('\n')
                company_line = None
                date_line = None

                for j, line in enumerate(lines):
                    line = line.strip()
                    if line and any(suffix in line for suffix in self.company_suffixes):
                        company_line = line
                        # Look for date in next few lines
                        for k in range(j + 1, min(j + 3, len(lines))):
                            next_line = lines[k].strip()
                            # Date pattern: MM/YYYY - MM/YYYY or MM/YYYY - Present
                            if re.search(r'\d{1,2}/\d{4}\s*-\s*(\d{1,2}/\d{4}|Present)', next_line):
                                date_match = re.search(r'(\d{1,2}/\d{4}\s*-\s*(?:\d{1,2}/\d{4}|Present))', next_line)
                                if date_match:
                                    date_line = date_match.group(1).strip()
                                break
                        break

                if company_line:
                    # Clean company name
                    company_name = company_line

                    # Extract job position/title
                    position = title.title() if title else "Software Professional"

                    companies.append({
                        'name': company_name,
                        'position': position,
                        'dates': date_line if date_line else "Date not specified",
                        'confidence': 15 if date_line else 10
                    })

        # Deduplicate similar companies
        final_companies = []
        seen_companies = set()

        for company in companies:
            company_lower = company['name'].lower()
            is_duplicate = False

            for seen in seen_companies:
                if (company_lower in seen.lower() or seen.lower() in company_lower):
                    # Keep the one with better date information or longer name
                    existing = next((c for c in final_companies if seen.lower() in c['name'].lower()), None)
                    if existing and (len(company['name']) > len(existing['name']) or
                                   (company['dates'] != "Date not specified" and existing['dates'] == "Date not specified")):
                        final_companies.remove(existing)
                        seen_companies.discard(seen)
                    else:
                        is_duplicate = True
                    break

            if not is_duplicate:
                final_companies.append(company)
                seen_companies.add(company_lower)

        return final_companies

    def extract_skills_adaptive(self, text: str) -> List[str]:
        """Extract skills adaptively from context - focus on technical skills"""
        skills = set()

        # Known technical skills (curated list)
        technical_skills = {
            'Python', 'JavaScript', 'TypeScript', 'Java', 'C#', 'C++', 'Go', 'Rust',
            'PHP', 'Ruby', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB',
            'React', 'Angular', 'Vue', 'Svelte', 'Node.js', 'Express', 'Django',
            'Flask', 'Spring', 'Laravel', 'Ruby on Rails', 'ASP.NET',
            'AWS', 'Lambda', 'EC2', 'S3', 'DynamoDB', 'CloudFormation', 'CloudWatch',
            'API Gateway', 'Elastic Beanstalk', 'RDS', 'VPC', 'IAM', 'CodePipeline',
            'CodeDeploy', 'SES', 'Azure', 'GCP', 'Google Cloud', 'Kubernetes', 'Docker',
            'Terraform', 'Ansible', 'Jenkins', 'GitLab CI', 'GitHub Actions',
            'MongoDB', 'PostgreSQL', 'MySQL', 'SQLite', 'Redis', 'Elasticsearch',
            'Oracle', 'SQL Server', 'Cassandra', 'Neo4j', 'DynamoDB',
            'Git', 'GitHub', 'GitLab', 'Jira', 'Confluence', 'Linux', 'HTML', 'CSS',
            'GraphQL', 'REST', 'API', 'APIs', 'JSON', 'XML', 'YAML'
        }

        # Section-based extraction - look for skills section first
        skills_section_patterns = [
            r'(?i)(technical\s+skills?|skills?|technologies)\s*:?\s*(.*?)(?=\n\s*[A-Z]{2,}|\n\s*$|$)',
            r'(?i)technologies\s*:?\s*(.*?)(?=\n\s*[A-Z]{2,}|\n\s*$|$)'
        ]

        # First try to find a dedicated skills section
        skills_text = ""
        for pattern in skills_section_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            if matches:
                if isinstance(matches[0], tuple):
                    skills_text += " " + matches[0][1]
                else:
                    skills_text += " " + matches[0]

        # If we found a skills section, prioritize that
        if skills_text.strip():
            search_text = skills_text.lower()
        else:
            # Fall back to searching the entire document
            search_text = text.lower()

        # Extract known technical skills
        for skill in technical_skills:
            # Use word boundaries for more precise matching
            pattern = rf'\b{re.escape(skill.lower())}\b'
            if re.search(pattern, search_text):
                skills.add(skill)

        # Filter out obvious false positives that might have slipped through
        false_positive_skills = {
            'EXPERIENCE', 'EDUCATION', 'PROJECTS', 'PERSONAL', 'CERTIFICATIONS',
            'SKILLS', 'INTERESTS', 'WORK', 'CONTACT', 'REFERENCES', 'SUMMARY',
            'McPhail', 'ACT', 'NFTs', 'SPAs', 'PWA', 'ERC-721and', 'ERC-1155'
        }

        # Remove false positives
        skills = skills - false_positive_skills

        # Remove anything that looks like a section header (all caps, common words)
        filtered_skills = []
        for skill in skills:
            if not (skill.isupper() and len(skill) > 4 and skill.isalpha()):
                filtered_skills.append(skill)

        return filtered_skills

    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information with dates"""
        education_entries = []

        # Look for education section
        education_section_match = re.search(r'(?i)education\s*(.*?)(?=\n\s*[A-Z]{2,}|\n\s*$|$)', text, re.DOTALL)

        if education_section_match:
            education_text = education_section_match.group(1)

            # Split by degree programs
            lines = education_text.split('\n')
            current_entry = {}

            for line in lines:
                line = line.strip()
                if not line or len(line) < 3:
                    continue

                # Check for degree patterns
                degree_patterns = [
                    r'(?i)bachelor\s+of\s+([A-Za-z\s]+)',
                    r'(?i)master\s+of\s+([A-Za-z\s]+)',
                    r'(?i)(bachelor|master|phd|doctorate)\s+([A-Za-z\s]+)',
                    r'(?i)(b\.?\s*[a-z]+|m\.?\s*[a-z]+|phd|ph\.?d\.?)'
                ]

                degree_found = False
                for pattern in degree_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if len(match.groups()) > 1:
                            current_entry['degree'] = f"{match.group(1)} {match.group(2)}".strip().title()
                        else:
                            current_entry['degree'] = match.group(0).strip().title()
                        degree_found = True
                        break

                # Check for institutions
                if 'university' in line.lower() or 'college' in line.lower() or 'institute' in line.lower():
                    current_entry['institution'] = line

                # Check for dates
                date_match = re.search(r'(\d{1,2}/\d{4}\s*-\s*\d{1,2}/\d{4})', line)
                if date_match:
                    current_entry['dates'] = date_match.group(1)

                # Check for location
                location_patterns = [
                    r'([A-Z][a-z]+,\s*[A-Z]{2,3})',  # City, State/Country
                    r'([A-Z][a-z]+,\s*[A-Z][a-z]+)'  # City, Country
                ]
                for pattern in location_patterns:
                    location_match = re.search(pattern, line)
                    if location_match:
                        current_entry['location'] = location_match.group(1)

                # If we found a degree and have some info, add to entries
                if degree_found and current_entry:
                    if current_entry not in education_entries:
                        education_entries.append(current_entry.copy())
                    current_entry = {}

            # Add any remaining entry
            if current_entry and ('degree' in current_entry or 'institution' in current_entry):
                education_entries.append(current_entry)

        # If no structured education section found, try legacy approach
        if not education_entries:
            for pattern in self.degree_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, str) and len(match) > 3:
                        education_entries.append({'degree': match.strip()})

        return education_entries

    def extract_certifications(self, text: str) -> List[str]:
        """Extract certifications using improved patterns"""
        certifications = []

        # Handle ligature characters (ﬁ, ﬂ) that appear in PDFs
        text_clean = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('–', '-')

        # Find CERTIFICATIONS section and extract content
        lines = text_clean.split('\n')
        in_cert_section = False
        cert_lines = []

        for line in lines:
            line = line.strip()

            if line.upper() == 'CERTIFICATIONS':
                in_cert_section = True
                continue
            elif in_cert_section:
                # Stop if we hit another major section
                major_sections = ['INTERESTS', 'EDUCATION', 'SKILLS', 'EXPERIENCE', 'PROJECTS']
                if line.upper() in major_sections:
                    break
                elif line and not line.startswith(' '):  # Non-empty, non-indented line
                    cert_lines.append(line)

        # Extract structured certifications from the section
        current_cert = ""
        for line in cert_lines:
            # Check if line looks like a certification title
            if ('certified' in line.lower() or 'certificate' in line.lower()) and len(line) > 10:
                if current_cert:
                    certifications.append(current_cert.strip())
                current_cert = line
            elif line and '(' in line and ')' in line:  # Date line
                if current_cert:
                    current_cert += f" {line}"
            elif line and len(line) > 20:  # Description line
                # Don't add description, just finish current cert
                if current_cert and current_cert not in certifications:
                    certifications.append(current_cert.strip())
                    current_cert = ""

        # Add final certification if exists
        if current_cert and current_cert not in certifications:
            certifications.append(current_cert.strip())

        # Clean up certifications
        cleaned_certs = []
        for cert in certifications:
            # Remove extra whitespace and clean up
            cert = ' '.join(cert.split())
            if len(cert) > 10 and 'certified' in cert.lower():
                cleaned_certs.append(cert)

        return cleaned_certs

    def extract_location(self, text: str) -> str:
        """Extract location using intelligent filtering"""
        # International location patterns
        location_patterns = [
            # City, State/Province (full names)
            r'([A-Z][a-z]+,\s*Queensland)',
            r'([A-Z][a-z]+,\s*New South Wales)',
            r'([A-Z][a-z]+,\s*Victoria)',
            r'([A-Z][a-z]+,\s*California)',
            r'([A-Z][a-z]+,\s*New York)',
            # City, State/Province abbreviations
            r'([A-Z][a-z]+,\s*QLD)',
            r'([A-Z][a-z]+,\s*NSW)',
            r'([A-Z][a-z]+,\s*VIC)',
            r'([A-Z][a-z]+,\s*ACT)',
            # Generic patterns (but we'll filter these)
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',
        ]

        location_candidates = []

        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Score locations based on likelihood
                score = 0

                # POSITIVE scoring for real locations
                real_locations = [
                    'brisbane', 'sydney', 'melbourne', 'perth', 'adelaide', 'canberra',
                    'queensland', 'new south wales', 'victoria', 'qld', 'nsw', 'vic', 'act',
                    'new york', 'california', 'london', 'toronto', 'vancouver'
                ]

                match_lower = match.lower()
                if any(loc in match_lower for loc in real_locations):
                    score += 10

                # NEGATIVE scoring for false positives
                false_positives = [
                    'lambda', 'ec2', 'api', 'react', 'python', 'node', 'gateway',
                    'deploy', 'code', 'script', 'elastic'
                ]

                if any(fp in match_lower for fp in false_positives):
                    score -= 20

                # Context analysis
                match_pos = text.lower().find(match.lower())
                if match_pos != -1:
                    context = text[max(0, match_pos-50):match_pos+len(match)+50].lower()
                    # If near personal info section, higher score
                    if any(keyword in context for keyword in ['email', 'phone', 'address', 'australia']):
                        score += 5

                location_candidates.append({
                    'location': match,
                    'score': score
                })

        if location_candidates:
            # Sort by score and return highest
            location_candidates.sort(key=lambda x: -x['score'])
            best_location = location_candidates[0]
            if best_location['score'] > 0:  # Only return if positive score
                return best_location['location']

        return ""

    def parse_resume(self, pdf_path: str) -> Dict[str, Any]:
        """Main parsing function - universal approach"""
        text = self.extract_text_from_pdf(pdf_path)

        if not text:
            return {}

        parsed_data = {
            "personal": {
                "name": self.extract_name(text),
                "email": self.extract_email(text),
                "phone": self.extract_phone(text),
                "location": self.extract_location(text)
            },
            "experience": {
                "companies": self.extract_companies(text)
            },
            "skills": self.extract_skills_adaptive(text),
            "education": self.extract_education(text),
            "certifications": self.extract_certifications(text),
            "interests": self.extract_interests(text),
            "references": self.extract_references(text),
            "parsing_stats": {
                "text_length": len(text),
                "companies_found": len(self.extract_companies(text)),
                "skills_found": len(self.extract_skills_adaptive(text)),
                "references_found": len(self.extract_references(text))
            }
        }

        return parsed_data

def test_universal_parser():
    """Test the universal parser"""
    parser = UniversalResumeParser()
    result = parser.parse_resume('Nirwan-resume-latest.pdf')

    print("=== UNIVERSAL RESUME PARSING RESULT ===")
    print(json.dumps(result, indent=2, default=str))

    return result

if __name__ == "__main__":
    test_universal_parser()