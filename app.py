from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import random
import hashlib
from datetime import datetime
# ==================== ADD AI IMPORTS ====================
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import re
import json
from collections import Counter
import requests
import os
import time
import threading
from typing import Dict, List, Any

app = Flask(__name__)
app.secret_key = 'career-guidance-secret-key-2024'

# Debug file paths
print("🔍 Debugging file paths...")
print("Current directory:", os.getcwd())
print("Templates folder exists:", os.path.exists('templates'))
if os.path.exists('templates'):
    print("Files in templates:", os.listdir('templates'))

class AICareerPredictor:
    def __init__(self):
        # Simulated trained model
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.model = RandomForestClassifier(n_estimators=100)
        
        # Sample career dataset for ML
        self.career_dataset = self._create_sample_dataset()
        self._train_model()
    
    def _create_sample_dataset(self):
        # Simulated user profiles and their successful careers
        data = {
            'skills': [
                'python java sql machine_learning data_analysis',
                'javascript html css react nodejs',
                'python sql excel statistics data_visualization',
                'java spring hibernate sql microservices',
                'python tensorflow deep_learning neural_networks',
                'marketing seo social_media analytics content',
                'leadership strategy product_management agile',
                'cybersecurity networking security cryptography',
                'aws docker kubernetes cloud devops',
                'ui ux design figma prototyping'
            ],
            'interests': [
                'technology programming data_science',
                'web_development design creativity',
                'analytics business intelligence',
                'software_engineering systems architecture',
                'ai research innovation',
                'business marketing strategy',
                'management leadership planning',
                'security systems networking',
                'infrastructure cloud scalability',
                'design user_experience creativity'
            ],
            'career_path': [
                'data_scientist',
                'frontend_developer',
                'business_analyst',
                'backend_developer',
                'ai_engineer',
                'digital_marketer',
                'product_manager',
                'cybersecurity_analyst',
                'cloud_engineer',
                'ui_ux_designer'
            ]
        }
        return pd.DataFrame(data)
    
    def _train_model(self):
        # Combine features for training
        features = self.career_dataset['skills'] + ' ' + self.career_dataset['interests']
        X = self.vectorizer.fit_transform(features)
        y = self.career_dataset['career_path']
        self.model.fit(X, y)
    
    def predict_career(self, skills, interests):
        """Predict best career path using ML"""
        try:
            input_text = ' '.join(skills) + ' ' + ' '.join(interests)
            X_input = self.vectorizer.transform([input_text])
            prediction = self.model.predict(X_input)[0]
            probability = np.max(self.model.predict_proba(X_input))
            
            return {
                'predicted_career': prediction,
                'confidence': round(probability * 100, 2),
                'match_percentage': round(probability * 100)
            }
        except Exception as e:
            print(f"AI Prediction error: {e}")
            return {'predicted_career': 'software_developer', 'confidence': 75.0, 'match_percentage': 75}

class NLPResumeAnalyzer:
    def __init__(self):
        self.skills_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'data_science': ['machine learning', 'data analysis', 'statistics', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'databases': ['sql', 'mysql', 'mongodb', 'postgresql', 'oracle'],
            'cloud': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes'],
            'tools': ['git', 'jenkins', 'jira', 'linux', 'windows']
        }
    
    def analyze_resume_text(self, text):
        """Extract skills and experience from resume text using NLP patterns"""
        text_lower = text.lower()
        
        extracted_skills = {}
        for category, skills in self.skills_keywords.items():
            found_skills = []
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
            if found_skills:
                extracted_skills[category] = found_skills
        
        # Experience extraction
        experience = self._extract_experience(text)
        
        return {
            'skills': extracted_skills,
            'experience_level': experience,
            'skill_count': sum(len(skills) for skills in extracted_skills.values()),
            'primary_domain': self._identify_primary_domain(extracted_skills)
        }
    
    def _extract_experience(self, text):
        """Extract years of experience using regex patterns"""
        experience_patterns = [
            r'(\d+)\s*years?\s*experience',
            r'experience\s*:\s*(\d+)\s*years?',
            r'(\d+)\s*years?\s*in',
            r'(\d+)\s*years?\s*of\s*professional'
        ]
        
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                years = int(matches[0])
                if years <= 2:
                    return 'Entry Level (0-2 years)'
                elif years <= 5:
                    return 'Intermediate (2-5 years)'
                else:
                    return 'Senior (5+ years)'
        
        return 'Not specified'
    
    def _identify_primary_domain(self, skills):
        domains = {
            'data_science': ['python', 'machine learning', 'data analysis', 'statistics'],
            'web_development': ['javascript', 'html', 'css', 'react'],
            'software_development': ['java', 'c++', 'python', 'software engineering'],
            'cloud_devops': ['aws', 'azure', 'docker', 'kubernetes'],
            'cybersecurity': ['security', 'networking', 'cryptography']
        }
        
        domain_scores = {}
        for domain, domain_skills in domains.items():
            score = sum(1 for skill in domain_skills if any(skill in cat_skills for cat_skills in skills.values()))
            domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'

class IntelligentCareerRecommender:
    def __init__(self):
        self.job_profiles = self._load_job_profiles()
        self.vectorizer = TfidfVectorizer()
        self._build_similarity_matrix()
    
    def _load_job_profiles(self):
        return {
            'data_scientist': {
                'skills': 'python machine_learning statistics data_analysis sql tensorflow',
                'description': 'Analyze complex data and build predictive models',
                'demand': 'high'
            },
            'frontend_developer': {
                'skills': 'javascript html css react angular vue web_design',
                'description': 'Build user interfaces and web applications',
                'demand': 'high'
            },
            'backend_developer': {
                'skills': 'java python nodejs sql spring microservices api',
                'description': 'Develop server-side logic and databases',
                'demand': 'high'
            },
            'cybersecurity_analyst': {
                'skills': 'security networking cryptography risk_assessment ethical_hacking',
                'description': 'Protect systems from cyber threats',
                'demand': 'very_high'
            },
            'cloud_engineer': {
                'skills': 'aws azure docker kubernetes infrastructure devops',
                'description': 'Design and manage cloud infrastructure',
                'demand': 'high'
            }
        }
    
    def _build_similarity_matrix(self):
        job_texts = [profile['skills'] for profile in self.job_profiles.values()]
        self.job_vectors = self.vectorizer.fit_transform(job_texts)
        self.job_names = list(self.job_profiles.keys())
    
    def recommend_careers(self, user_skills, top_n=3):
        """AI-powered career recommendations using cosine similarity"""
        user_text = ' '.join(user_skills)
        user_vector = self.vectorizer.transform([user_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(user_vector, self.job_vectors).flatten()
        
        # Get top recommendations
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            job_name = self.job_names[idx]
            job_profile = self.job_profiles[job_name]
            match_score = round(similarities[idx] * 100, 2)
            
            recommendations.append({
                'title': job_name.replace('_', ' ').title(),
                'match': f'{match_score}%',
                'description': job_profile['description'],
                'demand': job_profile['demand'],
                'required_skills': job_profile['skills'].split(),
                'ai_generated': True
            })
        
        return recommendations

# Initialize AI modules
career_predictor = AICareerPredictor()
resume_analyzer = NLPResumeAnalyzer()
career_recommender = IntelligentCareerRecommender()

# ==================== ADVANCED AI CAREER AGENT ====================
class AdvancedCareerAI:
    def __init__(self):
        self.user_context = {}
        self.conversation_history = []
        self.personality_traits = {}
        
        # Enhanced career knowledge base
        self.career_knowledge_base = self._build_knowledge_base()
        self.skill_graph = self._build_skill_graph()
        
    def _build_knowledge_base(self):
        """Build comprehensive career knowledge base"""
        return {
            "emerging_roles": {
                "prompt_engineer": {
                    "name": "Prompt Engineer",
                    "description": "Specialist in crafting effective prompts for AI systems",
                    "skills": ["Natural Language Processing", "AI Understanding", "Creative Writing", "Testing"],
                    "salary": "$100k - $200k",
                    "growth": "Very High",
                    "companies": ["OpenAI", "Google", "Microsoft", "AI startups"]
                },
                "ai_ethics_specialist": {
                    "name": "AI Ethics Specialist",
                    "description": "Ensures ethical development and deployment of AI systems",
                    "skills": ["Ethics", "Law", "AI Technology", "Policy Making"],
                    "salary": "$90k - $180k",
                    "growth": "High",
                    "companies": ["Tech giants", "Research institutions", "Government"]
                }
            },
            "skill_trends": {
                "ai_ml": ["Transformers", "LLMs", "Computer Vision", "Reinforcement Learning"],
                "cloud": ["Multi-cloud", "Serverless", "Edge Computing", "Kubernetes"],
                "data": ["Data Engineering", "MLOps", "Data Governance", "Real-time Analytics"]
            },
            "learning_paths": {
                "fast_track": "3-6 months intensive learning",
                "balanced": "6-12 months with projects",
                "comprehensive": "12-24 months with certifications"
            }
        }
    
    def _build_skill_graph(self):
        """Build skill dependency graph for career progression"""
        return {
            "python": {"prerequisites": [], "leads_to": ["data_analysis", "web_development", "machine_learning"]},
            "javascript": {"prerequisites": [], "leads_to": ["frontend", "backend", "mobile_development"]},
            "sql": {"prerequisites": [], "leads_to": ["data_engineering", "business_intelligence", "database_administration"]},
            "machine_learning": {"prerequisites": ["python", "statistics"], "leads_to": ["deep_learning", "nlp", "computer_vision"]},
            "aws": {"prerequisites": ["networking"], "leads_to": ["devops", "cloud_architecture", "security"]}
        }
    
    def analyze_user_profile(self, user_data: Dict) -> Dict:
        """Analyze user profile for personalized recommendations"""
        analysis = {
            "skill_level": "beginner",
            "learning_pace": "moderate",
            "career_aspirations": [],
            "skill_gaps": [],
            "recommended_path": "balanced"
        }
        
        if user_data.get('experience', 0) > 3:
            analysis["skill_level"] = "intermediate"
        if user_data.get('experience', 0) > 7:
            analysis["skill_level"] = "advanced"
            
        return analysis
    
    def generate_personalized_roadmap(self, target_role: str, user_profile: Dict) -> Dict:
        """Generate AI-powered personalized career roadmap"""
        base_roadmap = self._get_base_roadmap(target_role)
        user_analysis = self.analyze_user_profile(user_profile)
        
        # Customize based on user profile
        if user_analysis["skill_level"] == "beginner":
            timeline_multiplier = 1.5
            emphasis = "fundamentals"
        elif user_analysis["skill_level"] == "intermediate":
            timeline_multiplier = 1.0
            emphasis = "specialization"
        else:
            timeline_multiplier = 0.7
            emphasis = "advanced_topics"
        
        personalized_roadmap = {
            "target_role": target_role,
            "timeline_months": base_roadmap["timeline"] * timeline_multiplier,
            "emphasis": emphasis,
            "phases": [],
            "skill_gaps": self._identify_skill_gaps(target_role, user_profile),
            "confidence_score": self._calculate_roadmap_confidence(target_role, user_profile)
        }
        
        return personalized_roadmap
    
    def _get_base_roadmap(self, role: str) -> Dict:
        """Get base roadmap for a role"""
        roadmaps = {
            "ai_engineer": {
                "timeline": 12,
                "phases": ["Foundation", "Core ML", "Specialization", "Production"]
            },
            "data_scientist": {
                "timeline": 10,
                "phases": ["Data Basics", "Analysis", "ML", "Deployment"]
            }
        }
        return roadmaps.get(role, {"timeline": 12, "phases": []})
    
    def _identify_skill_gaps(self, target_role: str, user_profile: Dict) -> List[str]:
        """Identify skill gaps for target role"""
        required_skills = self._get_required_skills(target_role)
        user_skills = user_profile.get('skills', [])
        return [skill for skill in required_skills if skill not in user_skills]
    
    def _get_required_skills(self, role: str) -> List[str]:
        """Get required skills for a role"""
        skill_requirements = {
            "ai_engineer": ["python", "machine_learning", "deep_learning", "statistics", "software_engineering"],
            "data_scientist": ["python", "sql", "statistics", "machine_learning", "data_visualization"],
            "cloud_engineer": ["aws", "docker", "kubernetes", "networking", "security"]
        }
        return skill_requirements.get(role, [])
    
    def _calculate_roadmap_confidence(self, target_role: str, user_profile: Dict) -> float:
        """Calculate confidence score for roadmap recommendation"""
        skill_gaps = self._identify_skill_gaps(target_role, user_profile)
        experience = user_profile.get('experience', 0)
        
        base_confidence = 0.7
        if len(skill_gaps) == 0:
            base_confidence += 0.2
        if experience >= 3:
            base_confidence += 0.1
            
        return min(base_confidence, 0.95)

# Initialize advanced AI
advanced_career_ai = AdvancedCareerAI()

# ==================== REAL-TIME JOB MARKET INTEGRATION ====================
class RealTimeMarketAnalyzer:
    def __init__(self):
        self.market_data = {}
        self.trend_indicators = {}
        
    def get_current_trends(self):
        """Get current job market trends (simulated)"""
        return {
            "high_demand_skills": [
                "Generative AI", "LLM Development", "MLOps", "Cloud Security",
                "Data Engineering", "DevOps", "Cybersecurity", "Blockchain"
            ],
            "growing_roles": [
                "AI Engineer", "Prompt Engineer", "Cloud Architect", 
                "Data Engineer", "Security Analyst", "DevOps Engineer"
            ],
            "declining_roles": [
                "Traditional IT Support", "Manual Testing", 
                "Legacy System Administration"
            ],
            "salary_trends": {
                "AI_ML": "15-25% annual growth",
                "Cloud": "12-20% annual growth", 
                "Cybersecurity": "10-18% annual growth",
                "Data_Science": "10-15% annual growth"
            }
        }

# Initialize market analyzer
market_analyzer = RealTimeMarketAnalyzer()

# ==================== ENHANCED CAREER AI ASSISTANT ====================
class EnhancedCareerAI:
    def __init__(self):
        self.career_domains = {
            "technology": {
                "ai_ml": {
                    "name": "🤖 Artificial Intelligence & Machine Learning",
                    "description": "Building intelligent systems that learn and adapt",
                    "demand": "Very High (45% growth)",
                    "skills": ["Python", "TensorFlow", "Deep Learning", "Statistics", "MLOps"],
                    "roadmap": self._get_ai_ml_roadmap(),
                    "salary": "₹8-35 LPA",
                    "companies": ["Google", "Microsoft", "Amazon", "TCS", "Infosys"]
                },
                "data_science": {
                    "name": "📊 Data Science & Analytics",
                    "description": "Extracting insights from data to drive decisions",
                    "demand": "High (35% growth)", 
                    "skills": ["Python", "SQL", "Statistics", "Data Visualization", "Big Data"],
                    "roadmap": self._get_data_science_roadmap(),
                    "salary": "₹7-30 LPA",
                    "companies": ["Analytics firms", "IT companies", "E-commerce", "Banks"]
                },
                "cybersecurity": {
                    "name": "🔒 Cybersecurity",
                    "description": "Protecting systems and data from cyber threats",
                    "demand": "Very High (50% growth)",
                    "skills": ["Network Security", "Ethical Hacking", "Cryptography", "Risk Assessment"],
                    "roadmap": self._get_cybersecurity_roadmap(),
                    "salary": "₹6-25 LPA", 
                    "companies": ["Security firms", "Banks", "IT companies", "Government"]
                },
                "cloud_computing": {
                    "name": "☁️ Cloud Computing",
                    "description": "Designing and managing cloud infrastructure",
                    "demand": "High (40% growth)",
                    "skills": ["AWS/Azure", "Docker", "Kubernetes", "DevOps", "Networking"],
                    "roadmap": self._get_cloud_roadmap(),
                    "salary": "₹8-28 LPA",
                    "companies": ["AWS", "Azure", "Google Cloud", "IT services"]
                }
            },
            "business": {
                "digital_marketing": {
                    "name": "📱 Digital Marketing",
                    "description": "Promoting products through digital channels",
                    "demand": "High (30% growth)",
                    "skills": ["SEO", "Social Media", "Content Marketing", "Analytics"],
                    "roadmap": self._get_digital_marketing_roadmap(),
                    "salary": "₹4-20 LPA",
                    "companies": ["Marketing agencies", "E-commerce", "Startups"]
                },
                "product_management": {
                    "name": "💼 Product Management", 
                    "description": "Leading product development and strategy",
                    "demand": "High (25% growth)",
                    "skills": ["Market Research", "Strategy", "Leadership", "Analytics"],
                    "roadmap": self._get_product_management_roadmap(),
                    "salary": "₹12-40 LPA",
                    "companies": ["Tech companies", "Startups", "E-commerce"]
                },
                "business_analysis": {
                    "name": "📈 Business Analysis",
                    "description": "Bridging business needs with technology solutions",
                    "demand": "Medium (20% growth)",
                    "skills": ["Requirements Gathering", "SQL", "Process Modeling", "Communication"],
                    "roadmap": self._get_business_analysis_roadmap(),
                    "salary": "₹5-18 LPA",
                    "companies": ["IT companies", "Consulting firms", "Banks"]
                }
            },
            "creative": {
                "ui_ux_design": {
                    "name": "🎨 UI/UX Design",
                    "description": "Designing user-friendly digital experiences", 
                    "demand": "High (35% growth)",
                    "skills": ["Figma", "User Research", "Wireframing", "Prototyping"],
                    "roadmap": self._get_ui_ux_roadmap(),
                    "salary": "₹5-22 LPA",
                    "companies": ["Tech companies", "Startups", "Design agencies"]
                }
            },
            "emerging": {
                "blockchain": {
                    "name": "⛓️ Blockchain Technology",
                    "description": "Developing decentralized applications and systems",
                    "demand": "Very High (60% growth)",
                    "skills": ["Solidity", "Smart Contracts", "Cryptography", "Web3"],
                    "roadmap": self._get_blockchain_roadmap(),
                    "salary": "₹10-35 LPA",
                    "companies": ["Crypto companies", "FinTech", "Startups"]
                }
            }
        }

    def understand_and_respond(self, user_message):
        user_message_lower = user_message.lower().strip()
        
        print(f"💬 User question: {user_message}")
        
        # Career roadmap requests
        if any(word in user_message_lower for word in ['roadmap', 'path', 'journey', 'how to become']):
            return self._generate_career_roadmap(user_message_lower)
        
        # Job market queries
        elif any(word in user_message_lower for word in ['job market', 'market trend', 'demand', 'hiring']):
            return self._get_job_market_insights()
        
        # Specific career domain queries
        elif any(word in user_message_lower for word in ['ai', 'machine learning', 'artificial intelligence']):
            return self._get_career_domain_info("ai_ml")
        elif any(word in user_message_lower for word in ['data science', 'data analyst']):
            return self._get_career_domain_info("data_science")
        elif any(word in user_message_lower for word in ['cyber security', 'cybersecurity']):
            return self._get_career_domain_info("cybersecurity")
        elif any(word in user_message_lower for word in ['cloud', 'aws', 'azure']):
            return self._get_career_domain_info("cloud_computing")
        elif any(word in user_message_lower for word in ['digital marketing', 'seo', 'social media']):
            return self._get_career_domain_info("digital_marketing")
        elif any(word in user_message_lower for word in ['product management', 'product manager']):
            return self._get_career_domain_info("product_management")
        elif any(word in user_message_lower for word in ['ui ux', 'ux design', 'user experience']):
            return self._get_career_domain_info("ui_ux_design")
        elif any(word in user_message_lower for word in ['blockchain', 'web3', 'crypto']):
            return self._get_career_domain_info("blockchain")
        elif any(word in user_message_lower for word in ['business analysis', 'business analyst']):
            return self._get_career_domain_info("business_analysis")
        
        # General career guidance
        elif any(word in user_message_lower for word in ['career', 'job', 'profession']):
            return self._get_career_overview()
        elif any(word in user_message_lower for word in ['hello', 'hi', 'hey']):
            return self._greet_user()
        elif any(word in user_message_lower for word in ['thank', 'thanks']):
            return "You're welcome! 😊 I'm glad I could help you plan your career journey. Feel free to ask about any specific career path!"
        
        else:
            return self._get_general_guidance(user_message)

    def _greet_user(self):
        return """🌟 **Welcome to Your Enhanced Career AI Assistant!** 🌟

I'm your intelligent career guide with **advanced features**:

🎯 **Career Roadmap Generator** - Get step-by-step learning paths
📊 **Real-time Job Market Insights** - Latest trends and demands  
🚀 **Specialized Career Domains** - In-depth guidance for modern careers
💡 **Skill Development Plans** - Structured learning approaches

**Try asking me:**
• "Show me AI career roadmap"
• "What's the job market for data science?"
• "How to become a cybersecurity expert?"
• "Blockchain developer career path"

**What career would you like to explore?**"""

    def _generate_career_roadmap(self, query):
        # Analyze query to determine which career roadmap to generate
        if any(word in query for word in ['ai', 'machine learning', 'artificial intelligence']):
            domain = self.career_domains["technology"]["ai_ml"]
        elif any(word in query for word in ['data science', 'data analyst']):
            domain = self.career_domains["technology"]["data_science"]
        elif any(word in query for word in ['cyber security', 'cybersecurity']):
            domain = self.career_domains["technology"]["cybersecurity"]
        elif any(word in query for word in ['cloud', 'aws', 'azure']):
            domain = self.career_domains["technology"]["cloud_computing"]
        elif any(word in query for word in ['digital marketing']):
            domain = self.career_domains["business"]["digital_marketing"]
        elif any(word in query for word in ['product management', 'product manager']):
            domain = self.career_domains["business"]["product_management"]
        elif any(word in query for word in ['ui ux', 'ux design']):
            domain = self.career_domains["creative"]["ui_ux_design"]
        elif any(word in query for word in ['blockchain', 'web3']):
            domain = self.career_domains["emerging"]["blockchain"]
        elif any(word in query for word in ['business analysis', 'business analyst']):
            domain = self.career_domains["business"]["business_analysis"]
        else:
            return self._get_career_overview()
        
        return f"""🗺️ **Career Roadmap: {domain['name']}**

**{domain['description']}**

📈 **Market Demand:** {domain['demand']}
💰 **Salary Range:** {domain['salary']}
🏢 **Top Companies:** {', '.join(domain['companies'][:5])}

**🚀 Step-by-Step Career Roadmap:**

{domain['roadmap']}

**💡 Pro Tips:**
• Build projects to showcase your skills
• Network with professionals in the field
• Stay updated with latest trends
• Consider relevant certifications

**Ready to start your journey? Which step interests you most?**"""

    def _get_career_domain_info(self, domain_key):
        # Find the domain in our structure
        domain = None
        for category in self.career_domains.values():
            if domain_key in category:
                domain = category[domain_key]
                break
        
        if not domain:
            return "I specialize in various career domains. Could you specify which field you're interested in?"
        
        return f"""🎯 **Career Domain: {domain['name']}**

**Description:** {domain['description']}

**📊 Market Insights:**
• Demand: {domain['demand']}
• Salary Range: {domain['salary']}
• Top Companies: {', '.join(domain['companies'][:5])}

**🔧 Key Skills Required:**
{chr(10).join(['• ' + skill for skill in domain['skills']])}

**Want to see the complete career roadmap for {domain['name']}?**"""

    def _get_job_market_insights(self):
        return """📈 **Real-time Job Market Insights 2024**

🔥 **Most In-Demand Skills:**
• Artificial Intelligence/Machine Learning
• Cloud Computing (AWS/Azure/GCP)
• Data Science & Analytics
• Cybersecurity
• Digital Marketing
• Blockchain Development

🚀 **High-Demand Roles:**
• Machine Learning Engineer
• Data Scientist
• Cloud Architect
• Cybersecurity Analyst
• Full Stack Developer

💰 **Salary Trends:**
• AI/ML: ₹8-35 LPA
• Data Science: ₹7-30 LPA
• Cloud Computing: ₹8-28 LPA
• Cybersecurity: ₹6-25 LPA

🌍 **Market Statistics:**
• 42% of companies offer remote options
• 35% increase in tech hiring in 2024

**Which career field would you like to explore in detail?**"""

    def _get_career_overview(self):
        return """🎯 **Career Exploration Hub**

I specialize in these **modern career domains**:

🤖 **Technology Careers:**
• Artificial Intelligence & Machine Learning
• Data Science & Analytics
• Cybersecurity
• Cloud Computing

💼 **Business Careers:**
• Digital Marketing
• Product Management  
• Business Analysis

🎨 **Creative Careers:**
• UI/UX Design

🚀 **Emerging Careers:**
• Blockchain Technology

**Which career path would you like to explore?**"""

    def _get_general_guidance(self, question):
        return f"""🤔 **I understand you're asking about:** "{question}"

As your Enhanced Career AI Assistant, I can help you with:

🗺️ **Career Roadmaps** - Step-by-step learning paths
📊 **Market Insights** - Real-time job trends and demands
🎯 **Domain Specialization** - Deep dives into specific careers
💡 **Skill Development** - Structured learning approaches

**What specific career guidance are you looking for?**"""

    # Career Roadmap Generators
    def _get_ai_ml_roadmap(self):
        return """**Phase 1: Foundation (3-4 months)**
• Learn Python programming basics
• Study mathematics (Linear Algebra, Calculus, Statistics)
• Understand basic data structures and algorithms

**Phase 2: Core ML (4-5 months)**
• Learn machine learning fundamentals
• Practice with Scikit-learn
• Work on data preprocessing and EDA
• Build basic ML projects

**Phase 3: Deep Learning (4-6 months)**
• Study neural networks and deep learning
• Learn TensorFlow/PyTorch
• Work on computer vision or NLP projects
• Understand model deployment basics

**Phase 4: Specialization (3-4 months)**
• Choose specialization (CV, NLP, etc.)
• Build advanced projects
• Learn MLOps and deployment
• Prepare for interviews"""

    def _get_data_science_roadmap(self):
        return """**Phase 1: Foundation (3 months)**
• Master Python for data analysis
• Learn SQL for data querying
• Study statistics and probability
• Practice data visualization

**Phase 2: Core Skills (4 months)**
• Learn Pandas, NumPy for data manipulation
• Practice data cleaning and preprocessing
• Study machine learning basics
• Work on analytical projects

**Phase 3: Advanced Topics (4 months)**
• Learn advanced statistical analysis
• Study big data tools (Spark)
• Practice A/B testing and experimentation
• Build end-to-end data projects

**Phase 4: Specialization (3 months)**
• Choose domain (business, healthcare, etc.)
• Learn domain-specific knowledge
• Build portfolio projects
• Prepare for data science interviews"""

    def _get_cybersecurity_roadmap(self):
        return """**Phase 1: Foundation (3 months)**
• Learn networking fundamentals
• Understand operating systems (Linux/Windows)
• Study basic programming (Python)
• Learn about cyber threats and attacks

**Phase 2: Core Security (4 months)**
• Study network security principles
• Learn about cryptography
• Practice ethical hacking basics
• Understand security protocols

**Phase 3: Specialization (5 months)**
• Choose path (penetration testing, security analysis)
• Get hands-on with security tools
• Practice in controlled environments
• Study compliance and regulations

**Phase 4: Advanced (3 months)**
• Learn advanced threat detection
• Study cloud security
• Get relevant certifications
• Build security portfolio"""

    def _get_cloud_roadmap(self):
        return """**Phase 1: Foundation (2 months)**
• Learn cloud computing concepts
• Choose a platform (AWS/Azure/GCP)
• Understand basic services and pricing
• Study networking fundamentals

**Phase 2: Core Services (3 months)**
• Learn compute services (EC2, Lambda)
• Understand storage solutions (S3, EBS)
• Study database services (RDS, DynamoDB)
• Practice with hands-on labs

**Phase 3: Advanced Topics (4 months)**
• Learn about security and IAM
• Study containerization (Docker, Kubernetes)
• Understand DevOps and CI/CD
• Practice infrastructure as code

**Phase 4: Specialization (3 months)**
• Get cloud certifications
• Build complex projects
• Learn about architecture design
• Prepare for cloud roles"""

    def _get_digital_marketing_roadmap(self):
        return """**Phase 1: Foundation (2 months)**
• Learn marketing fundamentals
• Understand digital marketing landscape
• Study consumer behavior
• Learn about marketing funnel

**Phase 2: Core Channels (3 months)**
• Master SEO and SEM
• Learn social media marketing
• Understand email marketing
• Study content marketing

**Phase 3: Analytics & Tools (2 months)**
• Learn Google Analytics
• Study marketing automation tools
• Understand A/B testing
• Practice campaign analysis

**Phase 4: Strategy & Management (3 months)**
• Learn marketing strategy
• Study budget management
• Understand ROI measurement
• Build marketing portfolio"""

    def _get_product_management_roadmap(self):
        return """**Phase 1: Foundation (2 months)**
• Learn product management basics
• Understand agile methodologies
• Study market research techniques
• Learn about user-centered design

**Phase 2: Core Skills (3 months)**
• Master requirement gathering
• Learn prioritization frameworks
• Study product metrics and analytics
• Understand roadmap planning

**Phase 3: Execution (3 months)**
• Learn about development processes
• Study launch strategies
• Understand user feedback collection
• Practice stakeholder management

**Phase 4: Leadership (2 months)**
• Learn product strategy
• Study team leadership
• Understand business alignment
• Build product portfolio"""

    def _get_ui_ux_roadmap(self):
        return """**Phase 1: Design Foundation (2 months)**
• Learn design principles
• Study color theory and typography
• Understand user psychology
• Practice with design tools (Figma)

**Phase 2: UX Research (2 months)**
• Learn user research methods
• Study information architecture
• Practice wireframing
• Understand usability principles

**Phase 3: UI Design (3 months)**
• Master visual design
• Learn prototyping
• Study interaction design
• Practice design systems

**Phase 4: Portfolio & Specialization (3 months)**
• Build complete projects
• Learn about specific industries
• Study advanced UX methodologies
• Prepare for design interviews"""

    def _get_blockchain_roadmap(self):
        return """**Phase 1: Foundation (2 months)**
• Learn blockchain fundamentals
• Understand cryptography basics
• Study Bitcoin and Ethereum
• Learn about smart contracts

**Phase 2: Development (3 months)**
• Learn Solidity programming
• Understand Web3 development
• Practice with development frameworks
• Study decentralized applications

**Phase 3: Advanced Concepts (3 months)**
• Learn about different consensus mechanisms
• Study token economics
• Understand security best practices
• Practice with test networks

**Phase 4: Specialization (2 months)**
• Choose specialization (DeFi, NFTs, etc.)
• Build complex dApps
• Learn about blockchain security
• Prepare for blockchain roles"""

    def _get_business_analysis_roadmap(self):
        return """**Phase 1: Foundation (2 months)**
• Learn business analysis fundamentals
• Understand requirements gathering
• Study process modeling techniques
• Learn about stakeholder management

**Phase 2: Core Skills (3 months)**
• Master requirements documentation
• Learn data analysis basics
• Study business process improvement
• Understand project management basics

**Phase 3: Technical Skills (3 months)**
• Learn SQL for data querying
• Understand basic system architecture
• Study data visualization tools
• Practice with BA tools (JIRA, Confluence)

**Phase 4: Specialization (2 months)**
• Choose industry domain
• Learn domain-specific knowledge
• Build BA portfolio
• Prepare for business analyst roles"""

# Global enhanced career AI instance
enhanced_career_ai = EnhancedCareerAI()

# ==================== USER STORAGE ====================
users = {
    'test@example.com': {
        'name': 'Demo User',
        'email': 'test@example.com',
        'password': hashlib.sha256('password123'.encode()).hexdigest(),
        'education_level': 'general',
        'profile_completion': 30,
        'completed_questionnaire': False,
        'created_at': '2024-01-01'
    }
}

# ==================== MONGODB SIMULATION ====================
print("🟢 Attempting to connect to MongoDB...")
print("📊 MongoDB connected successfully! (Simulated)")
print("🔗 Database: career_guidance_system")
print("👥 Collection: users")

# ==================== DASHBOARD ROUTE ====================
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        print("❌ User not logged in, redirecting to index")
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    print(f"📊 Dashboard accessed by: {session['user_id']}")
    
    # Get user data
    user_email = session['user_id']
    user_data = users.get(user_email, {})
    
    # Create profile data for dashboard with safe defaults
    profile_data = {
        'profile_completion': user_data.get('profile_completion', 30),
        'completed_questionnaire': user_data.get('completed_questionnaire', False),
        'education_level': user_data.get('education_level', 'Not set yet')
    }
    
    return render_template('dashboard.html', profile=profile_data)

# ==================== EDUCATION SELECTION ROUTES ====================
@app.route('/select-education/<level>')
def select_education(level):
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    # Update user's education level
    user_email = session['user_id']
    if user_email in users:
        users[user_email]['education_level'] = level
        users[user_email]['profile_completion'] = 60
        session['education_level'] = level
        
        print(f"✅ Education level updated to: {level} for {user_email}")
        flash(f'Education level set to {level.capitalize()}!', 'success')
    
    return redirect(url_for('dashboard'))

# ==================== AJAX EDUCATION SELECTION ====================
@app.route('/select-education-ajax/<level>', methods=['POST'])
def select_education_ajax(level):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'}), 401
    
    print(f"✅ AJAX Education level updated to: {level} for {session['user_id']}")
    
    # Update user's education level
    user_email = session['user_id']
    if user_email in users:
        users[user_email]['education_level'] = level
        users[user_email]['profile_completion'] = 60
        session['education_level'] = level
    
    return jsonify({'success': True, 'education_level': level})

# ==================== EDUCATION QUESTIONNAIRE ROUTES ====================
@app.route('/questionnaire/<education_level>')
def questionnaire(education_level):
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    if education_level in ['undergraduate', 'graduate', 'postgraduate']:
        session['education_level'] = education_level
        # Update user's education level in storage
        user_email = session['user_id']
        if user_email in users:
            users[user_email]['education_level'] = education_level
            users[user_email]['profile_completion'] = 60
        
        print(f"📝 Questionnaire started for: {education_level}")
        return render_template(f'questionnaire_{education_level}.html')
    else:
        flash('Invalid education level!', 'error')
        return redirect(url_for('dashboard'))

@app.route('/questionnaire/undergraduate')
def questionnaire_undergraduate():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    session['education_level'] = 'undergraduate'
    # Update user's education level
    user_email = session['user_id']
    if user_email in users:
        users[user_email]['education_level'] = 'undergraduate'
        users[user_email]['profile_completion'] = 60
    
    print(f"📝 Undergraduate questionnaire accessed by: {session['user_id']}")
    return render_template('questionnaire_undergraduate.html')

@app.route('/questionnaire/graduate')
def questionnaire_graduate():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    session['education_level'] = 'graduate'
    # Update user's education level
    user_email = session['user_id']
    if user_email in users:
        users[user_email]['education_level'] = 'graduate'
        users[user_email]['profile_completion'] = 60
    
    print(f"📝 Graduate questionnaire accessed by: {session['user_id']}")
    return render_template('questionnaire_graduate.html')

@app.route('/questionnaire/postgraduate')
def questionnaire_postgraduate():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    session['education_level'] = 'postgraduate'
    # Update user's education level
    user_email = session['user_id']
    if user_email in users:
        users[user_email]['education_level'] = 'postgraduate'
        users[user_email]['profile_completion'] = 60
    
    print(f"📝 Postgraduate questionnaire accessed by: {session['user_id']}")
    return render_template('questionnaire_postgraduate.html')

# ==================== GET QUESTIONNAIRE FORM ====================
@app.route('/get-questionnaire/<education_level>')
def get_questionnaire(education_level):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    # Render the appropriate questionnaire template
    return render_template(f'questionnaire_{education_level}.html')

@app.route('/submit-questionnaire', methods=['POST'])
def submit_questionnaire():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    try:
        # Collect form data
        education_level = request.form.get('education_level', '')
        field_of_study = request.form.get('field_of_study', '')
        technical_skills = request.form.get('technical_skills', '')
        interests = request.form.getlist('interests')
        work_environment = request.form.get('work_environment', '')
        experience = request.form.get('experience', '')
        
        print(f"📝 Questionnaire submitted for: {education_level}")
        print(f"   Field: {field_of_study}")
        print(f"   Skills: {technical_skills}")
        print(f"   Interests: {interests}")
        
        # Store questionnaire data in session
        questionnaire_data = {
            'education_level': education_level,
            'field_of_study': field_of_study,
            'technical_skills': technical_skills,
            'interests': interests,
            'work_environment': work_environment,
            'experience': experience,
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        session['questionnaire_data'] = questionnaire_data
        
        # Update user profile completion
        user_email = session['user_id']
        if user_email in users:
            users[user_email]['completed_questionnaire'] = True
            users[user_email]['profile_completion'] = 90
        
        flash('Questionnaire submitted successfully! Generating recommendations...', 'success')
        return redirect(url_for('recommendations'))
        
    except Exception as e:
        print(f"❌ Error submitting questionnaire: {e}")
        flash('Error submitting questionnaire. Please try again.', 'error')
        return redirect(url_for('dashboard'))

# ==================== AJAX QUESTIONNAIRE SUBMISSION ====================
@app.route('/submit-questionnaire-ajax', methods=['POST'])
def submit_questionnaire_ajax():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please login first'})
    
    try:
        # Collect form data
        education_level = request.form.get('education_level', '')
        field_of_study = request.form.get('field_of_study', '')
        technical_skills = request.form.get('technical_skills', '')
        interests = request.form.getlist('interests')
        work_environment = request.form.get('work_environment', '')
        experience = request.form.get('experience', '')
        
        print(f"📝 Questionnaire submitted for: {education_level}")
        print(f"   Field: {field_of_study}")
        print(f"   Skills: {technical_skills}")
        print(f"   Interests: {interests}")
        
        # Store questionnaire data in session
        questionnaire_data = {
            'education_level': education_level,
            'field_of_study': field_of_study,
            'technical_skills': technical_skills,
            'interests': interests,
            'work_environment': work_environment,
            'experience': experience,
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        session['questionnaire_data'] = questionnaire_data
        
        # Update user profile completion
        user_email = session['user_id']
        if user_email in users:
            users[user_email]['completed_questionnaire'] = True
            users[user_email]['profile_completion'] = 90
        
        # Generate recommendations
        recommendations = generate_enhanced_recommendations(education_level, questionnaire_data)
        
        return jsonify({
            'success': True, 
            'message': 'Questionnaire submitted successfully!',
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"❌ Error submitting questionnaire: {e}")
        return jsonify({'success': False, 'message': 'Error submitting questionnaire'})

def generate_enhanced_recommendations(education_level, questionnaire_data):
    """Generate career recommendations with AI enhancement"""
    
    base_recommendations = {
        'undergraduate': [
            {
                'title': '💻 Software Developer Intern', 
                'company': 'Tech Corp', 
                'match': '85%',
                'description': 'Build applications and gain real-world programming experience',
                'skills': 'Python, Java, Problem Solving',
                'salary': '$25-35/hour',
                'ai_generated': False
            },
            {
                'title': '📊 Data Analyst Trainee', 
                'company': 'Analytics Inc', 
                'match': '78%',
                'description': 'Analyze data and help businesses make data-driven decisions',
                'skills': 'SQL, Excel, Statistics',
                'salary': '$22-30/hour',
                'ai_generated': False
            },
            {
                'title': '🔒 Cybersecurity Intern', 
                'company': 'Security Firm', 
                'match': '82%',
                'description': 'Learn about network security and threat protection',
                'skills': 'Networking, Security Basics, Linux',
                'salary': '$24-32/hour',
                'ai_generated': False
            }
        ],
        'graduate': [
            {
                'title': '💻 Junior Software Engineer', 
                'company': 'Software Solutions', 
                'match': '88%',
                'description': 'Develop software applications and work in agile teams',
                'skills': 'Java, Spring Boot, SQL, Git',
                'salary': '$70,000 - $90,000',
                'ai_generated': False
            },
            {
                'title': '📈 Business Analyst', 
                'company': 'Consulting Firm', 
                'match': '82%',
                'description': 'Bridge business needs with technology solutions',
                'skills': 'Requirements Analysis, SQL, Communication',
                'salary': '$65,000 - $85,000',
                'ai_generated': False
            },
            {
                'title': '🌐 Full Stack Developer', 
                'company': 'Web Agency', 
                'match': '85%',
                'description': 'Build complete web applications from frontend to backend',
                'skills': 'JavaScript, React, Node.js, MongoDB',
                'salary': '$75,000 - $95,000',
                'ai_generated': False
            }
        ],
        'postgraduate': [
            {
                'title': '🧠 AI Research Scientist', 
                'company': 'AI Research Lab', 
                'match': '92%',
                'description': 'Conduct cutting-edge research in artificial intelligence',
                'skills': 'Python, TensorFlow, Research Methodology',
                'salary': '$120,000 - $160,000',
                'ai_generated': False
            },
            {
                'title': '💼 Senior Product Manager', 
                'company': 'Tech Giant', 
                'match': '89%',
                'description': 'Lead product strategy and cross-functional teams',
                'skills': 'Product Strategy, Leadership, Market Analysis',
                'salary': '$130,000 - $170,000',
                'ai_generated': False
            },
            {
                'title': '☁️ Cloud Architect', 
                'company': 'Enterprise Solutions', 
                'match': '87%',
                'description': 'Design and implement cloud infrastructure solutions',
                'skills': 'AWS, Azure, Kubernetes, Security',
                'salary': '$110,000 - $150,000',
                'ai_generated': False
            }
        ],
        'general': [
            {
                'title': 'Complete Career Assessment', 
                'company': 'Start your journey', 
                'match': '0%',
                'description': 'Please complete the career assessment questionnaire to get personalized recommendations',
                'skills': 'All fields',
                'salary': 'Varies',
                'ai_generated': False
            }
        ]
    }
    
    # Get AI-powered recommendations if questionnaire data exists
    ai_recommendations = []
    if questionnaire_data and questionnaire_data.get('technical_skills'):
        try:
            user_skills = questionnaire_data.get('technical_skills', '').split(',')
            user_interests = questionnaire_data.get('interests', [])
            
            # Get AI career prediction
            ai_career_prediction = career_predictor.predict_career(user_skills, user_interests)
            
            # Get intelligent recommendations
            ai_recommendations = career_recommender.recommend_careers(user_skills)
            
            print(f"🤖 AI Career Prediction: {ai_career_prediction}")
            print(f"🤖 AI Recommendations: {len(ai_recommendations)} careers")
            
        except Exception as e:
            print(f"❌ AI Recommendation error: {e}")
            ai_recommendations = []
    
    # Combine traditional and AI recommendations
    traditional_recs = base_recommendations.get(education_level, base_recommendations['general'])
    
    # Add AI flag to traditional recommendations
    for rec in traditional_recs:
        rec['ai_generated'] = False
    
    # Merge recommendations (AI first, then traditional)
    all_recommendations = ai_recommendations + traditional_recs
    
    return all_recommendations[:5]  # Return top 5 recommendations

# ==================== ENHANCED RECOMMENDATIONS ROUTE ====================
@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    print(f"🎯 Recommendations accessed by: {session['user_id']}")
    
    # Get user data
    user_email = session['user_id']
    user_data = users.get(user_email, {})
    education_level = session.get('education_level', 'general')
    questionnaire_data = session.get('questionnaire_data', {})
    
    # Generate enhanced recommendations based on questionnaire
    enhanced_recommendations = generate_enhanced_recommendations(education_level, questionnaire_data)
    
    return render_template('recommendations.html', 
                         recommendations=enhanced_recommendations,
                         education_level=education_level,
                         questionnaire_data=questionnaire_data)

# ==================== ENHANCED RESUME ANALYSIS ROUTES ====================
@app.route('/resume-upload')
def resume_upload():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    print(f"📄 Resume upload accessed by: {session['user_id']}")
    return render_template('resume_upload.html')

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('resume_upload'))
    
    try:
        # Simulated resume text
        sample_resume_text = """
        Python Developer with 3 years of experience in web development.
        Skills: Python, JavaScript, Django, Flask, React, SQL, MongoDB.
        Experience in building scalable applications and machine learning models.
        Education: Bachelor's in Computer Science.
        """
        
        # AI Resume Analysis
        resume_analysis = resume_analyzer.analyze_resume_text(sample_resume_text)
        
        # AI Career Recommendations based on resume
        user_skills = []
        for category_skills in resume_analysis['skills'].values():
            user_skills.extend(category_skills)
        
        resume_recommendations = career_recommender.recommend_careers(user_skills)
        
        # AI Career Prediction
        career_prediction = career_predictor.predict_career(
            user_skills, 
            ['technology', 'programming']
        )
        
        resume_data = {
            'skills': resume_analysis['skills'],
            'experience_level': resume_analysis['experience_level'],
            'education': 'Bachelor\'s in Computer Science',
            'summary': 'Software developer with experience in web development and programming.',
            'ai_analysis': {
                'primary_domain': resume_analysis['primary_domain'],
                'skill_count': resume_analysis['skill_count'],
                'career_prediction': career_prediction,
                'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Store in session
        session['resume_data'] = resume_data
        session['resume_recommendations'] = resume_recommendations
        
        flash('✅ Resume analyzed with AI successfully! Check your personalized job matches.', 'success')
        return redirect(url_for('resume_results'))
        
    except Exception as e:
        print(f"❌ Error processing resume: {e}")
        flash('Error processing resume. Please try again.', 'error')
        return redirect(url_for('resume_upload'))

@app.route('/resume-results')
def resume_results():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    resume_data = session.get('resume_data', {})
    recommendations = session.get('resume_recommendations', [])
    
    return render_template('resume_results.html', 
                         resume_data=resume_data, 
                         recommendations=recommendations)

# ==================== PROFILE ROUTE ====================
@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    print(f"👤 Profile accessed by: {session['user_id']}")
    
    user_email = session['user_id']
    user_data = users.get(user_email, {})
    
    return render_template('profile.html', user=user_data)

# ==================== ADMIN ROUTES ====================
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """Handle admin login"""
    if request.method == 'GET':
        print("🔐 Admin login page accessed")
        return render_template('admin_login.html')
    
    # Handle POST request for admin login
    try:
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        print(f"🔐 Admin login attempt: {username}")
        
        # Validation
        if not username or not password:
            flash('Admin username and password are required', 'error')
            return redirect('/admin-login')
        
        # Check admin credentials
        admin_credentials = {
            'admin': 'admin123',
            'careeradmin': 'career2024'
        }
        
        if username in admin_credentials and admin_credentials[username] == password:
            # Successful admin login
            session['admin_id'] = username
            session['admin_logged_in'] = True
            session['user_role'] = 'admin'
            
            print(f"✅ Admin logged in successfully: {username}")
            flash(f'Welcome, Admin {username}!', 'success')
            return redirect('/admin-dashboard')
        else:
            # Failed admin login
            print(f"❌ Admin login failed for: {username}")
            flash('Invalid admin credentials', 'error')
            return redirect('/admin-login')
            
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        flash('Admin login failed. Please try again.', 'error')
        return redirect('/admin-login')

@app.route('/admin-dashboard')
def admin_dashboard():
    """Admin dashboard page"""
    if not session.get('admin_logged_in'):
        print("❌ Unauthorized access to admin dashboard")
        flash('Please login as admin first', 'error')
        return redirect('/admin-login')
    
    print(f"📊 Admin dashboard accessed by: {session.get('admin_id')}")
    
    # Get user statistics
    user_count = len(users)
    demo_users = sum(1 for user in users.values() if user.get('email') == 'test@example.com')
    real_users = user_count - demo_users
    
    # Education level distribution
    education_distribution = {
        'undergraduate': 65,
        'graduate': 25, 
        'postgraduate': 10
    }
    
    # Career recommendations data
    career_data = {
        'Software Developer': 45,
        'Data Analyst': 30,
        'Cybersecurity': 25,
        'Business Analyst': 20,
        'Research': 15
    }
    
    return render_template('admin_dashboard.html', 
                         user_count=user_count,
                         demo_users=demo_users,
                         real_users=real_users,
                         users=users,
                         education_distribution=education_distribution,
                         career_data=career_data)

@app.route('/admin-logout')
def admin_logout():
    """Handle admin logout"""
    admin_id = session.get('admin_id', 'Unknown')
    session.pop('admin_id', None)
    session.pop('admin_logged_in', None)
    session.pop('user_role', None)
    
    print(f"👋 Admin logged out: {admin_id}")
    flash('Admin logged out successfully!', 'info')
    return redirect('/admin-login')

# ==================== AUTHENTICATION ROUTES ====================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Handle user registration"""
    if request.method == 'GET':
        print("📝 Signup page accessed")
        return render_template('signup.html')
    
    # Handle POST request for signup
    try:
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        print(f"📝 Signup attempt - Name: {name}, Email: {email}")
        
        # Validation
        if not name or not email or not password:
            flash('All fields are required', 'error')
            return redirect('/signup')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return redirect('/signup')
        
        # Check if user already exists
        if email in users:
            flash('Email already registered. Please login.', 'error')
            return redirect('/')
        
        # Create user
        users[email] = {
            'name': name,
            'email': email,
            'password': hashlib.sha256(password.encode()).hexdigest(),
            'education_level': 'general',
            'profile_completion': 30,
            'completed_questionnaire': False,
            'created_at': datetime.now().strftime('%Y-%m-%d')
        }
        
        print(f"✅ New user registered: {email}")
        print(f"📊 Users in database: {len(users)}")
        
        # AUTO LOGIN AFTER SIGNUP
        session['user_id'] = email
        session['user_name'] = name
        session['education_level'] = 'general'
        
        print(f"✅ Auto-login after signup: {email}")
        flash(f'Welcome to Career AI, {name}! Your account has been created successfully.', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        print(f"❌ Signup error: {e}")
        flash('Registration failed. Please try again.', 'error')
        return redirect('/signup')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'GET':
        print("🔐 Login page accessed")
        # If user is already logged in, redirect to dashboard
        if 'user_id' in session:
            return redirect(url_for('dashboard'))
        return render_template('index.html')
    
    # Handle POST request for login
    print("🔐 Login attempt")
    
    try:
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()
        
        print(f"🔐 Login attempt for: {email}")
        
        # Validation
        if not email or not password:
            flash('Email and password are required', 'error')
            return redirect('/')
        
        # Check if user exists and password matches
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        if email in users and users[email]['password'] == hashed_password:
            # Successful login
            session['user_id'] = email
            session['user_name'] = users[email]['name']
            session['education_level'] = users[email].get('education_level', 'general')
            
            print(f"✅ User logged in successfully: {email}")
            print(f"👤 Session created for: {users[email]['name']}")
            flash(f'Welcome back, {users[email]["name"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            # Failed login
            print(f"❌ Login failed for: {email}")
            flash('Invalid email or password', 'error')
            return redirect('/')
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        flash('Login failed. Please try again.', 'error')
        return redirect('/')

@app.route('/logout')
def logout():
    user_id = session.get('user_id', 'Unknown')
    session.clear()
    print(f"👋 User logged out: {user_id}")
    flash('You have been logged out successfully!', 'info')
    return redirect(url_for('index'))

# ==================== NEW AI API ENDPOINTS ====================
@app.route('/api/ai/career-prediction', methods=['POST'])
def ai_career_prediction():
    """AI endpoint for career prediction"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.get_json()
    skills = data.get('skills', [])
    interests = data.get('interests', [])
    
    prediction = career_predictor.predict_career(skills, interests)
    
    return jsonify({
        'success': True,
        'prediction': prediction,
        'ai_model': 'Random Forest Classifier',
        'confidence': prediction['confidence']
    })

@app.route('/api/ai/skill-analysis', methods=['POST'])
def ai_skill_analysis():
    """AI endpoint for skill gap analysis"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.get_json()
    current_skills = data.get('current_skills', [])
    target_role = data.get('target_role', '')
    
    # AI skill gap analysis
    recommended_skills = career_recommender.recommend_careers(current_skills)
    
    return jsonify({
        'success': True,
        'current_skills': current_skills,
        'target_role': target_role,
        'skill_gap_analysis': recommended_skills,
        'ai_recommendations': 'Based on cosine similarity matching'
    })

# ==================== ENHANCED CAREER AGENT ROUTES ====================
@app.route('/advanced-career-agent')
def advanced_career_agent():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    
    print(f"🤖 Advanced Career Agent accessed by: {session['user_id']}")
    return render_template('career_agent.html')

@app.route('/api/ai/analyze-profile', methods=['POST'])
def analyze_user_profile():
    """AI endpoint for comprehensive user profile analysis"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.get_json()
    user_profile = {
        'skills': data.get('skills', []),
        'experience': data.get('experience', 0),
        'education': data.get('education', ''),
        'interests': data.get('interests', [])
    }
    
    analysis = advanced_career_ai.analyze_user_profile(user_profile)
    
    return jsonify({
        'success': True,
        'analysis': analysis,
        'recommendations': generate_ai_recommendations(user_profile)
    })

@app.route('/api/ai/personalized-roadmap', methods=['POST'])
def generate_personalized_roadmap():
    """Generate AI-powered personalized career roadmap"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.get_json()
    target_role = data.get('target_role', '')
    user_profile = data.get('user_profile', {})
    
    roadmap = advanced_career_ai.generate_personalized_roadmap(target_role, user_profile)
    
    return jsonify({
        'success': True,
        'roadmap': roadmap,
        'market_insights': market_analyzer.get_current_trends()
    })

@app.route('/api/ai/skill-gap-analysis', methods=['POST'])
def skill_gap_analysis():
    """Advanced skill gap analysis with learning recommendations"""
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    data = request.get_json()
    current_skills = data.get('current_skills', [])
    target_role = data.get('target_role', '')
    
    # AI-powered skill gap analysis
    required_skills = advanced_career_ai._get_required_skills(target_role)
    skill_gaps = [skill for skill in required_skills if skill not in current_skills]
    
    # Generate learning path for skill gaps
    learning_path = generate_learning_path(skill_gaps)
    
    return jsonify({
        'success': True,
        'skill_gaps': skill_gaps,
        'learning_path': learning_path,
        'timeline_estimate': len(skill_gaps) * 2,  # weeks estimate
        'priority_skills': prioritize_skills(skill_gaps, target_role)
    })

def generate_learning_path(skills: List[str]) -> List[Dict]:
    """Generate learning path for skills"""
    learning_resources = {
        "python": {
            "resources": ["Python Crash Course", "Real Python", "Codecademy Python"],
            "projects": ["Build a calculator", "Web scraper", "Data analysis script"],
            "duration": "4-6 weeks"
        },
        "machine_learning": {
            "resources": ["Coursera ML", "Fast.ai", "Hands-on ML with Scikit-Learn"],
            "projects": ["Predict housing prices", "Image classifier", "Recommendation system"],
            "duration": "8-12 weeks"
        }
    }
    
    return [learning_resources.get(skill, {
        "resources": ["Online courses", "Documentation", "Practice projects"],
        "projects": ["Build related projects"],
        "duration": "4-8 weeks"
    }) for skill in skills]

def prioritize_skills(skills: List[str], target_role: str) -> List[str]:
    """Prioritize skills based on role importance"""
    role_skill_priority = {
        "ai_engineer": ["python", "machine_learning", "deep_learning"],
        "data_scientist": ["python", "sql", "statistics"],
        "cloud_engineer": ["aws", "docker", "networking"]
    }
    
    priority_order = role_skill_priority.get(target_role, [])
    return sorted(skills, key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))

def generate_ai_recommendations(user_profile: Dict) -> List[Dict]:
    """Generate AI-powered career recommendations"""
    recommendations = []
    
    # Analyze user profile for suitable roles
    skills = user_profile.get('skills', [])
    experience = user_profile.get('experience', 0)
    
    # Role matching logic
    if 'python' in skills and 'machine_learning' in skills:
        recommendations.append({
            'role': 'AI/ML Engineer',
            'match_score': 85,
            'reason': 'Your skills align perfectly with AI engineering requirements',
            'next_steps': ['Deep Learning', 'MLOps', 'Cloud Deployment']
        })
    
    if 'sql' in skills and 'python' in skills:
        recommendations.append({
            'role': 'Data Scientist',
            'match_score': 78,
            'reason': 'Strong foundation for data analysis and machine learning',
            'next_steps': ['Statistics', 'Data Visualization', 'Big Data Tools']
        })
    
    return recommendations

# ==================== ENHANCED ASK ENDPOINT WITH ADVANCED AI ====================
@app.route('/ask-advanced', methods=['POST'])
def ask_advanced():
    """Enhanced AI endpoint with advanced career guidance"""
    print("🚀 Advanced /ask-advanced route called")
    
    if 'user_id' not in session:
        print("❌ Unauthorized access to /ask-advanced")
        return jsonify({'error': 'Please login first'})
    
    try:
        user_message = request.form.get('message', '').strip()
        user_context = request.form.get('context', '{}')
        
        print(f"💬 Advanced User question: {user_message}")
        
        if not user_message:
            print("❌ Empty message received")
            return jsonify({'error': 'Message cannot be empty'})
        
        # Get enhanced response from advanced AI
        bot_response = generate_enhanced_response(user_message, user_context)
        print(f"🤖 Advanced AI Response generated successfully")
        
        return jsonify({
            'response': bot_response,
            'status': 'success',
            'context': get_updated_context(user_message),
            'suggestions': get_followup_suggestions(user_message)
        })
        
    except Exception as e:
        print(f"❌ Error in advanced ask route: {e}")
        return jsonify({
            'error': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        })

def generate_enhanced_response(user_message: str, context: str) -> str:
    """Generate enhanced AI response with advanced capabilities"""
    user_message_lower = user_message.lower()
    
    # Advanced intent recognition
    if any(word in user_message_lower for word in ['personalized', 'custom', 'for me']):
        return generate_personalized_advice(user_message, context)
    
    elif any(word in user_message_lower for word in ['trend', 'market', 'demand', 'future']):
        return generate_market_insights()
    
    elif any(word in user_message_lower for word in ['skill gap', 'missing', 'improve']):
        return generate_skill_gap_analysis(user_message)
    
    elif any(word in user_message_lower for word in ['salary', 'pay', 'compensation']):
        return generate_salary_insights(user_message)
    
    elif any(word in user_message_lower for word in ['interview', 'prepare', 'hiring']):
        return generate_interview_prep(user_message)
    
    else:
        # Fall back to enhanced career AI
        return enhanced_career_ai.understand_and_respond(user_message)

def generate_personalized_advice(user_message: str, context: str) -> str:
    """Generate personalized career advice"""
    return f"""🎯 **Personalized Career Analysis**

Based on your query: "{user_message}"

📊 **AI-Powered Insights:**
• Recommended learning path: 6-9 months intensive
• Key skills to focus on: Python, Machine Learning, Cloud Computing
• Target roles: AI Engineer, Data Scientist, ML Engineer

🚀 **Customized Roadmap:**
1. **Months 1-3:** Foundation (Python, Statistics, SQL)
2. **Months 4-6:** Core Skills (ML, Data Analysis, Cloud Basics)
3. **Months 7-9:** Specialization (Choose: AI/ML or Data Science)

💡 **Next Steps:**
• Build 2-3 portfolio projects
• Get AWS Cloud Practitioner certification
• Practice with real-world datasets

**Would you like me to create a detailed weekly study plan?**"""

def generate_market_insights() -> str:
    """Generate real-time market insights"""
    trends = market_analyzer.get_current_trends()
    
    return f"""📈 **Real-Time Market Insights 2024**

🔥 **Exploding Fields:**
• Generative AI & LLMs: 300% growth in jobs
• MLOps & AI Infrastructure: 150% growth
• Cloud Security: 120% growth

💰 **Salary Hotspots:**
• AI Research Scientists: $150k - $300k
• Cloud Architects: $130k - $250k  
• Data Engineers: $110k - $200k

🎯 **Future-Proof Skills:**
{chr(10).join(['• ' + skill for skill in trends['high_demand_skills'][:5]])}

🌍 **Remote Work Trends:**
• 65% of tech companies offer remote options
• Global hiring increased by 40%
• Freelance tech work up by 75%

**Which specific area would you like to explore?**"""

def generate_skill_gap_analysis(user_message: str) -> str:
    """Generate skill gap analysis"""
    return f"""🔍 **AI Skill Gap Analysis**

For your career goals mentioned in: "{user_message}"

📋 **Identified Skill Gaps:**
• Cloud Computing (AWS/Azure)
• Machine Learning Deployment
• Containerization (Docker/Kubernetes)

🎯 **Priority Learning Path:**
1. **Immediate (4 weeks):** AWS Fundamentals
2. **Short-term (8 weeks):** Docker & Kubernetes
3. **Medium-term (12 weeks):** MLOps & Model Deployment

🛠️ **Recommended Resources:**
• AWS Cloud Practitioner Certification
• Docker & Kubernetes: The Complete Guide
• MLOps Specialization on Coursera

💡 **Quick Wins:**
• Build a simple containerized web app
• Deploy a ML model on AWS SageMaker
• Create CI/CD pipeline for a project

**Ready to bridge these skill gaps?**"""

def generate_salary_insights(user_message: str) -> str:
    """Generate salary insights"""
    return f"""💰 **Comprehensive Salary Guide 2024**

Based on your interest in roles related to: "{user_message}"

💵 **Salary Ranges (Annual):**
• Entry-Level AI Engineer: $80,000 - $120,000
• Mid-Level Data Scientist: $100,000 - $150,000
• Senior Cloud Architect: $140,000 - $220,000
• Lead ML Engineer: $160,000 - $280,000

📊 **Factors Influencing Salary:**
• Location (SF/NY vs Remote)
• Company Size (Startup vs FAANG)
• Specialized Skills (LLMs, MLOps)
• Certifications (AWS, Google Cloud)

🎯 **Maximizing Your Earnings:**
• Add cloud certifications: +15-25%
• Specialize in high-demand areas: +20-30%
• Build strong portfolio: +10-20%
• Develop leadership skills: +15-25%

**Want specific salary negotiation tips?**"""

def generate_interview_prep(user_message: str) -> str:
    """Generate interview preparation guide"""
    return f"""🎤 **AI-Powered Interview Preparation**

For roles related to: "{user_message}"

📝 **Technical Interview Structure:**
• Coding Challenges (LeetCode Medium)
• System Design (Scalability, Architecture)
• ML/Data Case Studies
• Behavioral Questions (STAR method)

🧠 **Key Preparation Areas:**
• Algorithms & Data Structures
• System Design Principles
• Machine Learning Concepts
• Cloud Architecture Patterns

🛠️ **Practice Resources:**
• LeetCode (Top 100 questions)
• "Designing Data-Intensive Applications"
• "Machine Learning System Design" interviews
• AWS/GCP case studies

💡 **Pro Tips:**
• Practice explaining your thought process
• Prepare 5-10 project stories using STAR
• Research the company's tech stack
• Prepare questions for the interviewer

**Need mock interview questions for a specific role?**"""

def get_updated_context(user_message: str) -> Dict:
    """Get updated conversation context"""
    return {
        "last_topic": extract_topic(user_message),
        "user_interests": extract_interests(user_message),
        "conversation_stage": "active",
        "timestamp": datetime.now().isoformat()
    }

def get_followup_suggestions(user_message: str) -> List[str]:
    """Get AI-generated followup suggestions"""
    topic = extract_topic(user_message)
    
    suggestions = {
        "ai": ["Show me AI learning resources", "AI job market trends", "AI project ideas"],
        "data": ["Data science roadmap", "SQL practice questions", "Data visualization tools"],
        "cloud": ["Cloud certification guide", "AWS vs Azure comparison", "Cloud project ideas"],
        "career": ["Career transition tips", "Resume building guide", "Networking strategies"]
    }
    
    return suggestions.get(topic, [
        "Learn more about this field",
        "Explore related career paths", 
        "Get skill development tips"
    ])

def extract_topic(message: str) -> str:
    """Extract main topic from user message"""
    message_lower = message.lower()
    if any(word in message_lower for word in ['ai', 'machine learning', 'neural']):
        return "ai"
    elif any(word in message_lower for word in ['data', 'analysis', 'sql']):
        return "data"
    elif any(word in message_lower for word in ['cloud', 'aws', 'azure']):
        return "cloud"
    else:
        return "career"

def extract_interests(message: str) -> List[str]:
    """Extract user interests from message"""
    interests = []
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['python', 'programming']):
        interests.append("programming")
    if any(word in message_lower for word in ['machine learning', 'ai']):
        interests.append("ai_ml")
    if any(word in message_lower for word in ['cloud', 'aws']):
        interests.append("cloud")
        
    return interests

# ==================== ADDITIONAL ROUTES ====================
@app.route('/')
def index():
    print("🌐 Home page accessed")
    return render_template('index.html')

@app.route('/career-agent')
def career_agent_route():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('index'))
    print(f"🤖 Enhanced Career agent accessed by: {session['user_id']}")
    return render_template('career_agent.html')

@app.route('/ask', methods=['POST'])
def ask():
    print("🔍 /ask route called")
    
    if 'user_id' not in session:
        print("❌ Unauthorized access to /ask")
        return jsonify({'error': 'Please login first'})
    
    try:
        user_message = request.form.get('message', '').strip()
        
        print(f"💬 User question: {user_message}")
        
        if not user_message:
            print("❌ Empty message received")
            return jsonify({'error': 'Message cannot be empty'})
        
        # Get response from enhanced career AI
        bot_response = enhanced_career_ai.understand_and_respond(user_message)
        print(f"🤖 Enhanced AI Response generated successfully")
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"❌ Error in ask route: {e}")
        return jsonify({
            'error': 'Sorry, I encountered an error. Please try again.',
            'status': 'error'
        })

@app.route('/test')
def test():
    print("🧪 Test route accessed")
    return jsonify({'message': 'Enhanced Career AI Server is working!'})

# ==================== AI TESTING ROUTE ====================
@app.route('/test-ai-features')
def test_ai_features():
    """Test all AI features and show results"""
    
    print("🧪 TESTING ALL AI FEATURES...")
    
    # Test 1: Career Prediction AI
    test_skills = ['python', 'machine learning', 'data analysis', 'sql']
    test_interests = ['technology', 'data science', 'programming']
    
    career_prediction = career_predictor.predict_career(test_skills, test_interests)
    
    # Test 2: Career Recommendations AI
    ai_recommendations = career_recommender.recommend_careers(test_skills)
    
    # Test 3: Resume Analysis AI
    sample_resume = """
    Python Developer with 3 years of experience in web development.
    Skills: Python, JavaScript, Django, Flask, React, SQL, MongoDB, AWS.
    Experience in building scalable applications and machine learning models.
    Education: Bachelor's in Computer Science.
    """
    resume_analysis = resume_analyzer.analyze_resume_text(sample_resume)
    
    # Test 4: Career Agent AI
    test_question = "What career path for AI and machine learning?"
    career_agent_response = enhanced_career_ai.understand_and_respond(test_question)
    
    # Display results in a nice format
    results = {
        'ai_modules_loaded': True,
        'career_prediction': career_prediction,
        'ai_recommendations': ai_recommendations,
        'resume_analysis': resume_analysis,
        'career_agent_response': career_agent_response[:500] + "..." if len(career_agent_response) > 500 else career_agent_response
    }
    
    print("✅ AI FEATURES TEST RESULTS:")
    print(f"   Career Prediction: {career_prediction}")
    print(f"   AI Recommendations: {len(ai_recommendations)} jobs")
    print(f"   Resume Analysis: {resume_analysis['skill_count']} skills detected")
    print(f"   Career Agent: Response generated ({len(career_agent_response)} chars)")
    
    return jsonify(results)

# ==================== ENHANCED CAREER AGENT TESTING ====================
@app.route('/test-advanced-ai')
def test_advanced_ai():
    """Test advanced AI features"""
    print("🧪 TESTING ADVANCED AI FEATURES...")
    
    # Test advanced AI capabilities
    test_profile = {
        'skills': ['python', 'sql', 'machine learning'],
        'experience': 2,
        'education': 'Bachelor\'s in Computer Science',
        'interests': ['ai', 'data science']
    }
    
    # Test personalized roadmap
    roadmap = advanced_career_ai.generate_personalized_roadmap('ai_engineer', test_profile)
    
    # Test market analysis
    market_insights = market_analyzer.get_current_trends()
    
    results = {
        'advanced_ai_loaded': True,
        'personalized_roadmap': roadmap,
        'market_insights': market_insights,
        'skill_gap_analysis': advanced_career_ai._identify_skill_gaps('ai_engineer', test_profile),
        'user_analysis': advanced_career_ai.analyze_user_profile(test_profile)
    }
    
    print("✅ ADVANCED AI FEATURES TEST RESULTS:")
    print(f"   Personalized Roadmap: {roadmap['target_role']}")
    print(f"   Market Insights: {len(market_insights['high_demand_skills'])} trends")
    print(f"   Skill Gaps: {len(results['skill_gap_analysis'])} identified")
    
    return jsonify(results)

# ==================== DEBUG AI ROUTE ====================
@app.route('/debug-ai')
def debug_ai():
    """Check if AI modules are properly loaded"""
    return jsonify({
        'career_predictor_loaded': hasattr(career_predictor, 'predict_career'),
        'resume_analyzer_loaded': hasattr(resume_analyzer, 'analyze_resume_text'),
        'career_recommender_loaded': hasattr(career_recommender, 'recommend_careers'),
        'enhanced_career_ai_loaded': hasattr(enhanced_career_ai, 'understand_and_respond'),
        'advanced_career_ai_loaded': hasattr(advanced_career_ai, 'analyze_user_profile')
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting ADVANCED Flask Server with Enhanced AI...")
    print(f"📍 Server running on: http://127.0.0.1:5000")
    print(f"🔧 Debug mode: {True}")
    print("🤖 ADVANCED AI Features:")
    print("   • Personalized Career Roadmaps")
    print("   • Real-time Market Analysis") 
    print("   • Skill Gap Analysis")
    print("   • Advanced Career Agent")
    print("   • Enhanced Career Predictions")
    print("   • NLP Resume Analysis")
    print("   • Intelligent Recommendations")
    print("🧪 Test Advanced AI at: http://127.0.0.1:5000/test-advanced-ai")
    print("🔍 Career Agent at: http://127.0.0.1:5000/advanced-career-agent")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)