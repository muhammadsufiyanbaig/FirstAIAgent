from decouple import config
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from langchain_core.prompts import ChatPromptTemplate
import os
import PyPDF2
import time
from google.api_core.exceptions import ResourceExhausted

key = config('GENAI_API_KEY')

# # print(key)


llm = ChatGoogleGenerativeAI(
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    api_key=key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# Function to read CV content from a PDF with error handling
def read_cv(cv_path):
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"The file at path {cv_path} does not exist.")
    with open(cv_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

cv_content = read_cv('MuhammadSufiyanBaig.pdf')

# New prompt template for writing a personalized email to HR
personalized_email_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that writes a personalized short email to HR for hiring in 2 paragraphs (each paragraphy have just 3 to 4 maximum) with complete contact details in the last including my porfolio website link."
            "Analyze the CV content provided and write an email addressed to {hr_email}, "
            "mention the job role {job_role}, and optionally include the job description {job_description}.",
        ),
        ("human", "{cv_content}"),
    ]
)

personalized_email_chain = personalized_email_prompt | llm

# Function to train the model with random data
def train_model():
    random_data = [
        {
            "hr_email": "hr@example.com",
            "job_role": "Software Engineer",
            "job_description": "We are looking for a Software Engineer with experience in Python and Django.",
            "cv_content": "John Doe\nExperience: 5 years in software development\nSkills: Python, Django, REST APIs\nEducation: B.Sc. in Computer Science"
        },
        {
            "hr_email": "recruiter@techcorp.com",
            "job_role": "Data Scientist",
            "job_description": "TechCorp is seeking a Data Scientist with expertise in machine learning and data analysis.",
            "cv_content": "Jane Smith\nExperience: 3 years in data science\nSkills: Machine Learning, Data Analysis, Python\nEducation: M.Sc. in Data Science"
        }
    ]

    for data in random_data:
        response = personalized_email_chain.invoke(data)
        print(response.content)

# Ensure the company name is added if not mentioned in the HR email
def add_company_name_if_missing(hr_email, company_name):
    if "@" in hr_email:
        domain = hr_email.split("@")[1]
        if not company_name:
            company_name = domain.split(".")[0]
    return company_name

personalized_email_response = personalized_email_chain.invoke(
    {
        "hr_email": "zaeemaltaf@flatgigs.com",
        "job_role": "",
        "job_description": "About the job EpicMetry is seeking a skilled MERN Stack Developer to join our dynamic team. As a Full Stack Developer, you will be responsible for designing and implementing high-quality software solutions that enhance our products and services. If you have a passion for technology, a knack for problem-solving, and a desire to work in an innovative environment, we want to hear from you! Key Responsibilities: Application Development: Design, build, and maintain high-performance web applications using MongoDB, Express, React, and Node.js. Collaborative Work: Collaborate with cross-functional teams to gather requirements and develop robust front-end and back-end components. Database Management: Develop and manage databases, ensuring data integrity and optimizing performance. API Integration: Create and integrate RESTful APIs to facilitate seamless communication between front-end and back-end services. Code Review and Maintenance: Conduct code reviews for quality assurance and maintainability, plus perform troubleshooting and debugging as required. Performance Tuning: Identify performance bottlenecks and optimize applications for speed and scalability. Stay Current: Keep up to date with the latest trends and technologies in web development. Requirements: Proven Experience: Solid experience as a MERN Stack Developer or in a similar role Technical Proficiency: Strong understanding of JavaScript, TypeScript, and the MERN stack technologies Front-End Skills: Proficiency in front-end frameworks (especially React) along with HTML, CSS, and responsive design principles Back-End Skills: Experience with Express and Node.js for creating server-side logic and APIs Database Expertise: Hands-on experience with MongoDB, including designing and managing data structures Version Control: Familiarity with version control systems, particularly Git Problem Solving: Strong problem-solving skills and an analytical mindset to troubleshoot and optimize code Communication Skills: Excellent communication and collaboration skills to engage effectively with team members and stakeholders. Requirements Who Should Apply? We are looking for candidates who: Are self-motivated and able to work independently as well as collaborativel Have a strong sense of accountability and ownership over their wor Embrace a positive team culture with a proactive attitude towards problem-solvin Show a commitment to continuous learning and professional developmen Are eager to contribute to innovative projects and new initiative  Hiring Process: Our hiring process aims to align the right candidates with our team culture and technical needs: Initial Interview: A friendly conversation to discuss your background and capabilities Technical Assessment: A practical assessment to showcase your technical skills Team Fit Interview: Meet with potential team members to discuss collaboration and culture Final Offer: If all goes well, we'll make you an offer to join us at Epicmetry Additional Information: References may be requested during the hiring process We may request access to your LinkedIn profile, if available Benefits: Market Competive Salary, Leaves, Health Insurance, Hybrid Work Model",
        "cv_content": cv_content,
    }
)

print(personalized_email_response.content)

# Train the model with random data
train_model()