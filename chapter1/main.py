from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import PyPDF2

key = config('GENAI_API_KEY')

# # print(key)

# from langchain_google_genai import ChatGoogleGenerativeAI

# llm = ChatGoogleGenerativeAI(model="gemini-pro" , api_key=key)
# res = llm.invoke("Hello, how are you?")

# print(res.content)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to Urdu. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)
# print(ai_msg.content)

# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "{input}"),
#     ]
# )

# chain = prompt | llm
# chain.invoke(
#     {
#         "input_language": "English",
#         "output_language": "German",
#         "input": "I love programming.",
#     }
# )

# New prompt template for writing an email to HR for hiring
# email_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that writes an email to HR for hiring. "
#             "The email should be addressed to {hr_email}, mention the job role {job_role}, "
#             "and optionally include the job description {job_description}.",
#         ),
#         ("human", "Write an email to HR."),
#     ]
# )

# email_chain = email_prompt | llm
# email_response = email_chain.invoke(
#     {
#         "hr_email": "hr@example.com",
#         "job_role": "Software Engineer",
#         "job_description": "Experience with Python and machine learning is preferred.",
#     }
# )

# print(email_response.content)

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
            "You are a helpful assistant that writes a personalized short email to HR for hiring in 2 paragraphs. "
            "Analyze the CV content provided and write an email addressed to {hr_email}, "
            "mention the job role {job_role}, and optionally include the job description {job_description}.",
        ),
        ("human", "{cv_content}"),
    ]
)

personalized_email_chain = personalized_email_prompt | llm
personalized_email_response = personalized_email_chain.invoke(
    {
        "hr_email": "isbah@alphabat.com",
        "job_role": "Software Engineer",
        "job_description": "Experience with Full Stack Development, Typescript is preferred.",
        "cv_content": cv_content,
    }
)

print(personalized_email_response.content)