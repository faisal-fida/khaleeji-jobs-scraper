import random
import time

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import json
import numpy as np 



def load_api(file):
    try:
        with open(file, 'r') as f:
            api_key = f.read().strip()  # Read the API key from the file and remove leading/trailing whitespaces
        return api_key
    except FileNotFoundError:
        print(f"File '{file}' not found.")
        return None
    except Exception as e:
        print(f"Error reading API key from file '{file}': {e}")
        return None

def extract_job_info(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""
                
        Your task is to extract the following information from a given job description (JD) and return it in a JSON format:
        
        RecruitmentPlace (recruitment place)
        responsibilities (responsibilities)
        skills (skills required for the role)
        qualifications (qualification required for the job)
        educationRequirements (educational qualification required for the job)
        experienceRequirements (professional experience required for the job)
        jobLocationType (job location type)
        orgheadquarter (organization headquarter)
        orgfounded (organization founded in year)
        orgtype (organization type)
        orgWeb(web domain of the organization)
        streetAddress (Address if available)
        addressLocality (Area name or city name)
        addressRegion (Region of address if available)
        postalCode (Postal code if available)
        deadLine(Dead line date. Format: DD MMM YYYY.)
        postedDate(job posted date. Format: DD MMM YYYY. Extract the time by subtracting the time the job was added by the current time)
        MonthsOfExperience(minimum number of months of total experience required)
        occupationalCategory(occupational category refer to ISCO-08, just the category name without the code)
        industry(category of business and organization)
        meta_description (SEO optimized clear and concise, 120-150 characters)
        addressCountryISO (ISO country code)
        jobLevel(give job role i.e Intern, General Support, Entry Professional, Mid-level Professional, Director and Top Executive, Chief and Senior Professional)
        category(By analyzing the given JD, Select a category that best represent the job.)
        jobCity (job city)
        jobCountry(job country. give full country name for example: United States, India, China etc. If the job is to be in multiple countries. then return: Multiple Locations)
        jobSummary (job Summary. Maximum 200 characters long.)
        jobGrade (job grade. It can be different for difernet organizations but for UN Agency jobs, select from ['USG','ASG','D-2','D-1','P-5','P-4','P-3','P-2','P-1']. Return null if not present.)
        jobBenefits (employee benefits provided by the job)
        EducationalOccupationalCredential_Category (The category or type of educational credential required. Be specific. You must choose from ['high school','associate degree','bachelor degree','postgraduate degree','professional certificate'])
        employmentType(type of employement. Choose from FULL_TIME, PART_TIME, CONTRACTOR, TEMPORARY, INTERN, VOLUNTEER, OTHER, PER_DIEM. if it is scholarship or fellowship or is labelled "Roster/Talent Pool" then return OTHER)
        aboutOrg(give short description of the organization mentioned in the text in clear and concise way. It must always be filled and shouldn't be left empty.)
        Please follow these guidelines and instructions when extracting the information:

        IMPORTANT: If a particular piece of information is not present or cannot be accurately determined from the JD, set its value to null in the JSON.

        For the RecruitmentPlace, extract the location where the job is being offered. This may be a city, state, or country, and may be listed in the job title, summary, or description.
        For the responsibilities, extract a bulleted or numbered list of the key tasks and duties that the job entails. if none found, return empty list
        For the skills, extract a list of the technical and/or soft skills that are required or preferred for the job. 
        For the qualifications, extract a list of the certifications, skills, licenses, or other educational qualifications that are required or preferred for the job. Atleast one is required.
        For the educationRequirements, extract the level of education (e.g. bachelor's degree, master's degree) and/or field of study that is required or preferred for the job. 
        For the experienceRequirements, extract the number of years and/or type of experience (e.g. managerial, technical) that is required or preferred for the job.
        For the jobLocationType, extract whether the job is ON-SITE, REMOTE, or hybrid if it is remote then return TELECOMMUTE instead of REMOTE.
        For the orgheadquarter, extract the location of the organization's headquarters. This may be listed on the organization's website or in the JD.
        For the orgfounded, extract the year that the organization was founded. This may be listed on the organization's website or in the JD.
        For the orgtype, extract the type of organization (e.g. non-profit, startup, corporation). This may be listed on the organization's website or in the JD.
        For the orgweb, extract the URL of the organization's website. This may be listed in the JD or on the organization's profile page on the job board. If it is not present in the text, find on your own. Make sure 
        For the streetAddress, extract the full street address of the job location, if available.
        For the addressLocality, extract the area name or city name of the job location.
        For the addressRegion, extract the region of the job location, if available.
        For the postalCode, extract the postal code of the job location, if available.
        For the MonthsOfExperience, extract the number of minimum months of total experience required.
        For the experienceRequirements, extract the requirements for the experience.
        For the jobSummary, give brief description of the main points in the JD. Lenght should be one-fourth of the JD.
        For the jobGrade, extract job grade if it is explicitly mention in the JD, if not then retun null.

        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.

        Use this JSON schema:
        data = {
                 'RecruitmentPlace': str,
                 'responsibilities': [str],
                 'skills': [str],
                 'qualifications': [str],
                 'educationRequirements': str,
                 'experienceRequirements': str,
                 'MonthsOfExperience' : int,
                 'jobLocationType': str,
                 'orgHeadquarter': str,
                 'orgFounded': int,
                 'orgType': str,
                 'orgWeb' :str,
                 'streetAddress': str,
                 'addressLocality': str,
                 'addressRegion': str,
                 'postalCode': str,
                 'deadLine' : str,
                 'postedDate':str,
                 'occupationalCategory' : str
                 'industry': str,
                 'meta_description' :str,
                 'addressCountryISO': str,
                 'jobLevel' : str,
                 'category' : str,
                 'jobCity' : str,
                 'jobCountry' : str,
                 'jobSummary': str,
                 'jobGrade' : str,
                 'EducationalOccupationalCredential_Category':str,
                 'employmentType': str,
                 'aboutOrg':str,
                 'keywords : [str],
                 'jobBenefits' : [str]
                 }
        Return a `json` with the extracted information.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)

    return load


def summarize(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "text/plain"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""
        
        Your task is to make the given job description (JD) clear and concise.
        
        Follow these instructions:
        Make sure that the text is as clear and concise as possible while retaining maximum amount of the information.
        Retain all the key information important for a job description . 
        Get rid of filler words/sentences.
        Get rid of repitions.
        Preserve job details, address details, and organization details.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    try:
        print(response.text)
    except:
        print(response)
    # print(type(response.text))

    # Parse the JSON response
    return response.text

def extract_job_info1(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""

        Your task is to extract the following information from a given job description (JD) and return it in a JSON format:

        RecruitmentPlace (recruitment place)
        responsibilities (responsibilities)
        skills (skills required for the role)
        qualifications (qualification required for the job.)
        educationRequirements (educational qualification required for the job)
        experienceRequirements (professional experience required for the job)
        jobLocationType (job location type)
        orgheadquarter (organization headquarter)
        orgfounded (organization founded in year)
        orgtype (organization type)
        streetAddress (Address if available)
        addressLocality (Area name or city name)
        addressRegion (Region of address if available)
        postalCode (Postal code if available)
        deadLine(Dead line date. Format: DD MMM YYYY.)
        postedDate(job posted date. Format: DD MMM YYYY. Extract the time by subtracting the time the job was added by the current time)
        MonthsOfExperience(minimum number of months of total experience required)
        occupationalCategory(occupational category refer to ISCO-08, just the category name without the code)
        industry(category of business and organization)
        meta_description (SEO optimized clear and concise, 120-150 characters)
        addressCountryISO (ISO country code)
        jobLevel(give job role i.e Intern, General Support, Entry Professional, Mid-level Professional, Director and Top Executive, Chief and Senior Professional)
        category(By analyzing the given JD, Select category(ies) that best represent the job. You can select two categories at most.)
        jobCity (job city)
        jobCountry(job country. give full country name for example: United States, India, China etc.If the job is labelled "Multiple lcations" or "Worldwide". then return: Multiple Locations)
        jobSummary (job Summary. Maximum 200 characters long.)
        jobGrade (job grade. It can be different for difernet organizations but for UN Agency jobs, select from ['USG','ASG','D-2','D-1','P-5','P-4','P-3','P-2','P-1']. Return null if not present.)
        jobBenefits (employee benefits provided by the job)
        currency (currency in which the salary would be paid i.e USD or INR etc)
        minSalary( minimum salary. If only salary is given then extract it)
        maxSalary (maximum salary for the job. If only salary is given then extract it)
        salaryunittext(If the annual salary amount is given, return YEAR. If monthly then return MONTH.)
        EducationalOccupationalCredential_Category (The category or type of educational credential required. Be specific. You must choose from ['high school','associate degree','bachelor degree','postgraduate degree','professional certificate']))
        employmentType(type of employement. Choose from FULL_TIME, PART_TIME, CONTRACTOR, TEMPORARY, INTERN, VOLUNTEER, OTHER, PER_DIEM. if it is scholarship or fellowship or is labelled "Roster/Talent Pool" then return OTHER))
        aboutOrg(give short description of the organization mentioned in the text in clear and concise way. It must always be filled and shouldn't be left empty.)
        Please follow these guidelines and instructions when extracting the information:

        IMPORTANT: If a particular piece of information is not present or cannot be accurately determined from the JD, set its value to null in the JSON.

        For the RecruitmentPlace, extract the location where the job is being offered. This may be a city, state, or country, and may be listed in the job title, summary, or description.
        For the responsibilities, extract a bulleted or numbered list of the key tasks and duties that the job entails. if none found, return empty list.
        For the skills, extract a list of the technical and/or soft skills that are required or preferred for the job.
        For the qualifications, extract a list of the certifications, skills licenses, or other educational qualifications that are required or preferred for the job. Atleast one is required.
        For the educationRequirements, extract the level of education (e.g. bachelor's degree, master's degree) and/or field of study that is required or preferred for the job.
        For the experienceRequirements, extract the number of years and/or type of experience (e.g. managerial, technical) that is required or preferred for the job.
        For the jobLocationType, extract whether the job is ON-SITE, REMOTE, or hybrid if it is remote then return TELECOMMUTE instead of REMOTE.
        For the orgheadquarter, extract the location of the organization's headquarters. This may be listed on the organization's website or in the JD.
        For the orgfounded, extract the year that the organization was founded. This may be listed on the organization's website or in the JD.
        For the orgtype, extract the type of organization (e.g. non-profit, startup, corporation). This may be listed on the organization's website or in the JD.
        For the orgweb, extract the domain of the organization's website. This may be listed in the JD or on the organization's profile page on the job board. If it is not present in the text, find on your own. Make sure 
        For the streetAddress, extract the full street address of the job location, if available.
        For the addressLocality, extract the area name or city name of the job location.
        For the addressRegion, extract the region of the job location, if available.
        For the postalCode, extract the postal code of the job location, if available.
        For the MonthsOfExperience, extract the number of minimum months of total experience required.
        For the experienceRequirements, extract the requirements for the experience.
        For the jobSummary, give brief description of the main points in the JD. Lenght should be one-fourth of the JD.
        For the jobGrade, extract job grade if it is explicitly mention in the JD, if not then retun null.
        
        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.

        Use this JSON schema:
        data = {
                 'RecruitmentPlace': str,
                 'responsibilities': [str],
                 'skills': [str],
                 'qualifications': [str],
                 'educationRequirements': str,
                 'EducationalOccupationalCredential_Category':str,
                 'experienceRequirements': str,
                 'MonthsOfExperience' : int,
                 'jobLocationType': str,
                 'orgHeadquarter': str,
                 'orgFounded': int,
                 'orgType': str,
                 'orgWeb' : str,
                 'streetAddress': str,
                 'addressLocality': str,
                 'addressRegion': str,
                 'postalCode': str,
                 'deadLine' : str,
                 'postedDate':str,
                 'occupationalCategory' : str
                 'industry': str,
                 'meta_description' :str,
                 'addressCountryISO': str,
                 'jobLevel' : str,
                 'category' : [str],
                 'jobCity' : str,
                 'jobCountry' : str,
                 'jobSummary': str,
                 'jobGrade' : str,
                 'currency': str,
                 'minSalary':int,
                 'maxSalary':int,
                 'salaryunittext': str,
                 'EducationalOccupationalCredential_Category':str,
                 'employmentType':str,
                 'aboutOrg':str,
                 'keywords : [str],
                 'jobBenefits' : [str]
                 }
        Return a `json` with the extracted information.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)
    return load
def count_token(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "text/plain"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )

    # Call the LLM to extract the information
    response = model.count_tokens(JD)
    return response

def give_category(lisOfstrings):
    genai.configure(api_key="AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM")

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "text/plain"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )

    category = []
    for string in lisOfstrings:
        string_embedding = np.array(genai.embed_content(model="models/text-embedding-004", content=string)['embedding'])
        with open("categories_embeddings.pkl", 'rb') as f:
            categoriesEmbeddings = pickle.load(f)
        max_similarity = -1
        closest_word = None
        for word in categoriesEmbeddings.keys():
            similarity = cosine_similarity([string_embedding], [categoriesEmbeddings[word]])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                closest_word = word
        category.append(closest_word)
    return category

def extract_deadline(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""

        Your task is to extract the job deadline from a given job description (JD) and return it in a JSON format. if it is not available, set it to null. The date format must be 'DD MMM YYYY'.
        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.
        Use this JSON schema:

        data = {
            'deadLine' : str
        }

        if you can't find the data, just return null.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)
    return load


def extract_job_info2(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""

        Your task is to extract the following information from a given job description (JD) and return it in a JSON format:

        RecruitmentPlace (recruitment place)
        responsibilities (responsibilities)
        skills (skills required for the role)
        qualifications (qualification required for the job.)
        educationRequirements (educational qualification required for the job)
        experienceRequirements (professional experience required for the job)
        jobLocationType (job location type)
        orgheadquarter (organization headquarter)
        orgfounded (organization founded in year)
        orgtype (organization type)
        streetAddress (Address if available)
        addressLocality (Area name or city name)
        addressRegion (Region of address if available)
        postalCode (Postal code if available)
        deadLine(Dead line date. Format: DD MMM YYYY.)
        postedDate(job posted date. Format: DD MMM YYYY. Extract the time by subtracting the time the job was added by the current time)
        MonthsOfExperience(minimum number of months of total experience required)
        occupationalCategory(occupational category refer to ISCO-08, just the category name without the code)
        industry(category of business and organization)
        meta_description (SEO optimized clear and concise, 120-150 characters)
        addressCountryISO (ISO country code)
        jobLevel(give job role i.e Intern, General Support, Entry Professional, Mid-level Professional, Director and Top Executive, Chief and Senior Professional)
        category(By analyzing the given JD, Select category(ies) that best represent the job. You can select two categories at most.)
        jobCity (job city)
        jobCountry(job country. give full country name for example: United States, India, China etc.If the job is labelled "Multiple lcations" or "Worldwide". then return: Multiple Locations)
        jobSummary (job Summary. Maximum 200 characters long.)
        jobGrade (job grade. It can be different for difernet organizations but for UN Agency jobs, select from ['USG','ASG','D-2','D-1','P-5','P-4','P-3','P-2','P-1']. Return null if not present.)
        jobBenefits (employee benefits provided by the job)
        orgWeb(web domain of the organization)
        currency (currency in which the salary would be paid i.e USD or INR etc. if none is present default is USD)
        minSalary( minimum salary. If only salary is given then extract it, return 0 if not present)
        maxSalary (maximum salary for the job. If only salary is given then extract it. if it is not present assign the value of minSalary)
        salaryunittext(If the annual salary amount is given, return YEAR. If monthly then return MONTH.)
        EducationalOccupationalCredential_Category (The category or type of educational credential required. Be specific.. You must choose from ['high school','associate degree','bachelor degree','postgraduate degree','professional certificate']))
        employmentType(type of employement. Choose from FULL_TIME, PART_TIME, CONTRACTOR, TEMPORARY, INTERN, VOLUNTEER, OTHER, PER_DIEM. if it is scholarship or fellowship or is labelled "Roster/Talent Pool" then return OTHER))
        aboutOrg(give short description of the organization mentioned in the text in clear and concise way. It must always be filled and shouldn't be left empty.)
        Please follow these guidelines and instructions when extracting the information:

        IMPORTANT: If a particular piece of information is not present or cannot be accurately determined from the JD, set its value to null in the JSON.

        For the RecruitmentPlace, extract the location where the job is being offered. This may be a city, state, or country, and may be listed in the job title, summary, or description.
        For the responsibilities, extract a bulleted or numbered list of the key tasks and duties that the job entails. if none found, return empty list.
        For the skills, extract a list of the technical and/or soft skills that are required or preferred for the job.
        For the qualifications, extract a list of the certifications, skills licenses, or other educational qualifications that are required or preferred for the job. Atleast one is required.
        For the educationRequirements, extract the level of education (e.g. bachelor's degree, master's degree) and/or field of study that is required or preferred for the job.
        For the experienceRequirements, extract the number of years and/or type of experience (e.g. managerial, technical) that is required or preferred for the job.
        For the jobLocationType, extract whether the job is ON-SITE, REMOTE, or hybrid if it is remote then return TELECOMMUTE instead of REMOTE.
        For the orgheadquarter, extract the location of the organization's headquarters. This may be listed on the organization's website or in the JD.
        For the orgfounded, extract the year that the organization was founded. This may be listed on the organization's website or in the JD.
        For the orgtype, extract the type of organization (e.g. non-profit, startup, corporation). This may be listed on the organization's website or in the JD.
        For the orgweb, extract the domain of the organization's website. This may be listed in the JD or on the organization's profile page on the job board. If it is not present in the text, find on your own. Make sure it is corect.
        For the streetAddress, extract the full street address of the job location, if available.
        For the addressLocality, extract the area name or city name of the job location.
        For the addressRegion, extract the region of the job location, if available.
        For the postalCode, extract the postal code of the job location, if available.
        For the MonthsOfExperience, extract the number of minimum months of total experience required.
        For the experienceRequirements, extract the requirements for the experience.
        For the jobSummary, give brief description of the main points in the JD. Lenght should be one-fourth of the JD.
        For the jobGrade, extract job grade if it is explicitly mention in the JD, if not then retun null.
        
        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.

        Use this JSON schema:
        data = {
                 'RecruitmentPlace': str,
                 'responsibilities': [str],
                 'skills': [str],
                 'qualifications': [str],
                 'educationRequirements': str,
                 'EducationalOccupationalCredential_Category':str,
                 'experienceRequirements': str,
                 'MonthsOfExperience' : int,
                 'jobLocationType': str,
                 'orgHeadquarter': str,
                 'orgFounded': int,
                 'orgType': str,
                 'orgWeb':str,
                 'streetAddress': str,
                 'addressLocality': str,
                 'addressRegion': str,
                 'postalCode': str,
                 'deadLine' : str,
                 'postedDate':str,
                 'occupationalCategory' : str
                 'industry': str,
                 'meta_description' :str,
                 'addressCountryISO': str,
                 'jobLevel' : str,
                 'category' : [str],
                 'jobCity' : str,
                 'jobCountry' : str,
                 'jobSummary': str,
                 'jobGrade' : str,
                 'currency': str,
                 'minSalary':int,
                 'maxSalary':int,
                 'salaryunittext': str,
                 'EducationalOccupationalCredential_Category':str,
                 'employmentType':str,
                 'aboutOrg':str,
                 'keywords : [str],
                 'jobBenefits' : [str]
                 }
        Return a `json` with the extracted information.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)

    return load

def extract_description(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""

        Your task is to extract the following information from a given page and return it in a JSON format:
        is_valid (Check if the page contains the job posting with job description, response should be in boolean True/False. If you cannot find any job description just return False)
        header ( header part of the web page (which might be for a job posting), it might contain title of the job, job post date etc))
        description ( from the given page, extract the desctiption. Note that it should be properly formatted and looks appealing)
        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.

        Use this JSON schema:
        data = {
                'is_valid' : bool,
                'header':str,
                'description' : str
                 }
        Return a `json` with the extracted information.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)

    return load

def extract_job_info3(JD):
    API_KEY = 'AIzaSyAsCniqEhNyOK9xfPKP3bLZ4msfI9ul-qM'
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"response_mime_type": "application/json"},

                                  safety_settings=[
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HARASSMENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_HATE_SPEECH",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                      {
                                          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                          "threshold": "BLOCK_NONE",
                                      },
                                  ]
                                  )
    prompt = JD+"""

        Your task is to extract the following information from a given job description (JD) and return it in a JSON format:

        jtitle (main title of the job)
        orgName (Name of the organization offering the job. It shouldn't be an abbreviation. Also Give name of the organiztion (i.e UNOPS, UNDP) not its branchs (i.e UNOPS Pakistan, UNDP India etc)
        sname (abbreviation of the Organization name, if available. If it cannot be found, then return the organization name as it is)
        RecruitmentPlace (recruitment place)
        responsibilities (responsibilities)
        skills (skills required for the role)
        qualifications (qualification required for the job.)
        educationRequirements (educational qualification required for the job)
        experienceRequirements (professional experience required for the job)
        jobLocationType (job location type)
        orgheadquarter (organization headquarter)
        orgfounded (organization founded in year)
        orgtype (organization type)
        streetAddress (Address if available)
        addressLocality (Area name or city name)
        addressRegion (Region of address if available)
        postalCode (Postal code if available)
        deadLine(Dead line date. Format: DD MMM YYYY.)
        postedDate(job posted date. Format: DD MMM YYYY. Extract the time by subtracting the time the job was added by the current time)
        MonthsOfExperience(minimum number of months of total experience required)
        occupationalCategory(occupational category refer to ISCO-08, just the category name without the code)
        industry(category of business and organization)
        meta_description (SEO optimized clear and concise, 120-150 characters)
        addressCountryISO (ISO country code)
        jobLevel(give job role i.e Intern, General Support, Entry Professional, Mid-level Professional, Director and Top Executive, Chief and Senior Professional)
        category(By analyzing the given JD, Select category(ies) that best represent the job. You can select two categories at most.)
        jobCity (job city)
        jobCountry(job country. give full country name for example: United States, India, China etc.If the job is labelled "Multiple lcations" or "Worldwide". then return: Multiple Locations)
        jobSummary (job Summary. Maximum 200 characters long.)
        jobGrade (job grade. It can be different for difernet organizations but for UN Agency jobs, select from ['USG','ASG','D-2','D-1','P-5','P-4','P-3','P-2','P-1']. Return null if not present.)
        jobBenefits (employee benefits provided by the job)
        orgWeb(web domain of the organization)
        currency (currency in which the salary would be paid i.e USD or INR etc. if none is present default is USD)
        minSalary( minimum salary. If only salary is given then extract it, return 0 if not present)
        maxSalary (maximum salary for the job. If only salary is given then extract it. if it is not present assign the value of minSalary)
        salaryunittext(If the annual salary amount is given, return YEAR. If monthly then return MONTH.)
        EducationalOccupationalCredential_Category (The category or type of educational credential required. Be specific.. You must choose from ['high school','associate degree','bachelor degree','postgraduate degree','professional certificate']))
        employmentType(type of employement. Choose from FULL_TIME, PART_TIME, CONTRACTOR, TEMPORARY, INTERN, VOLUNTEER, OTHER, PER_DIEM. if it is scholarship or fellowship or is labelled "Roster/Talent Pool" then return OTHER))
        aboutOrg(give short description of the organization mentioned in the text in clear and concise way. It must always be filled and shouldn't be left empty.)
        Please follow these guidelines and instructions when extracting the information:

        IMPORTANT: If a particular piece of information is not present or cannot be accurately determined from the JD, set its value to null in the JSON.

        For the RecruitmentPlace, extract the location where the job is being offered. This may be a city, state, or country, and may be listed in the job title, summary, or description.
        For the responsibilities, extract a bulleted or numbered list of the key tasks and duties that the job entails. if none found, return empty list.
        For the skills, extract a list of the technical and/or soft skills that are required or preferred for the job.
        For the qualifications, extract a list of the certifications, skills licenses, or other educational qualifications that are required or preferred for the job. Atleast one is required.
        For the educationRequirements, extract the level of education (e.g. bachelor's degree, master's degree) and/or field of study that is required or preferred for the job.
        For the experienceRequirements, extract the number of years and/or type of experience (e.g. managerial, technical) that is required or preferred for the job.
        For the jobLocationType, extract whether the job is ON-SITE, REMOTE, or hybrid if it is remote then return TELECOMMUTE instead of REMOTE.
        For the orgheadquarter, extract the location of the organization's headquarters. This may be listed on the organization's website or in the JD.
        For the orgfounded, extract the year that the organization was founded. This may be listed on the organization's website or in the JD.
        For the orgtype, extract the type of organization (e.g. non-profit, startup, corporation). This may be listed on the organization's website or in the JD.
        For the orgweb, extract the domain of the organization's website. This may be listed in the JD or on the organization's profile page on the job board. If it is not present in the text, find on your own. Make sure it is corect.
        For the streetAddress, extract the full street address of the job location, if available.
        For the addressLocality, extract the area name or city name of the job location.
        For the addressRegion, extract the region of the job location, if available.
        For the postalCode, extract the postal code of the job location, if available.
        For the MonthsOfExperience, extract the number of minimum months of total experience required.
        For the experienceRequirements, extract the requirements for the experience.
        For the jobSummary, give brief description of the main points in the JD. Lenght should be one-fourth of the JD.
        For the jobGrade, extract job grade if it is explicitly mention in the JD, if not then retun null.
        
        Please ensure that the extracted information is accurate, relevant, and concise. Use your best judgment and common sense when interpreting the JD and extracting the information.

        Use this JSON schema:
        data = {
                 'jtitle' : str,
                 'orgName' : str,
                 'sname' : str,
                 'RecruitmentPlace': str,
                 'responsibilities': [str],
                 'skills': [str],
                 'qualifications': [str],
                 'educationRequirements': str,
                 'EducationalOccupationalCredential_Category':str,
                 'experienceRequirements': str,
                 'MonthsOfExperience' : int,
                 'jobLocationType': str,
                 'orgHeadquarter': str,
                 'orgFounded': int,
                 'orgType': str,
                 'orgWeb':str,
                 'streetAddress': str,
                 'addressLocality': str,
                 'addressRegion': str,
                 'postalCode': str,
                 'deadLine' : str,
                 'postedDate':str,
                 'occupationalCategory' : str
                 'industry': str,
                 'meta_description' :str,
                 'addressCountryISO': str,
                 'jobLevel' : str,
                 'category' : [str],
                 'jobCity' : str,
                 'jobCountry' : str,
                 'jobSummary': str,
                 'jobGrade' : str,
                 'currency': str,
                 'minSalary':int,
                 'maxSalary':int,
                 'salaryunittext': str,
                 'EducationalOccupationalCredential_Category':str,
                 'employmentType':str,
                 'aboutOrg':str,
                 'keywords : [str],
                 'jobBenefits' : [str]
                 }
        Return a `json` with the extracted information.
        """

    # Call the LLM to extract the information
    response = model.generate_content(prompt)
    print(response.text)
    try:
        data = response.text
        load = json.loads(rf"{data}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")
        print("returning eval decoded")
        return eval(response.text)

    return load