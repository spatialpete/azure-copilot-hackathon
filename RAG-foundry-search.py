import os  
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI  

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = ""
endpoint = os.getenv("ENDPOINT_URL", "https://<env>.openai.azure.com/")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY)  

# Azure Cognitive Search Configuration
AZURE_SEARCH_ENDPOINT = "https://<env>.search.windows.net/"
AZURE_SEARCH_API_KEY = ""
AZURE_SEARCH_INDEX_NAME = "azureblob-index2"

# Initialize Azure OpenAI Service client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2025-01-01-preview",
)

def query_cognitive_search(query: str):
    # Initialize the Azure Cognitive Search client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

    # Perform the search
    results = search_client.search(query, top=5)  # Retrieve top 5 results
    search_results = []
    for result in results:
        search_results.append(result.get("content", ""))
    
    return "\n".join(search_results)

# Prepare the user query
user_query = "raleigh zoning?"

# Query Azure Cognitive Search
search_context = query_cognitive_search(user_query)

# Prepare the chat prompt
chat_prompt = [  
    {  
        "role": "user",  
        "content": user_query  
    }  
]

# Generate the completion  
completion = client.chat.completions.create(  
    model=deployment,
    messages=chat_prompt,
    max_tokens=800,  
    temperature=0.7,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,
    stop=None,  
    stream=False
)

print(completion.to_json())