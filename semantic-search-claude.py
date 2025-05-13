import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from dotenv import load_dotenv
import os

async def setup_kernel_and_memory():
    # Load environment variables
    load_dotenv("C:/Users/ODL_User1697322/semantic-kernel/python/samples/getting_started/.env")
    
    # Initialize the kernel
    kernel = sk.Kernel()
    
    # Add Azure OpenAI chat service
    deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    # Define a consistent service ID
    service_id = "chat_completion"
    
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key
    )
    
    # Add the service with our defined service_id
    kernel.add_service(service_id, chat_service)

    # Configure Azure AI Search
    search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")

    print(search_endpoint, search_api_key, index_name)

    # Create memory store using Azure AI Search
    memory_store = AzureCognitiveSearchMemoryStore(
        search_endpoint=search_endpoint,
        key=search_api_key,
        index_name=index_name
    )

    # Initialize the memory store
    await memory_store.create_collection("endpoints")

    return kernel, memory_store

async def search_endpoints(query: str, kernel, memory_store):
    # Create a semantic function for searching
    prompt = """
    Search through the available endpoints and provide relevant information about: {{$input}}
    Please format the response in a clear and structured way.
    Available information:
    {{$context}}
    """
    
    search_function = kernel.create_semantic_function(
        prompt_template=prompt,
        max_tokens=200,
        temperature=0.7
    )

    # Perform the search using Azure AI Search
    search_results = await memory_store.search_async(
        collection="endpoints",
        query=query,
        limit=5
    )
    print(search_results)

    # Process results with the semantic function
    search_results_text = "\n".join([str(result) for result in search_results])
    
    # Create the context with our search results
    context = kernel.create_new_context()
    context["context"] = search_results_text
    context["input"] = query

    # Updated function invocation
    response = await search_function.invoke_async(context=context)
    return str(response)

async def main():
    try:
        # Setup kernel and memory store
        kernel, memory_store = await setup_kernel_and_memory()
        
        # Example search query
        query = input("Enter your search query about endpoints: ")
        results = await search_endpoints(query, kernel, memory_store)
        
        print("\nSearch Results:")
        print(results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())