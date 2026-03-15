import os
import asyncio
import hashlib
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Directory to store translated chunks to prevent re-processing
CACHE_DIR = "translation_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_path(text, model_id):
    """Generates a unique filename based on the text and model version."""
    unique_id = hashlib.md5(f"{text}_{model_id}".encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{unique_id}.json")

async def translate_chunk_async(text: str, model_id: str = "gemini-3-flash-preview"):
    if not text.strip():
        return ""

    # Check Cache First
    cache_path = get_cache_path(text, model_id)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)["translation"]

    # CRITICAL: temperature=0 ensures the same translation every time
    llm = ChatGoogleGenerativeAI(model=model_id, temperature=0)

    system_instruction = (
        "You are a professional translator specializing in the Meitei Mayek script. "
        "CRITICAL: Do NOT use the Bengali script (ফানুস). "
        "ONLY use the authentic Meitei Mayek Unicode characters (ꯐꯥꯅꯨꯁ). "
        "\n\nExample Translation:"
        "\nEnglish: The lantern was not only lighting the road."
        "\nManipuri: ꯐꯥꯅꯨꯁ ꯑꯗꯨꯅꯥ ꯂꯝꯕꯤ ꯃꯉꯥꯜ ꯊꯣꯛꯍꯟꯕꯥ ꯈꯛꯇꯥ ꯅꯠꯇꯦ꯫"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{text}") 
    ])
    
    chain = prompt | llm
    
    max_retries = 6
    base_delay = 15
    
    for attempt in range(max_retries):
        try:
            response = await chain.ainvoke({"text": text})
            
            content = response.content
            if isinstance(content, list):
                result = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            else:
                result = str(content)
            
            # Save to Cache before returning
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"translation": result, "original": text}, f, ensure_ascii=False)
                
            return result
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    print(f"Rate limit hit. Retrying chunk in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
            return f"Error: {error_msg}"
            
    return "Error: Maximum retries exceeded for rate limit."

async def translate_all_chunks(chunks, model_id="gemini-3-flash-preview"):
    """Processes chunks with conservative rate limits."""
    batch_size = 2  
    all_translations = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [translate_chunk_async(chunk, model_id) for chunk in batch]
        results = await asyncio.gather(*tasks)
        all_translations.extend(results)
        
        if i + batch_size < len(chunks):
            # Check if any tasks in the batch were NOT from cache
            # If they were cached, we don't need to sleep/wait for API limits
            await asyncio.sleep(10)
            
    return all_translations