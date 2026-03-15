import streamlit as st
import asyncio
import os
from translator import translate_all_chunks
from pdf_processor import process_pdf

st.set_page_config(page_title="Meitei PDF Translator", layout="wide")
st.title("📝 Meitei Mayek Translator")

# with st.sidebar:
#     st.header("Settings")
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAkoVASsCpQP0qa6oCdfAMG6XjAI9ynis0")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
# else:
#     api_key = st.text_input("Google API Key", type="password")
#     if api_key:
#         os.environ["GOOGLE_API_KEY"] = api_key

# model_id = st.text_input("Model ID", value="gemini-3-flash-preview")
model_id = "gemini-3-flash-preview"
# chunk_size = st.slider("Chunk Size", 2000, 8000, 4000)
chunk_size = 2000

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and api_key:
    if st.button("Start Parallel Translation"):
        with st.spinner("Processing large document..."):
            chunks = process_pdf(uploaded_file, chunk_size=chunk_size)
            st.info(f"Processing {len(chunks)} chunks in parallel...")
            
            # Running the async translation loop
            translations = asyncio.run(translate_all_chunks(chunks, model_id))
            
            st.success("Translation Complete!")
            
            # Combine translations into a single string
            full_translation_text = "\n\n".join(translations)
            
            # Display on screen
            st.subheader("Translated Text")
            st.text_area("Review translation here:", full_translation_text, height=400)
            
            # Text download button
            # st.download_button(
            #     label="Download Translated Text (.txt)",
            #     data=full_translation_text,
            #     file_name="translated.txt",
            #     mime="text/plain"
            # )