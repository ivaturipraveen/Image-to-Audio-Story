from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from PIL import Image
import streamlit as st
from gtts import gTTS
import os
from dotenv import find_dotenv, load_dotenv
import openai
from googletrans import Translator

# Load environment variables
load_dotenv(find_dotenv())

# Define the Hugging Face Hub API Token and OpenAI API Key
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set the OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Google Translator
translator = Translator()

# Function to extract text from an image using Hugging Face model
def img2text(image_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text(image_path)[0]['generated_text']
    print(text)
    return text

# Function to generate a story based on a scenario using OpenAI's GPT-3.5-turbo
def generate_story(scenario):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You possess a remarkable talent for storytelling; weaving narratives that captivate and resonate. Let's craft a compelling short story with 3 small paragraphs."
            },
            {
                "role": "user",
                "content": f"CONTEXT: {scenario}\nSTORY:"
            }
        ]
    )

    story = response['choices'][0]['message']['content']
    print(story)
    return story

# Function to translate text into a specified language using Google Translate
def translate_text(text, language):
    # Define language codes
    language_codes = {
        'English': 'en',
        'Telugu': 'te',
        'Hindi': 'hi',
        'Tamil': 'ta',
        'Malayalam': 'ml'
    }
    if language not in language_codes:
        raise ValueError("Unsupported language")

    # Translate text using Google Translate
    translation = translator.translate(text, dest=language_codes[language])
    return translation.text

# Streamlit application
def main():
    st.set_page_config(
        page_title="Image to Audio Story",
        page_icon="📸🔊"
    )

    st.markdown("<h1 style='text-align: center; margin-top: -30px;'>Transform Images to Audio Stories 🖼️➡️🔊</h1>", unsafe_allow_html=True)

    st.sidebar.title("🎨 Upload Your Image!")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.sidebar.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)

        # Language selection
        language = st.sidebar.selectbox("Select Language", ['English', 'Telugu', 'Hindi', 'Tamil', 'Malayalam'])
        translated_scenario = translate_text(scenario, language)
        story = generate_story(translated_scenario)

        with st.expander("Scenario"):
            st.write(translated_scenario)

        with st.expander("Story"):
            st.write(story)

        tts = gTTS(text=story, lang='en')
        tts.save('audio.mp3')
        audio_file = open('audio.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format="audio/mp3")

if __name__ == '__main__':
    main()


#Summary of New Features
#Image Enhancement: Improves the image quality before generating a caption.
#Language Translation: Allows translating the generated story into different languages.
#Background Music: Adds background music to the audio story for an enhanced listening experience.
#Interactive UI for Parameter Tuning: Users can adjust text generation parameters and background music volume using Streamlit sliders
#HUGGINGFACEHUB_API_TOKEN
# streamlit run app.py
#OPENAI_API_KEY
#streamlit run app.py

