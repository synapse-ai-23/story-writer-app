from transformers import pipeline
import streamlit as st
from PIL import Image
import io
import requests

logo = Image.open("logo.jpg")

st.image(logo, width=100)
st.markdown("<h2 style='text-align: center; color: grey;'>Create magical stories from your images</h2>", unsafe_allow_html=True)


def generate_image_description(image):
    model = pipeline("image-to-text", model = "nlpconnect/vit-gpt2-image-captioning")
    description = model(image)
    return description[0]['generated_text']

def generate_story(description):
    story_model = pipeline("text-generation", model="gpt2")
    prompt = f"create a story, based on the following description: {description}"
    story = story_model(prompt, max_length=500)
    return story[0]['generated_text']

def main():
    st.title("Personalized Storybook Creator")
    
    st.write("Upload images to create a personalized storybook.")
    st.write("Utilizing the OpenAI GPT2")
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        descriptions = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            description = generate_image_description(image)
            descriptions.append(description)
            st.write(f"Description: {description}")
        
        if st.button("Generate Story"):
            combined_descriptions = " ".join(descriptions)
            story = generate_story(combined_descriptions)
            st.write("Generated Story:")
            st.write(story)

    st.write("Built by synapse.ai")

if __name__ == "__main__":
    main()

