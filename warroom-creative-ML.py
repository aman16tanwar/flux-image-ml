

import os
import requests
import streamlit as st
import replicate

# Set the Replicate API token
os.environ["REPLICATE_API_TOKEN"] = ""

# Define available models with trigger words
models = {
    "Banff Christmas 2024": {
        "id": "war-room-inc/banff-christmas-2024:e7eac582c5cf62327182007e0ad0ce5baee3f593f20929b689120a7d30ddcb6d",
        "trigger": "bllt"
    },
    "Backcountry Biking": {
        "id": "war-room-inc/backcountry_biking:deb62af4cb70d9b9a34fb1cfb69bf1d5ec2a7e39ea78f4c9c2ebf500eff039de",
        "trigger": "backcountry_biking"
    },
   
}

# Streamlit App Title and Subheader
st.title("Kedet AI Creative Generator")
st.subheader("Generate custom AI images for the brand")

# Model selection with trigger word instructions
model_choice = st.selectbox("Choose a model:", list(models.keys()))
trigger_word = models[model_choice]["trigger"]
st.write(f"**Note:** Include the trigger word '{trigger_word}' in your prompt to guide the model output.")

# Prompt Input
prompt = st.text_input("Enter your prompt:", placeholder=f"Include '{trigger_word}' in your description...")

# Additional Parameters
num_outputs = st.slider("Number of outputs", 1, 5, 3)
guidance_scale = st.slider("Guidance scale", 1.0, 10.0, 3.5)
aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "4:3"], index=0)

# Initialize session state for outputs
if "outputs" not in st.session_state:
    st.session_state.outputs = []

# Button to Generate Images
if st.button("Generate"):
    if prompt.strip() == "":
        st.error("Please enter a prompt to generate images.")
    elif trigger_word not in prompt.lower():
        st.error(f"Please include the trigger word '{trigger_word}' in your prompt.")
    else:
        with st.spinner("Generating..."):
            try:
                # Run the selected model with the user-provided prompt and parameters
                output_urls = replicate.run(
                    models[model_choice]["id"],
                    input={
                        "model": "dev",
                        "prompt": prompt,
                        "lora_scale": 1,
                        "num_outputs": num_outputs,
                        "aspect_ratio": aspect_ratio,
                        "output_format": "jpg",
                        "guidance_scale": guidance_scale,
                        "output_quality": 90,
                        "prompt_strength": 0.8,
                        "extra_lora_scale": 1,
                        "num_inference_steps": 28,
                        "disable_safety_checker": True
                    }
                )
                st.session_state.outputs = output_urls
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display and download generated images
if st.session_state.outputs:
    for idx, image_url in enumerate(st.session_state.outputs):
        response = requests.get(image_url)
        if response.status_code == 200:
            st.image(response.content, caption=f"Output {idx + 1}", use_column_width=True)
            st.download_button(
                label=f"Download Output {idx + 1}",
                data=response.content,
                file_name=f"output_{idx + 1}.jpg",
                mime="image/jpeg"
            )
        else:
            st.error(f"Failed to load image {idx + 1}")
