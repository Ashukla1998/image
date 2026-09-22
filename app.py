import base64
from io import BytesIO
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Image Generator", page_icon="🎨")
st.title("🎨 AI Image Generator")

# 1. Initialize session state for image history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display previous history on every rerun
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.image(msg["image_bytes"], caption=msg["prompt"])
            st.download_button(
                label="📥 Download Image",
                data=msg["image_bytes"],
                file_name=f"generated_image_{i}.png",
                mime="image/png",
                key=f"download_{i}",
            )

# 3. Handle user prompt
if prompt := st.chat_input("Describe the image you want to generate..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Generate image
    with st.chat_message("assistant"):
        with st.spinner("Generating your image..."):
            try:
                response = client.images.generate(
                    model="gpt-image-1",
                    prompt=prompt,
                    size="1024x1024",
                    quality="auto",
                    n=1,
                )

                item = response.data[0]

                # Handle Base64 response (gpt-image-1) or URL response (dall-e)
                if getattr(item, "b64_json", None):
                    image_bytes = base64.b64decode(item.b64_json)
                elif getattr(item, "url", None):
                    image_bytes = requests.get(item.url).content
                else:
                    raise ValueError("No valid image data or URL received from API.")

                # Render the image and download button
                st.image(image_bytes, caption=prompt)
                st.download_button(
                    label="📥 Download Image",
                    data=image_bytes,
                    file_name="generated_image.png",
                    mime="image/png",
                    key=f"download_{len(st.session_state.messages)}",
                )

                # Store image bytes in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "image_bytes": image_bytes,
                    "prompt": prompt,
                })

            except Exception as e:
                st.error(f"Failed to generate image: {e}")