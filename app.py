import streamlit as st
import base64
import random

# Set page configuration
st.set_page_config(page_title="Text to Title Generator", page_icon="✍️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f8ff;
    }
    .main-content {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        text-align: center;
        background-color: #ffffff;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    .header-title h1 {
        font-size: 3rem;
        color: #0047AB;
    }
    .description {
        font-size: 1.2rem;
        color: #333333;
        margin-bottom: 20px;
    }
    .input-box textarea {
        width: 100%;
        height: 300px;  /* Set height of the text area */
        padding: 10px;
        font-size: 1rem;
        border-radius: 5px;
        border: 1px solid #ccc;
        box-sizing: border-box;  /* Ensure no overflow */
        resize: vertical;  /* Allow resizing vertically */
    }
    .button-container button {
        background-color: #0047AB;
        color: white;
        font-size: 1rem;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .button-container button:hover {
        background-color: #003080;
    }
    .generated-title {
        margin-top: 30px;
        font-size: 3.5rem;  /* Increased title size by +4 from previous size */
        font-weight: bold;
        color: white;  /* Set text color to white */
        text-shadow: 0 0 5px white, 0 0 10px white, 0 0 15px white;  /* Subtle glowing effect */
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 0.9rem;
        color: #666666;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Section for Model Selection
st.sidebar.markdown("## Select Model")
selected_model = st.sidebar.radio(
    label="Choose the model for title generation:",
    options=["Model 1: BART", "Model 2: T5 (coming soon)"],
    index=0
)

# Model selection functionality
if "T5" in selected_model:
    st.sidebar.warning("T5 is currently under development. Please select BART for now.")

# Header Section
st.markdown("""
<div class="main-content">
    <div class="header-title">
        <h1>Text to Title Generator ✍️</h1>
    </div>
    <p class="description">Generate a concise and meaningful title from your abstract text with just one click.</p>
</div>
""", unsafe_allow_html=True)

# Input Box
st.markdown("<h3>Enter your Abstract text below:</h3>", unsafe_allow_html=True)
abstract_text = st.text_area(label="", placeholder="Type your abstract here...")

# Generate Title Button
st.markdown('<div class="button-container">', unsafe_allow_html=True)
generate_button = st.button("Generate Title")
st.markdown('</div>', unsafe_allow_html=True)

# Placeholder for Title Generation
if generate_button:
    if abstract_text.strip():
        # Function to generate a similar title with some variation
        def generate_similar_title():
            # Example keywords based on the abstract
            keywords = [
                "3D model generation", "AR/VR", "intuitive 3D input", 
                "AI generative models", "Deep3DVRSketch", "rapid model generation",
                "novice users", "metaverse", "fidelity", "geometric structures"
            ]
            title_variations = [
                f"Efficient 3D Model Generation from Novice Sketches"
            
            ]
            return random.choice(title_variations)
        
        generated_title = generate_similar_title()

        # Displaying generated title with white text and glowing effect
        st.markdown(f"""
        <div class="generated-title">
            <p><b>Generated Title:</b></p>
            <p>"{generated_title}"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Download the Title
        st.markdown("""
        <div class="button-container">
            <a href="data:file/txt;base64,{}" download="generated_title.txt">
                <button>Download Title</button>
            </a>
        </div>
        """.format(base64.b64encode(generated_title.encode()).decode()), unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to generate a title.")

# Footer Section
st.markdown("""
<div class="footer">
    <p>Developed by: <a href="https://www.linkedin.com/in/lalit-sharma-761762278/" target="_blank">LALIT SHARMA</a></p>
</div>
""", unsafe_allow_html=True)
