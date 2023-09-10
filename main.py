import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import openai
import json
import requests
from typing import Dict, List
from streamlit_option_menu import option_menu

from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools  # type: ignore
from langchain.tools import BaseTool

openai.api_key = "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"

def main():
    
    st.title("Agri Aid Expert")
    
    tab1 , tab2 , tab4 , tab5 , tab6 = st.tabs(["**Home**", "**AI Detection**", "**Crop Recommendation**", "**Expert Advice**", "**News**"])
    
    with tab1:
        st.title("Home")
        st.write("**Welcome to 'Agri Aid Expert,' your all-in-one agricultural companion. Our cutting-edge app employs AI technology to identify and combat plant diseases, offering Gen AI solutions for healthier crops. Connect with local agricultural experts, stay informed with the latest news, and make data-driven planting decisions with our ML-powered crop recommendations. Excitingly, we're planning to launch an online selling store for farmers in the future. Join us today to revolutionize your farming experience and unlock the potential of modern agriculture**")
        
    with tab2:
        st.title("AI Detection")
        selected_option = st.selectbox(
        'Select the type of disease you want to detect',
        ('Apple', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato')
        )
        uploaded_file = st.file_uploader("Capture an image...", type="jpg")
        
        result = ""
        
        if st.button("Detect"):
            if selected_option == 'Apple':
                res = detect_apple(uploaded_file)
                result += res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Corn':
                res = detect_corn(uploaded_file)
                result += res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Grape':
                res = detect_grape(uploaded_file)
                result += res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Cherry':
                res = detect_cherry(uploaded_file)
                result += res
                st.write(f"Predicted Class: {res}")
            elif selected_option == 'Pepper':
                st.write(uploaded_file)
                st.write("Pepper is healthy")
            elif selected_option == 'Potato':
                st.write(uploaded_file)
                st.write("Potato is healthy")
            elif selected_option == 'Strawberry':
                st.write(uploaded_file)
                st.write("Strawberry is healthy")
            elif selected_option == 'Tomato':
                st.write(uploaded_file)
                st.write("Tomato is healthy")
           
        if st.button("Get Solution"):
            soln = get_solution(result,selected_option)
            st.write(soln)
         
        
    with tab4:        
        st.title("Crop Recommendation")
        size = st.text_input("Enter the size of your farm")
        climate =  st.text_input("Enter the climate in your area")
        location =  st.text_input("Enter your location")
        budget = st.text_input("Enter the budget you have")
        
        if st.button("Submit"):
            res = crop_recommendation(size , climate , location , budget)
            st.write(res)
        
        
    with tab5:
        st.title("Expert Advice")
        location = st.text_input("Enter your location to get details of expert advice")
        
        if st.button("Search Expert"):
            res = search_expert(location)
            #write the result of the dictionary
            st.markdown("---")
            for i in res:
                st.write("**Title** : " + i['title'])
                st.write("**Link** : " + i['link'])
                st.write("**Snippet** : " + i['snippet'])
                st.markdown("---")      
            
            
    with tab6:
        st.title("News")
        st.write("Explore the latest news in the field of agriculture")
        search = st.text_input("Enter the topic you want to search")
        
        if st.button("Search News"):
            res = search_news(search)
            
            st.markdown("---")
            for i in res:
                st.write("**Title** : " + i['title'])
                st.write("**Link** : " + i['link'])
                st.write("**Snippet** : " + i['snippet'])
                st.markdown("---")   



#crop recommendation function

def crop_recommendation(size , climate , location , budget):
    
    query = "i have a "+size+" acres of land and i have a budget of Rs:"+budget+" in a "+climate+" season in "+location+" which crop is best suited to my field. i need answer in a single word and a brief description about the reason why that crop?"
    
    messages = [
        {"role" : "system", "content" : "You are a kind healpful assistant."},
    ]
    
    messages.append(
            {"role" : "user", "content" : query},
        )
    
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    
    reply = chat.choices[0].message.content
    
    print(reply)
    
    return reply


#expert advice details

def search_expert(location):
    
    url = "https://google.serper.dev/search"

    payload = json.dumps({
    "q": "agricultural experts addresses in "+location,
    })
    headers = {
    'X-API-KEY': 'f28228b2388bb54e182961f3025c345a9a96c581',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    res = json.loads(response.text)
    
    return res['organic']


    
#news search in the field of agriculture

def search_news(search):
    
    url = "https://google.serper.dev/news"

    payload = json.dumps({
    "q": "agricultural experts addresses in "+search,
    })
    headers = {
    'X-API-KEY': 'f28228b2388bb54e182961f3025c345a9a96c581',
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    
    res = json.loads(response.text)
    
    return res['news']


def detect_apple(uploaded_image):
    
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\apple.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)

        # Define the class labels
        class_labels = ["Apple scab", "Apple Black rot", "Apple rust", "Healthy apple leaf"]

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class

def detect_corn(uploaded_image):
    
    if uploaded_image is not None:
        # Load your trained model
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\corn.h5')

        # Perform classification when the user uploads an image
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Perform the prediction
        predictions = model.predict(img_array)

        # Define the class labels
        class_labels = ["Cercospora leaf spot Gray leaf spot", "Common rust", "Northern Leaf Blight", "Healthy corn leaf"]

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Get the predicted class label
        predicted_class = class_labels[predicted_class_index]

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class label
        return predicted_class
        
def detect_grape(uploaded_image):
        
        if uploaded_image is not None:
            # Load your trained model
            model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\grape.h5')
    
            # Perform classification when the user uploads an image
            img = image.load_img(uploaded_image, target_size=(64, 64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize pixel values
    
            # Perform the prediction
            predictions = model.predict(img_array)
    
            # Define the class labels
            class_labels = ["Black rot", "Esca (Black Measles)", "Leaf blight (Isariopsis Leaf Spot)", "Healthy grape leaf"]
    
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions)
    
            # Get the predicted class label
            predicted_class = class_labels[predicted_class_index]
    
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
            # Display the predicted class label
            return predicted_class
            
def detect_cherry(uploaded_image):
        
    
    if uploaded_image is not None:
        
        model = tf.keras.models.load_model(r'E:\Agri_Aid_Expert\models\cherry.h5')
        
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        predictions = model.predict(img_array)
        
        if predictions[0][0] == 1:
            predicted_class = "Cherry healthy"
        else:
            predicted_class = "Cherry Powdery Mildew disease"
            
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        return predicted_class
    
def get_solution(res,plant):
    
    query = "my "+plant+" plant or crop has "+res+" disease"
    # Create a new instance of the OpenAI class
    llm = OpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        max_tokens=200,
        temperature=0,
        client=None,
        model="text-davinci-003",
        frequency_penalty=1,
        presence_penalty=0,
        top_p=1,
    )

    # Load the tools
    tools: List[BaseTool] = load_tools(["google-serper"], llm=llm)

    # Create a new instance of the AgentExecutor class
    agent: AgentExecutor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # Create the template
    template = """{topic}. Give me solution for how to solve it in a brief description."""

    # Generate the response
    response: str = agent.run(template.format(topic=query))


    return response


if __name__ == "__main__":
    main()