# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# import uuid

# app = Flask(__name__)

# # Function to load models
# def load_models():
#     models = {}
#     model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
#     for model_file in model_files:
#         model_name = os.path.splitext(model_file)[0]
#         models[model_name] = load_model(os.path.join('models', model_file))
#     return models

# models = load_models()

# # Class names for multi-class classification
# class_names_multiclass = ["Alstonia Scholaris", "Arjun", "jamun"]

# # Threshold for binary classification
# threshold_binary = 0.5
# class_names_binary = ["Arjun", "Alstonia Scholaris"]  # Update with appropriate class names

# # Render index page
# @app.route('/')
# def index():
#     return render_template('index.html', model_names=models.keys())

# # Handle form submission
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         selected_model_name = request.form['model_name']
#         selected_model = models[selected_model_name]

#         # Get file uploaded by user
#         uploaded_file = request.files['file']
        
#         # Ensure file is uploaded
#         if uploaded_file.filename != '':
#             try:
#                 # Save the uploaded file temporarily
#                 filename = str(uuid.uuid4()) + '_' + uploaded_file.filename
#                 uploaded_file.save(filename)
                
#                 # Load and preprocess the image
#                 img = image.load_img(filename, target_size=(150, 150))
#                 img_array = image.img_to_array(img)
#                 img_array = np.expand_dims(img_array, axis=0)
#                 img_array /= 255.0  # Normalize pixel values

#                 # Perform prediction using the selected model
#                 prediction = selected_model.predict(img_array)

#                 # Check if the model is binary or multi-class
#                 if len(prediction[0]) == 1:  # Binary classification
#                     class_name = class_names_binary[0] if prediction[0][0] >= threshold_binary else class_names_binary[1]
#                 else:  # Multi-class classification
#                     class_index = np.argmax(prediction)
#                     class_name = class_names_multiclass[class_index]

#                 return f"Prediction for {uploaded_file.filename} using model {selected_model_name}: {class_name}"
#             except Exception as e:
#                 return f"Error processing image: {e}"
#             finally:
#                 # Remove the temporary file
#                 os.remove(filename)

#         return "No file uploaded"

# if __name__ == '__main__':
#     app.run(debug=True)
