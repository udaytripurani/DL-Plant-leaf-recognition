# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
# import numpy as np
# import os
# import uuid

# app = Flask(__name__)

# # Function to load models
# def load_models():
#     models = {}
#     model_files = [f for f in os.listdir('models') if f.endswith('.h5')]
#     for model_file in model_files:
#         try:
#             model_name = os.path.splitext(model_file)[0]
#             model_path = os.path.join('models', model_file)
#             print(f"Loading model: {model_name} from {model_path}")
#             models[model_name] = load_model(model_path)
#             print(f"Model {model_name} loaded successfully.")
#         except Exception as e:
#             print(f"Error loading model {model_name}: {e}")
#     return models

# models = load_models()

# # Class names for multi-class classification
# class_names_multiclass = ["Alstonia Scholaris", "Arjun", "Jamun"]

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
                
#                 # Load and preprocess the image based on the selected model
#                 if selected_model_name == 'my_vgg_model':
#                     img = image.load_img(filename, target_size=(224, 224))  # Resize to 224x224 for VGG model
#                     img_array = image.img_to_array(img)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array = preprocess_input_vgg(img_array)  # VGG preprocessing
#                 elif selected_model_name == 'my_rnn_model':
#                     img = image.load_img(filename, target_size=(150, 150))  # Resize to 150x150 for RNN model
#                     img_array = image.img_to_array(img)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array /= 255.0  # Normalize pixel values
#                     img_array = img_array.reshape(img_array.shape[0], -1, 1)  # Reshape for RNN
#                 else:
#                     # For other models, resize to 150x150 and normalize
#                     img = image.load_img(filename, target_size=(150, 150))
#                     img_array = image.img_to_array(img)
#                     img_array = np.expand_dims(img_array, axis=0)
#                     img_array /= 255.0  # Normalize pixel values

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
