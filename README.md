🍎 Fruit Classification using PyTorch & Streamlit

A deep learning-based fruit classification application built with PyTorch for training and Streamlit for deployment.
The model is trained on a custom dataset containing multiple fruit categories, and can predict the fruit type from an uploaded image.

📌 Features                                                                                                                                                                                                          
✅ Image classification using a CNN model in PyTorch                                                                                                                                                                 
✅ Dataset organized into subfolders by class name for training & validation                                                                                                                                         
✅ Real-time image upload and prediction via Streamlit                                                                                                                                                               
✅ GPU support for faster training (if available)                                                                                                                                                                    
✅ End-to-end workflow from training to deployment                                                                                                                                                                   

| Uploaded Image | Model Prediction |
| -------------- | ---------------- |
| 🍌 Banana      | **Banana**       |
| 🍎 Apple       | **Apple**        |

🛠 Tech Stack                                                                                                                                                                                                        
✅ Python                                                                                                                                                                                                            
✅ PyTorch                                                                                                                                                                                                           
✅ TorchVision                                                                                                                                                                                                       
✅ Streamlit                                                                                                                                                                                                         
✅ PIL / OpenCV                                                                                                                                                                                                      

Fruit-Classification/                                                                                                                                                                                                
│                                                                                                                                                                                                                    
├── Training/               # Dataset (subfolders per fruit class)                                                                                                                                                   
├── model.pth               # Saved trained model                                                                                                                                                                    
├── train.py                 # Training script                                                                                                                                                                       
├── app.py                   # Streamlit app for prediction                                                                                                                                                          
├── requirements.txt         # Dependencies                                                                                                                                                                          
└── README.md                # Project documentation                                                                                                                                                                 





🚀 Installation & Usage                                                                                                                                                                                              
1️⃣ Clone the Repository                                                                                                                                                                                              
git clone https://github.com/yourusername/Fruit-Classification.git                                                                                                                                                   
cd Fruit-Classification                                                                                                                                                                                              

2️⃣ Install Dependencies                                                                                                                                                                                              
pip install -r requirements.txt                                                                                                                                                                                      

3️⃣ Train the Model                                                                                                                                                                                                   
python train.py                                                                                                                                                                                                      

4️⃣ Run the Streamlit App                                                                                                                                                                                             
streamlit run app.py                                                                                                                                                                                                 
