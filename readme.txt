Step 1 - Install virtual environment if it's not present else you can skip this step
pip install virtualenv
Step 2 - Create virtual environment in project base directory where the app.py file is present
virtualenv env
Step 3 - execute the command below(Navigate to Scripts folder and activate)
# cd env/Scripts
# activate
Step 4 - Return back to base directory 
# cd ../..
Step 5 - Install the required libraries 
# pip install -r requirements.txt

Step 6 - Run demo.py
streamlit run demo.py
