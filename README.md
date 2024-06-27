# gender-and-age

## Introduction
It is a project made with UTKFace dataset and pytorch to predict age and gender of a person and the same is hosted on streamlit


## Installation
```
python -m venv venv
```
```
.\venv\Scripts\activate
```
```
pip install -r requirements.txt
```
if the above does not work..
refer to [this](https://stackoverflow.com/questions/65980952/python-could-not-install-packages-due-to-an-oserror-errno-2-no-such-file-or)

Or,
Go to the ```pyvenv.cfg``` file in your Virtual environment folder and set ```include-system-site-packages = true``` and save the changes and run
```
pip install -r requirements.txt --user
```

Then run 
```
streamlit run realTime.py
```