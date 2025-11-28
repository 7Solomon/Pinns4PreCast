# Folder Structure
    - src/
        - blueprints/
            "Endpoint definition for the Backend "
        - DeepONet/
            "Everything for the the Neural Network part"
        - state_management/
            "the definition of the State() class and its base settings"
        - class definition 
            "Baseclases for ease of use"

    - content/
        "all generated files"
    - problem_defenition.py
        "physical problem definition, all PDE and Boudary definitions"
    - run.py 
        "entry for the programm"
    - .curent_state.json
        "HERE the files that are curently set that contain the info for the config is set"
    
    - static/
    - Templates/




# Data Pipeline
The data Pipiline gets started whena user creates a new run on the runs HTML page, in this page the calls are all to src/blueprints/api.py.

    - Define a model structure and physical problem in calling the api endpoint "api.define_model"
    - Define Training Pipeline in calling the "api.define_training_pipeline"
    - Train a model with calling "api.train", in the training there is every 10 epochs a graph plotted that shows the temp and alpha at defined Sensor Points


# State Management 
In the HTML page "Setup" one can add and create new or edit Configs that contain the material Data, the domain Data and the Training variables. 
They get saved inside a folder "content/states" and these configs are set over the complete programm and every where State() is called it takes these set Configs.



