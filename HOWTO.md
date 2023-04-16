# How to use the project template

This is just a project template to give you an idea of how your projects could be organized. It is recommended to follow a similar structure like this but you can decide to organize your projects in your own way of course. 

## Usage 

You can follow the instructions below to use this template or click the green `Use this template` button and follow the instructions from [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).

1. Create a new repository for *your project* at [IngolstadtMedicalImaging](https://github.com/IngolstadtMedicalImaging) 
    - Don't initialize it with a README, .gitignore, or license

2. Clone the `Project_Template` repository to your local machine or to a location on the server.
    ```console
    cd path/to/your/projects/
    git clone git@github.com:IngolstadtMedicalImaging/Project_Template.git
    ```

3. Rename the local repository's current `origin` to `upstream`.
    ```console
    git remote rename origin upstream
    ```

4. Give the local repository an `origin` that points to *your repository*.
    ```console
    git remote add origin git@github.com:IngolstadtMedicalImaging/your-repository.git
    ```

5. Push the local repository to *your repository* on Github.
    ```console
    git push origin main
    ```

Your local `origin` should now point to *your repository* and `upstream` points to the original `Project_Template`.


## Project Structure Overview

The project structure of this template is shown as a tree below. You can modify the structure if you want to. 

```bash
├───data                
│   ├───predictions     # you can add your predictions here, e.g. bounding box coordinates 
│   ├───raw             # do not store your images here as they may take too much space, rather add .csv files here (be aware of what you want to upload to Github)
│   └───transformed     # you can add some cleaned .csv files here
├───models              # you can add your trained models here, i.e. your checkpoint .pth files 
├───notebooks           # you can add your prototyping or exploration related .ipynb notebooks here
├───reports             # you can add textual and visual content for your project here, i.e. pdf, latex, jpg, png 
└───scripts             # you can add all your code for your model, datasets, training and inference here 
```