# FIT.LY
## Inspiration
Recently getting into weight training and starting to lift more, we came to a realization that many people are intimidated or afraid to start exercising. Exercise and fitness is an essential part of maintaining one's physical and mental well-being, but the motivation to start is often not there or could seem out of reach. Fit.ly is designed to welcome people to start living healthier lifestyles and begin their fitness journey. 

## What it does
There are two essential components of Fit.ly, a rep tracker and a workout generator. The Rep tracker uses movement and pose detection to count reps, ensuring a correct and full range of motion on each exercise. The second feature is a workout generator. Many people don't stick to a plan or find it hard to get a good workout plan. Listening to people online for guidance is useful but often overwhelming. Truth is, each person's workout plan should be tailored to their goals and circumstances. The workout generator allows users to input several parameters to generate a personalized seven-day workout plan. 

## How we built it
The entire application is built with Streamlit, including all forms, buttons, inputs, video outputs, and more. Everything was written with python, with Jupyter notebooks exported into python scripts. 

The rep counter was built off of a foundational YOLv8Pose model, fine-tuning was used on another dataset but was not used due to little improvement in accuracy. We then implemented a non-maximum suppression and used both confidence and filter by area to ensure only the subject will be counter. This ensures that people in the background will not interfere with the rep count. In order to count reps, we determined the angle between different keypoints. For instance, we used the shoulder, elbow and hand to determine what state of a curl they are in. We also didn't count bad reps such as a bent arm in lateral raises using the same idea. Angles can be tweaked to give more or less leeway. 

The workout generator is a PyTorch-loaded pre-trained GEMMA 2b-it model. We then scraped https://www.muscleandstrength.com/workout-routines for workouts using regex and used LORA to fine-tune the model. Then feeding the parameters from the streamlit app, we called an inference function that created the workout plan. 

## Challenges we ran into
The biggest challenges faced throughout this were the package dependency errors. Many packages had conflicting versions and as we were running this locally, many issues especially with PyTorch and OpenCV started to occur. In the end, we used a virtual environment for all of our installations. 

We also never used Streamlit prior to this project, but although we found it very straightforward to use, there was still a learning curve to get used to Streamlit as well as figure out how the async works. We looked up tutorials online and read the Streamlit documentation to ultimately complete a project we are proud of. 

## Accomplishments that we're proud of
We are proud of the completeness of the final product, as in prior hackathons we have only used notebooks and inferences as submissions. This time, we were able to complete a web app. 

## What we learned
Throughout this experience, we learned to use Streamlit, opening the doors for future web app development and integration of AI into websites. We also gained further experience working with YOLO models, fine-tuning NLP models, and web scraping. Prior to the hackathon, we have had little experience with web scraping, this experience taught us to use the beautiful-soup package while parsing HTML files. 

## What's next for FIT.LY
The possibilities are endless for Fit.ly, we plan on integrating more exercises and more input over the workout generator. We believe having the workout generator automatically formatted into a calendar could be more intuitive and easier to read. On top of that, we plan on creating a meal plan generator as well for those wishing to either bulk or cut, they can not only have a customized workout plan but also a customized meal plan. Our vision for Fit.ly is to turn it into a community and a platform for people of diverse backgrounds to connect through fitness. Adding posts, workout suggestions, and even hosting live guided courses, the future for Fit.ly is a bright one. 
