Website Url


Machine Learning Project Proposal

Official Title: 
Predicting Game Outcomes of League of Legends Based on Input Champion Selection

Project Code Name:
League of Legends Synergy Predictor

Group Members: Griffin Barnard, Simon Browning, Kaden Vasquez

Research Question: Can we accurately predict which team wins a League of Legends game based on both team’s champions and the match rank?

Programming Language and libraries: 
Data Sciencce: Python, Pandas, NumPy, Sci-Kit Learn,  
Web Development: JavaScript, ReactJS, Flask

Data Set: https://developer.riotgames.com/apis
We will use data from the Riot Games API. This will happen by getting many users per in game tier. Then getting a few matches per user. Then we collect all these matches and pull the relevant information from the riot API. So we will have a set of many different matches and their information, from which we can extract relevant features for our model.

Machine Learning methods: 
We will start by training a neural network
We are in the process of researching if Decision Trees or Random Forrest would be a superior option.

Validation: Initially, we will implement a standard test train split to validate our data. However, since we are acquiring new useful data every day, we can eventually test our algorithm on current data.

Final Product:  
	Our final product will be a website with a champion selection tool. Once the user selects the champions of both teams and the rank of the game,  the website will then give a prediction as to which team will become victorious. Ideally, the website will also give recommendations as to which champions would be best to select as you are finishing the team creation.
 

##Milestones and Timeline: 

Milestone
Description
Completion Date
Griffin’s Deliverable
Simon’s Deliverable
Kaden’s Deliverables
1
Getting and cleaning data
April 8th
Acquiring data from Riot Games website
Importing data into Python using Pandas
One-hot encoding the categorical data
2
Determine best training method
April 22nd
Begin Early Website Programming
Create model in Python using Random Forest and one using Neural Networks
Create model in Python using Random Forest and one using Neural Networks
3
Website completed
May 3rd
Finalized Website Programming
Continue investigating model to find further relevant information to be extracted
Create icon and User Interface elements for website



Github link: https://github.com/griffinbar123/SyngergyBuilder


