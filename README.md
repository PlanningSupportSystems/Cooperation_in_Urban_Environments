# Simulations-of-Urban-Environments

This text is just a temporary description:

Agent-Based Modeling (ABM) Urban Simulation Tool

Introduction:

Welcome to the Agent-Based Modeling (ABM) simulation tool, developed as part of a research collaboration between the Centro de Investigação do Território Transportes e Ambiente, Universidade do Porto, Portugal (CITTA | FEUP) and the Universidade Federal Fluminense (UFF) - Programa de Pós-Graduação em Arquitetura e Urbanismo (PPGAU) . This tool is designed to facilitate the study of urban dynamics by simulating interactions between agents and places within a city environment.

Objective:

The primary aim of this tool is to analyze the impact of urban morphology on the evolution of urban systems. By simulating the interactions between agents (representing individuals or entities) and places (representing locations within the city), researchers can investigate how the shape of cities influences the distribution and clustering of traits among agents and places. This research aims to explore how urban morphology shapes movement patterns, trait distribution, and the emergence of clusters within urban environments.

What is ABM?

Agent-Based Modeling (ABM) is a computational modeling technique used to simulate the actions and interactions of autonomous agents in a dynamic environment. In ABM, agents are entities with individual behaviors and decision-making capabilities, and their interactions with each other and the environment lead to emergent phenomena.

Entropy Calculation:

Entropy is a measure of disorder or randomness in a system. In the context of ABM simulations, entropy calculation is used to assess the diversity and distribution of traits among agents and places over time. By analyzing entropy, researchers can gain insights into the complexity and dynamics of urban systems.

About the Program:

Programming Language and Framework:

The program is written in C++ using the Qt framework for the user interface (UI). It utilizes libraries such as GDAL, Gnuplot, and OpenCV for various functionalities.

GDAL (Geospatial Data Abstraction Library):

GDAL is a library used for reading and writing raster and vector geospatial data formats. It allows the program to handle shapefiles, which are commonly used to represent geographic features.

OpenCV (Open Source Computer Vision Library):

OpenCV is a library of programming functions mainly aimed at real-time computer vision. It provides tools for image processing and computer vision tasks, which are utilized for various image-related operations in the program.

Shapefile Input: Load a shapefile representing the city environment. Shapefiles are widely used for storing vector GIS (Geographic Information System) data and consist of multiple files representing different aspects of geographic features (e.g., points, lines, polygons).

Land Use Label: Specify the land use label within the shapefile for calculating traits of each place. The selected label defines the attribute used to categorize different land uses within the city.

Agent Traits: Define random traits for agents, influencing their behavior during the simulation. Agent traits represent individual characteristics or attributes that influence agent decision-making.

Contamination: Set contamination proportions affecting agents and places during the simulation. Contamination parameters (gamma for agents, theta for places) control the extent of interaction and influence between agents and places.

Alpha and Beta Parameters: Adjust parameters controlling agent behavior during movement and destination selection. Alpha represents the openness of agents to different traits, while beta controls the weight of distance in destination selection.

Number of Iterations: Define the number of simulation iterations to run. Each iteration represents a step in the simulation where agents interact with places and move within the city environment.

Working Folder: Specify a folder to save simulation results, graphs, and other output data. All generated files, including images and text data, will be saved to this folder.

Batch Simulation: Explore a range of alpha and beta values systematically by running batch simulations. Batch simulations allow for the analysis of multiple scenarios and parameter combinations.

Heatmaps: Visualize average entropy values for agents and places as heatmaps. Heatmaps provide insights into the distribution and diversity of traits within the simulated urban environment.

Load Node File: Load pre-processed node file data to optimize simulations. Node files contain processed node data derived from shapefiles, allowing for faster simulation initialization without reprocessing the map.

Graphical Visualizations: Generate graphical representations of simulation results for analysis and visualization purposes. Graphs and plots provide visual insights into the dynamics and behavior of agents and places over time. The ABM simulation tool serves as a research instrument for studying urban dynamics, allowing researchers to analyze the interactions between agents and places in simulated city environments. By leveraging ABM techniques and entropy calculation, this tool facilitates the exploration of complex urban systems and emergent phenomena. By providing a user-friendly interface and comprehensive features, the ABM simulation tool empowers researchers to gain valuable insights into urban dynamics and contribute to the advancement of urban studies and planning.
