# Matplot-Sunburn-King-and-Rook-vs.-King-and-Pawn

This is a Python project where an analysis of chess endgames, with a special focus on King and Rook vs. King and Pawn, is performed. Following loading a dataset, the script tallies main statistics and visualizes the game outcomes using Matplotlib and Seaborn. Applying a Random Forest classifier depicts the results of a game; conclusions are illustrated.

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Running the Code](#running-the-code)
- [Editing File Location](#editing-file-location)
- [Code Explanation](#code-explanation)
- [Visualizations](#visualizations)
- [Conclusion](#conclusion)

## Dependencies

This project requires the following Python libraries:

- **numpy**: For numerical operations and data manipulation.
  ```bash
  pip install numpy
pandas: For data analysis and manipulation.

bash
Copy code
pip install pandas
matplotlib: For creating static, animated, and interactive visualizations in Python.

bash
Copy code
pip install matplotlib
seaborn: A data visualization library built on top of Matplotlib that makes drawing attractive statistical graphics easy, using a high-level interface.

bash
Copy code
pip install seaborn
scikit-learn: A machine learning library for Python, providing simple and efficient tools for data mining and data analysis.

bash
Copy code
pip install scikit-learn
tracemalloc: A built-in Python library for tracing memory usage.

bash
Copy code
# Installed by default, so not included here
Installation
Clone the Repository:
Clone this repository to your local machine using the following command:

bash
Copy code
git clone https://github.com/Diocese-trees/Matplot-Sunburn-King-and-Rook-vs.-King-and-Pawn-
Navigate to the Project Directory:
Open your terminal and navigate to the project directory:

bash
Copy code
cd "Chess Endgame Analysis with Matplot Sunburn"
Install the Required Dependencies:
Use pip to install the necessary libraries:

bash
Copy code
pip install -r requirements.txt
Running the Code
Ensure that your dataset file (kr-vs-kp.data) is located in the Chess (King-Rook Vs King-Pawn) folder of the project directory.

Run the Python script with the command:

bash
Copy code
python "discreet dataset matplotlib V2.py"
Editing File Location
To change the file location for the dataset:

Open the Python script in a text editor or IDE, discreet dataset matplotlib V2.py.

Locate the following line in your code:

python
Copy code
data_path = r"C:\Users\adeel\OneDrive\Documents\Internship\Chess Endgame Analysis with Matplot Sunburn\Chess (King-Rook Vs King-Pawn)"
Replace it with your new location for the dataset.

Save the edit and run it again.

Code Explanation
This is an end-game chess position analysis project focusing on the King and Rook against King and Pawn scenario. The script has several key components:

Data Loading
It uses the Pandas library to load the dataset, handling exceptions that may occur during loading to ensure the script runs smoothly.

Data Analysis
The code provides a comprehensive summary of the dataset, including statistical descriptions, class distributions, and frequencies of categorical features. It checks class distribution to evaluate the balance between winning and losing positions.

Visualizations
Various types of visualizations are created based on Matplotlib and Seaborn, such as:

Bar Plots: Count of winning and losing positions.
Pie Charts: Distribution of outcomes in percentage.
Dot Plots: Plot the location of the White King and the result (win or loss) with an additional jitter for better visibility.
Box Plots with Swarm Plots: Summary of the distribution of the White King's position by the outcome.
Model Training
A Random Forest classifier predicts game results based on encoded features of the dataset. The accuracy of the model is evaluated, and a classification report provides performance metrics such as precision, recall, and F1-score.

Findings Documentation
Key insights are documented, focusing on important features and strategies to enhance winning chances in chess endgames.

Memory Usage Monitoring
The script tracks memory usage during execution to ensure efficient resource management.

Visualizations
The following visualizations have been generated:

Bar Plot: Displays the count of winning and losing positions, providing an overview of class distribution.

Pie Chart: Illustrates the distribution of outcomes as a percentage, helping to visualize the proportion of winning positions.

Dot Plot: Depicts the position of the White King against the game outcome, allowing for easy identification of winning and losing positions.

Box Plot with Swarm Plot: Combines summary statistics and individual data points to provide a comprehensive view of the White King's position by outcome.

These visualizations enhance understanding of the dataset's characteristics and illustrate how different features impact a game's outcome.

Conclusion
This project provides a detailed analysis of chess endgame positions and demonstrates how to use Python for data analysis and visualization. The incorporation of machine learning techniques allows for predictive modeling of game outcomes, while comprehensive visualizations facilitate better understanding and strategic insights.
