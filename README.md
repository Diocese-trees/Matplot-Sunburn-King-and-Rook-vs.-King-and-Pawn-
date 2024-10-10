# Matplot-Sunburn-King-and-Rook-vs.-King-and-Pawn-
This is a Python project where analysis of chess endgames with special focus on King and Rook vs. King and Pawn is performed. Following loading a dataset, the script tallies main statistics and visualizes the game outcomes by using Matplotlib and Seaborn. Applying a Random Forest classifier depicts the results of a game; conclusions are illustrated

# Chess Endgame Analysis with Matplot Sunburn

This Python script was developed with the help of VS Code.

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

- **numpy**: For numerics and data manipulation.
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
seaborn: A data visualization library built on top of Matplotlib that makes drawing attractive statistical graphics very easy, using a high-level interface.
 
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
Check that your dataset file (kr-vs-kp.data) is in the Chess (King-Rook Vs King-Pawn) folder of the project directory.

Run the Python script with the command:

bash
Copy code
python "discreet dataset matplotlib V2.py"
Editing File Location
To change the file location for the dataset:

Open the Python script in a text editor or IDE, discreet dataset matplotlib V2.py.

For this line in your code

python
paste code
data_path = r"C:\Users\adeel\OneDrive\Documents\Internship\Chess Endgame Analysis with Matplot Sunburn\Chess (King-Rook Vs King-Pawn)"
Replace with your new location for your dataset.

Save the edit and run it again.

Code Explanation
This is an end-game chess position analysis project. The type of programming script under consideration is a particular one: King and Rook against King and Pawn. It has some key components:

Data Loading
It uses the Pandas library in loading the dataset, with several features representing the chess positions. Thus, handling possible exceptions that can occur during the loading, so this script won't crash unexpectedly.

Data Analysis
The code provides an overall summary of the dataset, like statistics descriptions, class distribution, and the frequency of categorical features. It checks up the class distribution for the balance in winning and losing positions.

Visualizations
Various types of visualizations are created based on Matplotlib and Seaborn, such as:
    Bar Plots: Count of winning and losing positions
    Pie Charts: Distribution of outcomes in percentage.
Dot Plots: Plot the location of the White King and the result (win or loss) with an additional jitter for better resolution
Box Plots with Swarm Plots: The distribution of the White King's position is summarized by the outcome
To predict game results based on encoded features of the dataset, a Random Forest classifier is used. The accuracy of the model checks and gives a classification report for performance metrics such as precision, recall, and F1-score.
Documentation of Findings
Here are key insights arising from the analysis, focusing on important features and strategies that would enhance chances to win a chess endgame like this.
Monitoring Memory Usage
It maintains record usage in terms of memory at run-time such that resources are utilized efficiently.
Visualizations
The following visualizations have been generated:
 
Bar Plot: This bar plot is used to put forth a count of winning and losing positions, hence serving as an overview of the class distribution.
Pie Chart: It also enlightens about the outcomes in terms of percentage and helps in deducing the distribution of winning positions.

Dot Plot: Here, one can easily depict which of the positions corresponds to winning or losing as illustrated by the positioning of the White King relative to the outcome of the game.

Box Plot with Swarm Plot: Summary statistics are combined with individual points in order to produce a combined view of the positioning of the White King by outcome.

These visualizations not only enhance characteristics of the dataset but also portray how varied features impact a game's success.
