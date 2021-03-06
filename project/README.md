# Netflix Movie Recommendation

![alt text](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/scalable.PNG)

This project fuses collaborative filtering and content-based recommendation for netflix movie recommendation. You can view our report at [Project_Scalable.pdf](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/Project_Scalable.pdf).

## Project Structure
The structure of the project is as follows:

1. [app.py](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/app.py) runs the flask GUI interface
2. [suggestion_model.py](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/suggestion_model.py) is the source for suggesting movies based on movie summaries
3. [SBERT_embedding.ipynb](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/SBERT_embedding.ipynb) generate movies summaries embeddings using SBERT
4. [BT_filtered.py](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/BT_filtered.py) filters the watch list with the only the movies present in the netflix dataset (2000-2015)
5. [train_CNF.ipynb](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/train_CNF.ipynb) trains the NCF network for collaborative filtering

## Libraries Required
The project makes use of the following libraries:
1. flask
2. tensorflow
3. sentence_transformers
4. scipy
5. pandas
6. numpy

## Running the code
To run the code, you just need to run:

```python flask run```

## Example output

![alt text](https://github.com/alishibli97/Scalable-Machine-Learning-and-Deep-Learning/blob/main/project/project.png)
