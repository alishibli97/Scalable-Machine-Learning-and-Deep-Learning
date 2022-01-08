# Netflix Movie Recommendation



The structure of the project is as follows:

1. app.py runs the flask GUI interface
2. suggestion_model.py is the source for suggesting movies based on movie summaries
3. SBERT_embedding.ipynb generate movies summaries embeddings using SBERT
4. BT_filtered.py filters the watch list with the only the movies present in the netflix dataset (2000-2015)
5. train_CNF.ipynb trains the NCF network for collaborative filtering


The project makes use of the following libraries:
1. flask
2. tensorflow
3. sentence_transformers
4. scipy
5. pandas
6. numpy

To run the code, you just need to run:
```python flask run```
