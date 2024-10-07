import pandas as pd

def make_submission_file(filename, probab, test_id, IdCol, targetCol, threshold=None):
    submit = pd.DataFrame()
    submit[IdCol] = test_id
    submit[targetCol] = probab
    if threshold!=None:
        pred = [True if x>=threshold else False for x in probab]
        submit[targetCol] = pred
    submit.to_csv(filename, index=False)
    return submit