from re import A
import numpy as np

from pathlib import Path

import argparse
import optuna
import json

def simple_cull(inputPoints):
    def dominates(row, candidateRow):
        return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(row)    
    
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    return paretoPoints, dominatedPoints


def select_pareto(df, paretoSet):
    arr = np.array([[x1, x2] for x1, x2 in list(pareto)])
    return df[(df.values_0.isin(arr[:, 0])) & (df.values_1.isin(arr[:, 1]))]

if __name__ == "__main__":
    study_names = ['MFVI']


    path = Path("results/ncmapss/studies")
    path.mkdir(exist_ok=True)

    for study_name in study_names:

        sampler = optuna.samplers.RandomSampler()
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name=study_name, 
            sampler=sampler,
            storage="sqlite:///"+path.as_posix()+"/optimization.db",
            load_if_exists=True,
        )

        df = study.trials_dataframe()
        df.dropna(axis=0, inplace=True)
        pareto, dominated = simple_cull(df[['values_0', 'values_1']].values.tolist())
        pareto = select_pareto(df, pareto)

        p = Path('results/ncmapss/best_models/', study_name)
        p.mkdir(exist_ok=True)
        pareto.sort_values('values_0', inplace=True)
        pareto.reset_index(drop=True, inplace=True)
        for i, row in pareto.iterrows():
            n = row['number']
            #string = json.dumps(, indent=4)
            with open(Path(p, f'{i:03d}.json').as_posix(), 'w') as f:
                params = study.trials[n].params
                params['value_0'] = study.trials[n].values[0]
                params['value_1'] = study.trials[n].values[1]
                json.dump(params, f, indent=4)