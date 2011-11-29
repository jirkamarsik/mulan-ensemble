from matrix2latex import matrix2latex
from glob import glob

results = dict()

for f in glob('tmc2007-500-small-*.txt'):
    method = f[18:-4]
    results[method] = dict()
    for line in open(f):
        if line != '':
            measure, _, value_pair = line.partition(': ')
            mean, _, _ = value_pair.partition('±')
            results[method][measure] = mean


table1 = list()
methods = ['rakel', 'clr', 'mlknn', 'homer', 'iblr',
           'intersection', 'union', 'majority']
measures = ['Hamming Loss', 'Subset Accuracy',
            'Example-Based Recall', 'Example-Based Accuracy',
            'Micro-averaged Precision', 'Micro-averaged Recall',
            'Micro-averaged F-Measure']

for method in methods:
    row = list()
    for measure in measures:
        row.append(results[method][measure])
    table1.append(row)

matrix2latex(table1, 'bipartition-table',
             columnLabels = methods, rowLabels = measures)

table2 = list()
methods = ['rakel', 'clr', 'mlknn', 'homer', 'iblr',
           'minimum', 'maximum', 'mean', 'topk']
measures = ['Average Precision', 'Coverage', 'OneError', 'IsError',
            'ErrorSetSize', 'Ranking Loss', 'Micro-averaged AUC']

for method in methods:
    row = list()
    for measure in measures:
        row.append(results[method][measure])
    table2.append(row)

matrix2latex(table2, 'score-table',
             columnLabels = methods, rowLabels = measures)
