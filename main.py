import pulp
import numpy as np


def read_file(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    data = [x.split(';') for x in data][1:]
    dmus = []
    values = []
    for x in data:
        dmus.append(x[0])
        values.append([float(v) for v in x[1:]])
    return dmus, values


dmus, inputs = read_file("crossinputs.csv")
_, outputs = read_file("crossoutputs.csv")
# dmus, inputs = read_file("inputstest.csv")
# _, outputs = read_file("outputstest.csv")
# dmus, inputs = read_file("hcuinput.csv")
# _, outputs = read_file("hcuoutput.csv")
# inputs = read_file("hcuinput.csv")
# outputs = read_file("outputstest.csv")

inputs = np.array(inputs)
outputs = np.array(outputs)
dmu_count = len(dmus)
input_count = inputs.shape[1]
output_count = outputs.shape[1]


def make_affine(param, values):
    return pulp.LpAffineExpression(list(zip(param.values(), values)))


def dmu_problem_effectivenes(dmu, inputs, outputs):
    dmu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMaximize)

    v = pulp.LpVariable.dicts("v", list(range(1, input_count + 1)), 0)
    u = pulp.LpVariable.dicts("u", list(range(1, output_count + 1)), 0)

    dmu_problem += pulp.LpAffineExpression(list(zip(u.values(), outputs[dmu, :])))
    dmu_problem += pulp.LpAffineExpression(list(zip(v.values(), inputs[dmu, :]))) == 1

    for x in range(dmu_count):
        dmu_problem += make_affine(u, outputs[x, :]) <= make_affine(v, inputs[x, :])

    dmu_problem.solve()
    for v in dmu_problem.variables():
        print(v.name, "=", v.varValue)
    return dmu_problem

def super_effectivenes(dmu, inputs, outputs):
    dmu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMaximize)

    v = pulp.LpVariable.dicts("v", list(range(1, input_count + 1)), 0)
    u = pulp.LpVariable.dicts("u", list(range(1, output_count + 1)), 0)

    dmu_problem += make_affine(u, outputs[dmu, :])
    dmu_problem += make_affine(v, inputs[dmu, :]) == 1

    for x in range(dmu_count):
        if x != dmu:
            dmu_problem += make_affine(u, outputs[x, :]) <= make_affine(v, inputs[x, :])

    dmu_problem.solve()
    return dmu_problem.objective.value()

def optimal_vector(dmu, eff, inputs, outputs):
    dmu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMinimize)

    v = pulp.LpVariable.dicts("v", list(range(1, input_count + 1)), 0)
    u = pulp.LpVariable.dicts("u", list(range(1, output_count + 1)), 0)

    dmu_problem += make_affine(u, np.vstack([outputs[:dmu], outputs[dmu+1:]]).sum(0))
    dmu_problem += make_affine(v, np.vstack([inputs[:dmu], inputs[dmu+1:]]).sum(0)) == 1

    for x in range(dmu_count):
        if x != dmu:
            dmu_problem += make_affine(u, outputs[x, :]) <= make_affine(v, inputs[x, :])

    dmu_problem += make_affine(u, outputs[dmu, :]) == eff * make_affine(v, inputs[dmu, :])

    print(dmu_problem)
    dmu_problem.solve()
    for v in dmu_problem.variables():
        print(v.name, "=", v.varValue)
    return dmu_problem.objective.value()


def solve_hcu(dmu, inputs, outputs):
    hcu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMinimize)

    theta = pulp.LpVariable("theta", 0)
    lambdas = pulp.LpVariable.dict("lambda", list(range(1, dmu_count + 1)), 0)

    hcu_problem += theta

    for i in range(input_count):
        hcu_problem += make_affine(lambdas, inputs[:, i]) <= theta * inputs[dmu][i]

    for i in range(output_count):
        hcu_problem += make_affine(lambdas, outputs[:, i]) >= outputs[dmu][i]

    hcu_problem.solve()
    lambda_values = ([(lambda_var.name[7:], lambda_var.varValue) for lambda_var in list(lambdas.values())])
    lambda_values.sort(key=lambda x: int(x[0]))
    lambda_values = np.array([x[1] for x in lambda_values])
    xhcu = (inputs.T * lambda_values).sum(1)
    return hcu_problem.objective.value(), xhcu


np.set_printoptions(precision=2)

# solution_hcu = {}
# for i, dmu in enumerate(dmus):
#     solution_hcu[dmu] = solve_hcu(i, inputs, outputs)
#
# for i, dmu in enumerate(dmus):
#     print(f'{dmu} & {np.round(solution_hcu[dmu][0], 2)} \\\\')
#
# print()
# for i, dmu in enumerate(dmus):
#     if solution_hcu[dmu][0] < 1:
#         hcu = solution_hcu[dmu][1]
#         print(
#             f'{dmu} & {" & ".join(list(np.round(hcu, 2).astype(str)))} & {" & ".join(list(np.round(inputs[i] - hcu, 2).astype(str)))} \\\\')

# seff = [super_effectivenes(x, inputs, outputs) for x in range(len(dmus))]
#
# for i, dmu in enumerate(dmus):
#     print(f'{dmu} & {np.round(seff[i], 2)} \\\\')
##          f'{np.round(solution_hcu[dmu][0], 2)} \\\\')

effs = [solve_hcu(dmu, inputs, outputs)[0] for dmu in range(dmu_count)]

dmu_problem_effectivenes(0, inputs, outputs)
optimal_vector(0, effs[0], inputs, outputs)