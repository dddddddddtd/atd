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


def read_samples(filename):
    with open(filename, 'r') as f:
        data = f.read().splitlines()
    data = [line.split(";") for line in data]
    header = data[0][1:]
    data = np.array(data[1:])[:, 1:].astype(float)
    input_count = len([x for x in header if 'i' in x])
    return data[:, :input_count], data[:, input_count:]


dmus, inputs = read_file("inputs.csv")
_, outputs = read_file("outputs.csv")

inputs = np.array(inputs)
outputs = np.array(outputs)
dmu_count = len(dmus)
input_count = inputs.shape[1]
output_count = outputs.shape[1]
sample_v, sample_u = read_samples("samples_homework.csv")


def make_affine(param, values):
    return pulp.LpAffineExpression(list(zip(param.values(), values)))


def dmu_problem_effectivenes(dmu):
    dmu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMaximize)

    v = pulp.LpVariable.dicts("v", list(range(input_count)), 0)
    u = pulp.LpVariable.dicts("u", list(range(output_count)), 0)

    dmu_problem += pulp.LpAffineExpression(list(zip(u.values(), outputs[dmu, :])))
    dmu_problem += pulp.LpAffineExpression(list(zip(v.values(), inputs[dmu, :]))) == 1

    for x in range(dmu_count):
        dmu_problem += make_affine(u, outputs[x, :]) <= make_affine(v, inputs[x, :])

    dmu_problem.solve()

    vs, us = extract_vector(v), extract_vector(u)
    return dmu_problem, vs, us


def super_effectivenes(dmu):
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


def extract_vector(uv):
    if type(uv) != dict:
        result = [uv.varValue]
    else:
        result = [uv[i].varValue for i in range(len(uv))]
    return np.array(result)


def optimal_vector(dmu, eff):
    dmu_problem = pulp.LpProblem(f'DMU_{dmu}', pulp.LpMinimize)

    v = pulp.LpVariable.dicts("v", list(range(input_count)), 0)
    u = pulp.LpVariable.dicts("u", list(range(output_count)), 0)

    dmu_problem += make_affine(u, np.vstack([outputs[:dmu], outputs[dmu + 1:]]).sum(0))
    dmu_problem += make_affine(v, np.vstack([inputs[:dmu], inputs[dmu + 1:]]).sum(0)) == 1

    for x in range(dmu_count):
        if x != dmu:
            dmu_problem += make_affine(u, outputs[x, :]) <= make_affine(v, inputs[x, :])

    dmu_problem += make_affine(u, outputs[dmu, :]) == eff * make_affine(v, inputs[dmu, :])

    dmu_problem.solve()

    us = extract_vector(u)
    vs = extract_vector(v)
    return vs, us


def solve_hcu(dmu):
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


def eff_dist():
    sample_count = sample_v.shape[0]
    rel_eff = np.zeros((dmu_count, sample_count))
    for i in range(dmu_count):
        for j in range(sample_count):
            # rel_eff[i][j] = sample_v[j].dot(inputs[i])/sample_u[j].dot(outputs[i])
            rel_eff[i][j] = sample_u[j].dot(outputs[i]) / sample_v[j].dot(inputs[i]) # odwrócone względem wzoru w pdfie?
    rel_eff /= rel_eff.max(0)

    bin_count = 5
    count_matrix = np.zeros((dmu_count, bin_count))
    bins = np.linspace(0, 1, bin_count + 1, endpoint=True)
    inds = np.digitize(rel_eff, bins, True) - 1
    for dmu in range(dmu_count):
        for i in range(inds.shape[1]):
            count_matrix[dmu][inds[dmu][i]] += 1
    count_matrix /= sample_count

    estimated = rel_eff.mean(1)

    return count_matrix, estimated


def ceff(effs):
    v = np.zeros((dmu_count, input_count))
    u = np.zeros((dmu_count, output_count))

    for i in range(dmu_count):
        vs, us = optimal_vector(i, effs[i])
        # _, vs, us = dmu_problem_effectivenes(i)
        v[i] = vs
        u[i] = us

    effmatrix = np.zeros((dmu_count, dmu_count))
    for i in range(dmu_count):
        for j in range(dmu_count):
            effmatrix[i][j] = u[j].dot(outputs[i]) / (v[j].dot(inputs[i]))

    crosseff = effmatrix.mean(1)
    return effmatrix, crosseff


np.set_printoptions(precision=3)

# effectiveness + hcu
solution_hcu = {}
for i, dmu in enumerate(dmus):
    solution_hcu[dmu] = solve_hcu(i)

# super effectiveness
seff = [super_effectivenes(x) for x in range(len(dmus))]

# cross effectiveness
effs = [solve_hcu(dmu)[0] for dmu in range(dmu_count)]
eff_matrix, cross_eff = ceff(effs)

# effectiveness distribution
count_matrix, estimated = eff_dist()

print("\n1")
for i, dmu in enumerate(dmus):
    print(f'{dmu} & {solution_hcu[dmu][0]:.3f} \\\\')

print("\n2")
for i, dmu in enumerate(dmus):
    if solution_hcu[dmu][0] < 1:
        hcu = solution_hcu[dmu][1]
        print(
            f'{dmu} & {" & ".join(list(np.round(hcu, 3).astype(str)))} & {" & ".join(list(np.round(inputs[i] - hcu, 3).astype(str)))} \\\\')

print("\n3")
for i, dmu in enumerate(dmus):
    print(f'{dmu} & {seff[i]:.3f} \\\\')

print("\n4")
for i in range(dmu_count):
    print(f'{dmus[i]} & ', end='')
    print(" & ".join(np.char.mod("%0.3f", eff_matrix[i])))
    print("&", np.char.mod("%0.3f", cross_eff[i]), '\\\\')

print("\n5")
for i in range(dmu_count):
    print(f'{dmus[i]} & ', end='')
    print(" & ".join(np.char.mod("%0.3f", count_matrix[i])), end='')
    print(" &", np.char.mod("%0.3f", estimated[i]), '\\\\')

print("\n6")
print("rankingi")
print('s_eff:', " \\succ ".join([dmus[i] for i in np.argsort(seff)[::-1]]))
print('cross_eff', " \\succ ".join([dmus[i] for i in np.argsort(cross_eff)[::-1]]))
print('estimated_eff', " \\succ ".join([dmus[i] for i in np.argsort(estimated)[::-1]]))
