import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 20,
        "text.latex.preamble": "\n".join(
            [r"\usepackage{siunitx}", r"\usepackage{eurosym}"]
        ),
    }
)


def task(
    df, c_col, min_nutrient, min_b, max_nutrient, max_b, eradicate=None, eq_nutrient=None, eq_b=None, ratios_df=None, food_groups=None
):
    if isinstance(c_col, str):
        c = df[c_col]
    else:
        c = np.zeros(len(df))
        for col, operation in c_col:
            c += df[col] / df[col].max() * (1 if operation == "+" else -1)

    min_A = [-df[name].to_numpy() for name in min_nutrient]
    min_b = [-e for e in min_b]

    max_A = [df[name].to_numpy() for name in max_nutrient]

    A_ub = min_A + max_A
    b_ub = min_b + max_b

    A_eq = []
    b_eq = []
    if eradicate is not None:
        condition = np.zeros(len(df))
        for zivilo in eradicate:
            ind = df.index[df["zivilo"] == zivilo][0]
            condition[ind] = 1
        A_eq.append(condition)
        b_eq.append(0)
    if eq_nutrient is not None:
        A_eq = [df[name].to_numpy() for name in eq_nutrient]
        b_eq += eq_b

    if ratios_df is not None:
        conditions = dict()
        for row in ratios_df.itertuples():
            ind0 = df.index[df["zivilo"] == row.food1][0]
            ind1 = df.index[df["zivilo"] == row.food2][0]
            if row.food1 not in conditions.keys():
                conditions[row.food1] = np.zeros(len(df))
                conditions[row.food1][ind0] = 1
            conditions[row.food1][ind1] += - row.ratio2 / row.ratio1
        parsed = [condition for condition in conditions.values()]
        A_ub += parsed
        b_ub += [0 for _ in range(len(parsed))]

    if food_groups is not None:
        food_arrays = dict()
        for group in food_groups.keys():
            array = (df["group"] == group).astype(int).to_numpy()
            food_arrays[group] = array

        conditions = []
        items = iter(food_groups.items())
        prev_group, prev_ratio = next(items)
        for group, ratio in items:
            condition = prev_ratio * food_arrays[prev_group] - ratio * food_arrays[group]
            conditions.append(condition)
        A_eq += conditions
        b_eq += [0 for _ in range(len(conditions))]

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)
    if len(A_eq) == 0: A_eq = None
    else:
        A_eq = np.vstack(A_eq)
        b_eq = np.array(b_eq)

    res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)

    return res


def draw_graphs(df, res, A_names, b, name="", pie_only=False):
    labels = []
    values = []

    onepercent = np.sum(res.x) * 0.01
    for i, x_zivilo in enumerate(res.x):
        if x_zivilo < onepercent: continue
        labels.append(df["zivilo"][i])
        values.append(x_zivilo)

    autopct = lambda pct: rf"{pct:.0f}\%" if pct > 3 else ""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
    colours = plt.cm.magma(np.linspace(0.2, 0.8, len(labels)))[::-1]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=autopct,
        startangle=30,
        colors=colours,
        pctdistance=0.8,
        wedgeprops=dict(width=0.5, edgecolor="white"),
    )

    plt.setp(autotexts, weight="bold", color="white")

    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    if name == "":
        plt.show()
    else:
        fig.savefig(f"images/{name}p.pdf")
    plt.close(fig)

    if pie_only: return
    labels = []
    values = []
    for i, val in enumerate(res.slack):
        if val < onepercent:
            continue
        labels.append((A_names)[i])
        values.append(val / b[i])

    fig, ax = plt.subplots()
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.bar(labels, values)
    ax.set_ylabel(r"$(\mathbf{b} - A\mathbf{x})_i/b_i$")
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    if name == "":
        plt.show()
    else:
        fig.savefig(f"images/{name}slack.pdf")
    plt.close(fig)


def task1(df):
    c_col = "energija[kcal]"
    min_A = [
        "mascobe[g]",
        "ogljikovi hidrati[g]",
        "proteini[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [70, 310, 50, 1000, 18, 60, 3500, 500]
    max_A = ["Natrij[mg]", "masa[g]"]
    max_b = [2400, 2000]

    res = task(df, c_col, min_A, min_b, max_A, max_b)
    draw_graphs(df, res, min_A + max_A, min_b + max_b, "task1")


def task2(df):
    c_col = "mascobe[g]"
    min_A = [
        "energija[kcal]",
        "ogljikovi hidrati[g]",
        "proteini[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [2000, 310, 50, 1000, 18, 60, 3500, 500]
    max_A = ["Natrij[mg]", "masa[g]"]
    max_b = [2400, 2000]

    res = task(df, c_col, min_A, min_b, max_A, max_b)
    print(res)
    draw_graphs(df, res, min_A + max_A, min_b + max_b, "task2")


def task3(df):
    c_col = "Cena[EUR]"
    min_nutrient = [
        "mascobe[g]",
        "ogljikovi hidrati[g]",
        "proteini[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [70, 310, 50, 1000, 18, 60, 3500, 500]
    max_A = ["Natrij[mg]", "masa[g]"]
    max_b = [2400, 2000]

    energies = list(range(500, 6_000, 100))
    costs = []
    for energy in energies:
        eq_A = ["energija[kcal]"]
        eq_b = [energy]
        res = task(df, c_col, min_nutrient, min_b, max_A, max_b, eq_A, eq_b)
        costs.append(res.fun)

    fig, ax = plt.subplots()
    ax.plot(energies, costs)
    ax.set_xlabel("energija[kcal]")
    ax.set_ylabel(r"cena[\EUR{}]")
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig("images/energija.pdf")
    plt.close(fig)

    min_nutrient = [
        "energija[kcal]",
        "ogljikovi hidrati[g]",
        "proteini[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [2000, 310, 50, 1000, 18, 60, 3500, 500]
    fats = list(range(0, 200, 5))
    costs = []
    for fat in fats:
        eq_A = ["mascobe[g]"]
        eq_b = [fat]
        res = task(df, c_col, min_nutrient, min_b, max_A, max_b, eq_A, eq_b)
        costs.append(res.fun)

    fig, ax = plt.subplots()
    ax.plot(fats, costs)
    ax.set_xlabel("mascobe[g]")
    ax.set_ylabel(r"cena[\EUR{}]")
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig("images/mascobe.pdf")

    plt.close(fig)

    c_col = [("Cena[EUR]", "+"), ("mascobe[g]", "+"), ("proteini[g]", "-")]
    min_A = [
        "energija[kcal]",
        "ogljikovi hidrati[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [2000, 310, 1000, 18, 60, 3500, 500]
    res = task(df, c_col, min_A, min_b, max_A, max_b, eq_A, eq_b)
    print(res.x)

    draw_graphs(df, res, (min_A + max_A), (min_b + max_b), "task3h")


def task4(df, ratios_df):
    # c_col = "Cena[EUR]"
    c_col = "mascobe[g]"
    min_A = [
        "energija[kcal]",
        "ogljikovi hidrati[g]",
        "proteini[g]",
        "Ca[mg]",
        "Fe[mg]",
        "Vitamin C[mg]",
        "Kalij[mg]",
        "Natrij[mg]",
    ]
    min_b = [2000, 310, 50, 1000, 18, 60, 3500, 500]
    max_A = ["Natrij[mg]", "masa[g]"]
    max_b = [2400, 2000]

    res = task(df, c_col, min_A, min_b, max_A, max_b, ratios_df=ratios_df)
    draw_graphs(df, res, (min_A + max_A), (min_b + max_b), "food_ratios", pie_only=True)

    food_group_ratio = {
        "Produce": 20,
        "Protein": 12,
        "Sweets": 8,
        "Grains": 6,
        "Dairy": 5,
        "Beverages": 4
    }
    res = task(df, c_col, min_A, min_b, max_A, max_b, food_groups=food_group_ratio)
    draw_graphs(df, res, (min_A + max_A), (min_b + max_b), "group_ratios", pie_only=True)

    res = task(df, c_col, min_A, min_b, max_A, max_b, ratios_df=ratios_df, food_groups=food_group_ratio)
    draw_graphs(df, res, (min_A + max_A), (min_b + max_b), "groupfodie_ratios", pie_only=True)
    eradicate = []
    for i in range(15):
        indices = list(range(len(res.x)))
        top_3 = sorted(indices, key=lambda ind: res.x[ind], reverse=True)
        eradicate.append(df["zivilo"][top_3[3]])
        res = task(df, c_col, min_A, min_b, max_A, max_b, eradicate=eradicate, ratios_df=ratios_df, food_groups=food_group_ratio)
        draw_graphs(df, res, (min_A + max_A), (min_b + max_b), name=f'eradicate{i}', pie_only=True)

#df = pd.read_csv(
#    "tabela-zivil.dat", sep=r"\t+", comment="#", na_values=[""], engine="python"
#)
df = pd.read_csv("tabela_zivil_edited.csv")
df[df.select_dtypes(include="number").columns] = (
    df.select_dtypes(include="number") / 100
)
df["masa[g]"] = np.ones(len(df))

print(df.head())
ratios_df = pd.read_csv("food_ratios.csv")
print(df[df["zivilo"] == "Tuna"])
print(df[df["zivilo"] == "Govedina"])
print(df[df["zivilo"] == "Svinjina"])
# task1(df)
# task2(df)
# task3(df)
task4(df, ratios_df)
