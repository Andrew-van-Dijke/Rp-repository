import random

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

plt.style.use(["bmh"])
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)
# np.random.seed(111)

# DI = initial displacement
# DT = target displacement

# De = equilibrium displacement
# Df = fixed displacement
# Dc = clamped displacement

# K = torsional stiffness
# ko = onsite stiffness
# kp = passive stiffness
# ka = active stiffness


def make_Ki_vector(N, ko, kp, ka, sigma=0):
    "make vector Ko, Kp, Ka of each node"
    "Parameters: sigma = random number"
    Ko = np.random.normal(ko, sigma, N)
    Kp = np.random.normal(kp, sigma, N)
    Ka = np.random.normal(ka, sigma, N)

    return Ko, Kp, Ka


def make_K_matrix(Ko, Kp, Ka):
    "make stiffness matrix for torsional spring"
    N = np.shape(Ko)[0]
    K = np.zeros((N, N), dtype=float)

    for i in range(N):
        K[i, (i - 1) % N] = Kp[(i - 1) % N] + Ka[(i - 1) % N]
        K[i, i] = Ko[i]
        K[i, (i + 1) % N] = Kp[i % N] - Ka[i % N]

    return K


def find_equilibrium(K, D, fixed_nodes, N):
    def loop_equation(i, N, type):
        polytheta = np.pi - (N - 2) * np.pi / N
        sum_theta = [
            sp.Add(*[val + polytheta for val in D_sym[1 : j + 2]]) for j in range(i, N)
        ]
        sum_theta = list(
            dict.fromkeys(sum_theta)
        )  # Remove any potential duplicates from sum_theta
        if type == "sin":
            loop = [sp.sin(val) for val in sum_theta]
        else:
            loop = [sp.cos(val) for val in sum_theta]
        loop_final = sum(loop)
        return loop_final

    def solve():
        tolerance = 1e-6
        max_attempts = 5  # Maximum number of attempts to decrease tolerance
        attempt = 0

        while attempt < max_attempts:
            try:
                solutions = sp.nsolve(equations, variables, guess, tol=tolerance)
                break  # If successful, break out of the loop
            except ValueError as e:
                if "Could not find root within given tolerance" in str(e):
                    # Double the precision by halving the tolerance
                    tolerance /= 10
                    attempt += 1
                    print("tolerance decrease since solution could not find. ")
                else:
                    raise  # If the error is not related to tolerance, raise it
        else:
            # This block will run if the loop completes without finding a solution
            raise ValueError(
                "Could not find a solution even after decreasing tolerance."
            )
        return solutions

    L = 1  # lengthS
    nodes = np.arange(0, N)
    fixed_nodes = np.sort(fixed_nodes)  # fixed nodes
    nodes = np.delete(nodes, fixed_nodes)  # free nodes

    D_sym = sp.symbols("D0:%d" % N)  # create D_0 ... D_N
    K_sym = sp.Matrix(K)

    "Lagrange multiplier"
    a = sp.symbols("a0:%d" % len(fixed_nodes))  # fixed angle
    b = sp.symbols("b")  # sum of angles
    c = sp.symbols("c")  # x_0 == 0
    d = sp.symbols("d")  # y_0 == 0

    equations = []
    j = 0  # fixed node id

    for i in range(N):
        equation = K_sym.row(i).dot(D_sym) + b  # Common part

        if i != 0:  # for i!=0 node, add constraint (x0, y0)==0
            equation -= c * L * loop_equation(i - 1, N, type="sin")
            equation += d * L * loop_equation(i - 1, N, type="cos")

        if i in fixed_nodes:  # for fixed node, add constrain D[i] = constant
            equation += a[j]
            j += 1

        equations.append(sp.Eq(equation, 0))

    for _, fi in enumerate(fixed_nodes):
        equations.append(sp.Eq(D_sym[fi], D[fi]))
    equations.append(sp.Eq(sp.Add(*D_sym), 0))
    equations.append(sp.Eq(L + L * loop_equation(0, N, type="cos"), 0))
    equations.append(sp.Eq(L * loop_equation(0, N, type="sin"), 0))

    multiplier_guess = [0.0] * (len(fixed_nodes) + 3)
    De_guess = [0.1] * N
    guess = De_guess + multiplier_guess
    variables = list(D_sym) + list(a) + [b] + [c] + [d]
    solutions = solve()

    De = [solutions[i] for i in range(N)]
    De = [
        float(val) for val in De
    ]  # Convert SymPy Float objects to native Python floats
    multiplier = solutions[:-4]

    "convert list to numpy array"
    De = np.array(De)
    multiplier = np.array(multiplier)
    return De


def make_kdot(DF, DC, lr, direction):
    N = np.shape(DF)[0]
    I = np.zeros_like(K)
    np.fill_diagonal(I, 1)
    Ir = np.roll(I, -1, axis=0)
    Il = np.roll(I, 1, axis=0)

    "rotation of free state"
    DFil = np.dot(Il, DF)  # DF[i-1]
    DFi = np.dot(I, DF)  # DF[i]
    DFir = np.dot(Ir, DF)  # DF[i+1]
    "clamped rotation"
    DCil = np.dot(Il, DC)  # DC[i-1]
    DCi = np.dot(I, DC)  # DC[i]
    DCir = np.dot(Ir, DC)  # DC[i+1]

    grad_ko = (np.square(DFi) - np.square(DCi)) * 0.5
    grad_kp = DFi * DFir - DCi * DCir
    grad_ka = -(DFi * DFir - DCi * DCir) * direction

    kodot = lr * grad_ko
    kpdot = lr * grad_kp
    kadot = lr * grad_ka

    def stability_constraint():
        for i in range(N):
            # the updated parameters
            Ko_ = Ko[i] + kodot[i]
            Kp_ = Kp[i] + kpdot[i]
            Ka_ = Ka[i] + kadot[i]
            Ki = Ko_
            Kil = Kp_ + Ka_
            kir = Kp_ - Ka_

            # Gershgorin disk
            Ci = Ki  # the center of ith disk
            Ri = abs(Kil) + abs(kir)  # the radius of ith disk
            Stable = (
                Ci > 0 and (Ri - Ci) < 0
            )  # if yield, the real part of ith eigenvalue is positive

            if not Stable:
                # kodot[i] = 0
                # kpdot[i] = 0
                kadot[i] = 0

    stability_constraint()

    return kodot, kpdot, kadot


def update_k(K, Ko, Kp, Ka, kodot, kpdot, kadot):
    N = np.shape(K)[0]
    for i in range(N):
        Ko[i] += kodot[i]
        Kp[i] += kpdot[i]
        Ka[i] += kadot[i]
    K = make_K_matrix(Ko, Kp, Ka)

    return K, Ko, Kp, Ka


N = 6  # ---------------------------------------------------------------------------------------------------------------
Nsteps = 20  # --------------------------------------------------------------------------------------------------------
Koi, Kpi, Kai = (
    0.05,
    0.01,
    0.00,
)  # --------------------------------------------------------------------------------------

"learning rate = alpha/eta, it should be << 1"
eta = 1  # -------------------------------------------------------------------------------------------------------------
alpha = 0.001  # --------------------------------------------------------------------------------------------------------
lr = alpha / eta  # learing rate
lrlist = [
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.025,
]
#     0.01,
#     0.02,
#     0.025,
#     0.05,
#     0.1,
#     0.2,
# ]

# lrlist = [
#     0.01,
#     0.012,
#     0.014,
#     0.016,
#     0.018,
#     0.02,
#     0.025,
#     0.04,
#     0.05,
#     0.075,
#     0.1,
# ]

"define targets"
DI_id_all = []
DT_id_all = []
DI_val_all = []
DT_val_all = []


def add_target():
    DI_id_all.append(DI_id)
    DT_id_all.append(DT_id)
    DI_val_all.append(DI_val)
    DT_val_all.append(DT_val)


"target 1"
" the angles are exterior angles"
DI_id = np.array(
    [0]
)  # ------------------------------------------------------------------------------------------------
DI_val = np.array([60]) * (
    np.pi / 180
)  # ----------------------------------------------------------------------------------
DT_id = np.array(
    [3]
)  # ------------------------------------------------------------------------------------------------
DT_val = np.array([180]) * (
    np.pi / 180
)  # ----------------------------------------------------------------------------------

" because the reference configuration is a regular polygon, the angluar difference of each unit should minus the exterior angle of this polygon"
polytheta = (N - 2) * np.pi / N  # Exterior angles of a regular N-gon
DI_val = polytheta - DI_val
DT_val = polytheta - DT_val
add_target()

"target 2"
" the angles are exterior angles"
DI_id = np.array(
    [3]
)  # ------------------------------------------------------------------------------------------------
DI_val = np.array([60]) * (
    np.pi / 180
)  # ----------------------------------------------------------------------------------
DT_id = np.array(
    [0]
)  # ------------------------------------------------------------------------------------------------
DT_val = np.array([180]) * (
    np.pi / 180
)  # ----------------------------------------------------------------------------------

" because the reference configuration is a regular polygon, the angluar difference of each unit should minus the exterior angle of this polygon"
polytheta = (N - 2) * np.pi / N  # Exterior angles of a regular N-gon
DI_val = polytheta - DI_val
DT_val = polytheta - DT_val

add_target()

# DI_id = np.array([3])
# DI_val = np.array([10])*(np.pi/180)
# DT_id = np.array([0])
# DT_val = np.array([])*(np.pi/180)
# add_target()-10




Errorlist = []
for learning_rate in lrlist:
    Nsteps = int(0.1 / learning_rate)

    Ntarget = np.shape(DI_id_all)[0]
    K_step = np.zeros((N, N, Nsteps + 1))  # the stiffness materix of each steps
    Ko_step = np.zeros((Nsteps + 1, N))
    Kp_step = np.zeros((Nsteps + 1, N))
    Ka_step = np.zeros((Nsteps + 1, N))
    DF_step = np.zeros(
        (Nsteps + 1, N, Ntarget)
    )  # the angle deflecton of the free state in each step
    eigvals_step = np.zeros(
        (Nsteps + 1, N), dtype=np.complex64
    )  # eigenvalues of stiffness matrix in each steps
    Error_step = np.zeros((Nsteps, Ntarget))

    Ko, Kp, Ka = make_Ki_vector(N, Koi, Kpi, Kai, sigma=0)
    K = make_K_matrix(Ko, Kp, Ka)
    print("Stiffness Matrix K = \n", K)

    "initial values"
    K_step[:, :, 0] = K
    Ko_step[0, :] = Ko
    Kp_step[0, :] = Kp
    Ka_step[0, :] = Ka
    eigvals_step[0, :] = np.sort(np.linalg.eigvals(K))


    def convert_Dval(D_val_all, D_id_all, target_id, N):
        D = np.zeros((1, N))
        D[:, D_id_all[target_id]] = D_val_all[target_id]
        return D.reshape(
            N,
        )
    for i in range(Ntarget):
        DF_step[0, :, i] = convert_Dval(DI_val_all, DI_id_all, i, N)
    Error = 0
    for step in range(Nsteps):
        if (step + 1) % 10 == 0:
            print("Current step: {} / {}".format(step + 1, Nsteps))
            print(Error)
        for target in range(Ntarget):
            DI = convert_Dval(DI_val_all, DI_id_all, target, N)
            DT = convert_Dval(DT_val_all, DT_id_all, target, N)
            nodesI = DI_id_all[target]
            nodesT = DT_id_all[target]
            d = np.sign(max(nodesI) - min(nodesT))

            "nodes fixed in clamped state"
            nodesC = np.sort(np.concatenate((nodesI, nodesT)))

            "find initial free states"
            DF = find_equilibrium(K, DI, nodesI, N)
            DF_step[step + 1, :, target] = DF

            " calculate the error"
            Error = np.sum((DF[nodesT] - DT[nodesT]) ** 2)
            Error_step[step, target] = Error

            "define nudge rotation vector DN"
            DN = np.copy(DI)
            DN[nodesT] = DF[nodesT] + eta * (DT[nodesT] - DF[nodesT])

            "clamped state"
            DC = find_equilibrium(K, DN, nodesC, N)

            "update stiffness matrix"
            kodot, kpdot, kadot = make_kdot(
                DF, DC, learning_rate, direction=d
            )  # --------------------------------
            (
                K,
                Ko,
                Kp,
                Ka,
            ) = update_k(K, Ko, Kp, Ka, kodot, kpdot, kadot)

        K_step[:, :, step + 1] = K
        Ko_step[step + 1, :] = Ko
        Kp_step[step + 1, :] = Kp
        Ka_step[step + 1, :] = Ka
        eigvals_step[step + 1, :] = np.sort(np.linalg.eigvals(K))

    Errorlist.append(Error)
    print(Errorlist)

    np.save("./K_step.npy", K_step)
    np.save("./Ko_step.npy", Ko_step)
    np.save("./Kp_step.npy", Kp_step)
    np.save("./Ka_step.npy", Ka_step)
    np.save("./DF_step.npy", DF_step)
    np.save("./Error_step.npy", Error_step)

    # print("Final Stiffness Matrix K = \n", K)
    # print("Onsite Passive Stiffness Matrix K = \n", Ko)
    # print("Symmetric Passive Stiffness Matrix K = \n", Kp)
    # print("Active Stiffness Matrix K = \n", Ka)
    # print("Eigenvalues = \n", eigvals_step[-1, :])

    print(learning_rate)

print(Errorlist)
plt.scatter(lrlist, Errorlist)
plt.xlabel("learning rate")
plt.ylabel("Error")
plt.xscale("log")
plt.yscale("log")


def visulization():
    font = 15
    f, ax = plt.subplots(1, 1)
    "error over time"
    for i in range(Ntarget):
        ax.plot(np.arange(Nsteps), Error_step[:, i], label="target {}".format(i + 1))
    ax.set_xlabel("Steps", fontsize=font)
    ax.set_ylabel("Error", fontsize=font)
    ax.set_yscale("log")
    ax.legend()
    plt.show()

    "angle over time"
    f, ax = plt.subplots(1, Ntarget)
    if Ntarget > 1:
        for i in range(Ntarget):
            for j in range(N):
                ax[i].plot(
                    np.arange(Nsteps + 1),
                    DF_step[:, j, i] * (180 / np.pi),
                    label=r"$\theta_{{{}}}$".format(j + 1),
                )
                ax[i].set_xlabel("Steps", fontsize=font)
                ax[i].set_ylabel(r"Angle ($\circ$)", fontsize=font)
                ax[i].legend(fontsize=font)
                ax[i].set_title("Target {}".format(i + 1))
                ax[i].legend()
    else:
        for j in range(N):
            ax.plot(
                np.arange(Nsteps + 1),
                DF_step[:, j, i] * (180 / np.pi),
                label=r"$\theta_{{{}}}$".format(j + 1),
            )
            ax.set_xlabel("Steps", fontsize=font)
            ax.set_ylabel(r"Angle ($\circ$)", fontsize=font)
            ax.legend(fontsize=font)
            ax.set_title("Target {}".format(i + 1))
            ax.legend()
    plt.show()

    "stiffnesses over time"
    f, ax = plt.subplots(1, 1)
    for i in range(N):
        ax.plot(
            np.arange(Nsteps + 1),
            Ko_step[:, i],
            label=r"$k_{{{}}}^{{o}}$".format(i + 1),
        )
    for i in range(N - 1):
        ax.plot(
            np.arange(Nsteps + 1),
            Kp_step[:, i],
            label=r"$k_{{{}}}^{{p}}$".format(i + 1),
        )
        ax.plot(
            np.arange(Nsteps + 1),
            Ka_step[:, i],
            label=r"$k_{{{}}}^{{a}}$".format(i + 1),
        )
    ax.set_xlabel("Steps", fontsize=font)
    ax.set_ylabel(r"Stiffnesses $k_{i}$", fontsize=font)
    ax.legend(fontsize=font, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=5)
    plt.show()

    "plot eigenvalues"
    f, ax = plt.subplots(2, 1, figsize=(10, 5))
    for i in range(N):
        ax[0].plot(np.real(eigvals_step[:, i]), label=r"$\lambda_{}$".format(i + 1))
    # ax.axhline(y=0, ls="--", c='#000000')
    ax[0].set_ylabel(r"Re($\lambda$)", fontsize=font)
    ax[0].legend()

    for i in range(N):
        ax[1].plot(np.imag(eigvals_step[:, i]), label=r"$\lambda_{}$".format(i + 1))
    # ax.axhline(y=0, ls="--", c='#000000')
    ax[1].set_xlabel("Steps", fontsize=font)
    ax[1].set_ylabel(r"Im($\lambda$)", fontsize=font)
    ax[1].legend()
    plt.show()

    # visulization
    def show_bar(D, nodesI, nodesT, target):
        "Plot mass-bar structure"

        N = np.shape(D)[0]
        polytheta = (180 - (N - 2) * 180 / N) * (np.pi / 180)
        L = 1
        P = np.arange(0, N, L)

        print(D * (180 / np.pi))
        print(np.sum(D) * (180 / np.pi))
        print(polytheta * (180 / np.pi))

        "new locations"
        Pxy = np.zeros((N, 2))
        Pxy[:, 0] = P
        theta = 0.0
        for i in range(1, N):
            theta += D[i - 1]
            Pxy[i, 0] = Pxy[i - 1, 0] + L * (np.cos(theta + polytheta * (i - 1)))
            Pxy[i, 1] = Pxy[i - 1, 1] + L * (np.sin(theta + polytheta * (i - 1)))

        # add the last bar information
        Pxy = np.append(Pxy, [[0, 0]], axis=0)

        # create a new subplot
        _, ax_sub = plt.subplots(1, 1)
        # plot the bars
        t = 1
        ax_sub.plot(Pxy[:, 0], Pxy[:, 1], lw=3, alpha=t, color="#87CEFA")

        # plot the masses
        size = 200
        ax_sub.scatter(
            Pxy[:, 0],
            Pxy[:, 1],
            marker="o",
            s=size,
            facecolor="w",
            edgecolors="k",
            linewidth=2,
            alpha=t,
            zorder=2,
        )
        ax_sub.scatter(
            Pxy[0],
            Pxy[0],
            marker="o",
            s=size,
            facecolor="#228B22",
            edgecolors="k",
            alpha=t,
            zorder=2,
        )  # inputs:first
        ax_sub.scatter(
            Pxy[nodesI, 0],
            Pxy[nodesI, 1],
            marker="o",
            s=size,
            facecolor="#FF8C00",
            edgecolors="k",
            alpha=t,
            zorder=2,
        )  # inputs:orange
        ax_sub.scatter(
            Pxy[nodesT, 0],
            Pxy[nodesT, 1],
            marker="o",
            s=size,
            facecolor="#B22222",
            edgecolors="k",
            alpha=t,
            zorder=2,
        )  # outputs:red

        # configure the axes
        ax_sub.axis("equal")
        ax_sub.set_xlim(Pxy[:, 0].min() - 1, Pxy[:, 0].max() + 1)
        ax_sub.set_ylim(Pxy[:, 1].min() - 1, Pxy[:, 1].max() + 1)
        ax_sub.set_axis_off()
        ax_sub.set_title("Target {}".format(target + 1))
        ax_sub.axis("equal")

        return ax

    "plot the ids_th structure"
    fig, ax = plt.subplots(Ntarget, 1)
    if Ntarget > 1:
        for i in range(Ntarget):
            nodesI = DI_id_all[i]
            nodesT = DT_id_all[i]
            DF_final = DF_step[-1, :, i]
            ax[i] = show_bar(DF_final, nodesI, nodesT, i)
    else:
        nodesI = DI_id_all[0]
        nodesT = DT_id_all[0]
        DF_final = DF_step[-1, :, 0]
        ax = show_bar(DF_final, nodesI, nodesT, 0)
    plt.show()


# visulization()
