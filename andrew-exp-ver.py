import csv
import json
import os
import pprint
import random
import sys
from datetime import date

import numpy as np

np.set_printoptions(threshold=np.inf, precision=6, suppress=True)
np.random.seed(1234)
random.seed(1234)


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
                kpdot[i] = 0
                kadot[i] = 0

    stability_constraint()

    return kodot, kpdot, kadot


# update the stiffness parameters
def update_k(K, Ko, Kp, Ka, kodot, kpdot, kadot):
    N = np.shape(K)[0]
    for i in range(N):
        Ko[i] += kodot[i]
        Kp[i] += kpdot[i]
        Ka[i] += kadot[i]
    K = make_K_matrix(Ko, Kp, Ka)

    return K, Ko, Kp, Ka


# order units
def order_units(file, firstnode):
    devdata = np.asarray(np.genfromtxt(file, dtype="str", delimiter=","))
    dev = devdata[0]
    nei = devdata[1:]
    # edgeA = np.where(nei[0]=='-1')[0][0]
    # edgeB = np.where(nei[1]=='-1')[0][0]

    dev_sorted = []
    dev_sorted.append(dev[0])
    for n in dev[1:]:
        try:
            nb = dev[np.where(nei[0] == dev_sorted[-1])[0][0]]
            dev_sorted.append(nb)
        except IndexError:
            pass

    for n, d in enumerate(dev_sorted):  # add zeros to single value id's
        if len(d) == 1:
            dev_sorted[n] = "0" + str(d)
    dev_sorted = np.asarray(dev_sorted)

    def rotate_dev(arr, start_value):
        # Find the index of the start value
        start_index = np.where(arr == start_value)[0][0]

        # Split the array into two parts
        arr1 = arr[:start_index]
        arr2 = arr[start_index:]

        # Concatenate two parts in the reverse order
        rotated_arr = np.concatenate((arr2, arr1))

        return rotated_arr

    rotated_dev = rotate_dev(dev_sorted, firstnode)

    return rotated_dev


# get data from file and sorted idlist
def get_data(fname, dev_sorted):
    with open(fname) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        for row in spamreader:
            header = row[1:]
            break
        N = int((len(header)) / 2)
    unit_order = np.char.lower([i.split("#")[1] for i in header[:N]])
    data = np.genfromtxt(fname, delimiter=",", skip_header=True)
    t = data[:, 0]
    ydata = data[:, 1:]
    angle = ydata[:, :N]
    torque = ydata[:, N:]

    idx = []
    for i, j in zip(unit_order, dev_sorted):
        idx.append(np.where(unit_order == str(j))[0][0])
    angle = angle[:, idx]
    torque = torque[:, idx]
    return t, angle, torque


# write a bat file contains command of setting parameters
def write_stiffness_bat(K, dev_sorted):
    N = dev_sorted.size
    "make stiffness paramaters"
    f = open("update_stiffness.bat", "w")
    path = "C:/Users/UVA/p5acli-jonas/p5acli.py"
    for i in range(N):
        "Unit setup: taui = eps_a dth_i-1 -eps_me dthi - eps_b dth_i+1"
        eps_me = K[i, i]
        eps_a = -K[i, (i - 1) % N]
        eps_b = K[i, (i + 1) % N]

        f.write("python " + path + " set-param ")
        f.write(str(dev_sorted[i]) + " ")  # id
        f.write("epsilon_me=" + format(eps_me, ".4f") + " ")
        f.write("epsilon_a=" + format(eps_a, ".4f") + " ")
        f.write("epsilon_b=" + format(eps_b, ".4f") + " ")
        f.write("\n\n")


# show the angle info
def show_info(D, dev_sorted, N):
    D_ = np.transpose(D * (180 / np.pi))
    D_show = dict()
    for i in range(N):
        D_show[i] = {"id": i + 1, "dD": D_[i], "name": dev_sorted[i]}
    pprint.pprint(D_show)

    return


# get the current angles of each node
# att: this is where the did them reach equilibrium state comes from
def get_angles(dev_sorted):
    # print('Did them reach equilibrium state?')
    # print('Please type y/n to continue:\n')
    # set = input()
    # if set == 'y':
    os.system(r"scan_info.bat")
    date_str = date.today().isoformat()
    with open(date_str + "_scan.json") as f:
        data = json.load(f)
    D = np.zeros(dev_sorted.size)
    for i in range(dev_sorted.size):
        id = int(dev_sorted[i], 16)
        theta = data[str(id)]["theta_me"]
        D[i] = theta / 2

    print("order", dev_sorted)
    print("scan angles", D * (180 / np.pi))
    # else:
    #     sys.exit()

    return D


"""
----- Define Parameters -----
N - number of nodes
L - the length of bars
P - initial locations of nodes
K - stiffness matrix
Ko - onsite passive stiffness vector
Kp - symmetric passive stiffness vector
Ka - active stiffness vector
Nsteps - number of training steps
eta - nudge rate
alpha = eta*learning rate
"""

N = 6
Nsteps = 25

# learning rate = alpha/eta, it should be << 1
eta = 1
alpha = 0.01
lr = alpha / eta

"initial stiffness parameters"
Koi, Kpi, Kai = 0.05, 0.02, 0.0

"initialization"
firstnode = "61"
dev_sorted = order_units("devlist.csv", firstnode)  # find the oder

"Set up stiffness parameters"
Ko, Kp, Ka = make_Ki_vector(N, Koi, Kpi, Kai)
print(Ko, Kp, Ka)
K = make_K_matrix(Ko, Kp, Ka)
print("Stiffness Matrix K = \n", K)
write_stiffness_bat(K, dev_sorted)

os.system(r"clear_stiffness.bat")
os.system(r"reset_rotation.bat")
os.system(r"update_stiffness.bat")
os.system(r"start_all.bat")
os.system(r"reset_rotation.bat")

"get target"
print("Please set the target and press Enter to continue:\n")
S1 = input()
D = get_angles(dev_sorted)

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
DI_id = np.array([0])
DI_val = D[DI_id]
DT_id = np.array([3])
DT_val = D[DT_id]
add_target()

Ntarget = np.shape(DI_id_all)[0]
K_step = np.zeros((N, N, Nsteps + 1))
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

for step in range(Nsteps):
    for target in range(Ntarget):
        print("*" * 20 + "\n")
        print(
            "Learning Info: Step: {}/{}, Target:{}/{}\n".format(
                step + 1, Nsteps, target + 1, Ntarget
            )
        )
        print("*" * 20 + "\n")

        DI = convert_Dval(DI_val_all, DI_id_all, target, N)
        DT = convert_Dval(DT_val_all, DT_id_all, target, N)
        nodesI = DI_id_all[target]
        nodesT = DT_id_all[target]
        d = np.sign(max(nodesI) - min(nodesT))

        "nodes fixed in clamped state"
        nodesC = np.concatenate((nodesI, nodesT))

        """
        ######################################
        implement input angle DI in experiment
        ######################################
        """
        print("######################################################################")
        print("Please set the INPUT angles and release Target Nodes as the follows:\n")
        print("######################################################################")
        show_info(DI, dev_sorted, N)  # show the defined input angular displacement 

        # print("Current DI:\n")
        # show_info(DI, dev_sorted, N)  # show the angular displacement
        print("Press Enter to continue:\n") 

        DI = get_angles(dev_sorted)  # measure and record the angular displacement

        continue_query_1 = input()
        if continue_query_1 == "":
            """
            ##########
            free state
            ##########
            """
            print("Did them reach equilibrium state?")
            print("Enter 'stop' to exit.")
            print("Press Enter to continue.\n")
            equilibrium_query_1 = input()
            if equilibrium_query_1 == "":
                DF = get_angles(
                    dev_sorted
                )  # ------------------------------------------------------------------------------------------------
                DF_step[step + 1, :, target] = DF

                print("Current DF:\n")
                show_info(DF, dev_sorted, N)

                "calculate the error"
                Error = np.sum((DF[nodesT] - DT[nodesT]) ** 2)
                Error_step[step, target] = Error
                print("Error = " + str(Error) + "\n")

                """
                ##########################################
                calculate the nudging angular displacement
                ##########################################
                """
                DN = np.copy(DI)
                DN[nodesT] = DF[nodesT] + eta * (DT[nodesT] - DF[nodesT])

                "implement nudging angular displacement DN in experiment"
                print("#####################################################")
                print("Please implement the NUDGING angles as the follows:\n")
                print("#####################################################")
                show_info(DN, dev_sorted, N)

                print("Press Enter to continue.\n")  
                continue_query_2 = input()

                if continue_query_2 == "":
                    """
                    #############
                    clamped state
                    #############
                    """

                    print("Did them reach equilibrium state?")
                    print("Enter 'stop' to exit.")
                    print("Press Enter to continue.\n")
                    equilibrium_query_1 = input()
                    if equilibrium_query_1 == "":
                        DC = get_angles(
                            dev_sorted
                        )  # --------------------------------------------------------------------------------------------
                        print("Current DC:\n")
                        show_info(DC, dev_sorted, N)

                        """
                        ################
                        update stiffness
                        ################
                        """
                        kodot, kpdot, kadot = make_kdot(DF, DC, lr, direction=d)
                        (
                            K,
                            Ko,
                            Kp,
                            Ka,
                        ) = update_k(K, Ko, Kp, Ka, kodot, kpdot, kadot)

                        "set up new stiffness parameters"
                        write_stiffness_bat(K, dev_sorted)
                        os.system(r"update_stiffness.bat")
                        print("New stiffness parameters have been setted!\n")
                    else:
                        sys.exit()
                else:
                    sys.exit()
            else:
                sys.exit()
        else:
            sys.exit()

    # print("Please reset the angle:\n")
    # print("Please type y/n to continue:\n")
    # S3 = input()
    # if S3 == "y":
    #     os.system(r'reset_rotation.bat')
    # else:
    #     sys.exit()

    K_step[:, :, step + 1] = K
    Ko_step[step + 1, :] = Ko
    Kp_step[step + 1, :] = Kp
    Ka_step[step + 1, :] = Ka
    eigvals_step[step + 1, :] = np.sort(np.linalg.eigvals(K))

    np.save("./K_step.npy", K_step)
    np.save("./Ko_step.npy", Ko_step)
    np.save("./Kp_step.npy", Kp_step)
    np.save("./Ka_step.npy", Ka_step)
    np.save("./DF_step.npy", DF_step)
    np.save("./Error_step.npy", Error_step)

    print("Final Stiffness Matrix K = \n", K)
    print("Onsite Passive Stiffness Matrix K = \n", Ko)
    print("Symmetric Passive Stiffness Matrix K = \n", Kp)
    print("Active Stiffness Matrix K = \n", Ka)
    print("Eigenvalues:\n", np.linalg.eigvals(K))

# dev_sorted = order_units('devlist.csv')
# print(dev_sorted)
# t,angle,torque = get_data('test.csv',dev_sorted)
# plt.imshow(angle,aspect='auto', interpolation='none')
