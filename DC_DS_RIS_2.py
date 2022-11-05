import copy
import numpy as np

np.set_printoptions(precision=6, threshold=1e3)
import warnings
import cvxpy as cp
import sys


def DC_theta(N, K, L, h_d, G, maxiter, f, verbose, epsilon, gamma):
    # Compute R,c:   #N服务器天线数 #K设备数量 #L反射元件
    A = np.zeros([L, K], dtype=complex)
    c = np.zeros([K, ], dtype=complex)
    R = np.zeros([L + 1, L + 1, K], dtype=complex)
    for k in range(K):
        c[k] = f.conj() @ h_d[:, k]
        A[:, k] = (f.conj() @ G[:, :, k])
        R[0:L, 0:L, k] = np.outer(A[:, k], A[:, k].conj())
        R[0:L, L, k] = A[:, k] * c[k]
        R[L, 0:L, k] = R[0:L, L, k].conj()
    #        R[:,:,k]=copy.deepcopy((R[:,:,k].conj().T+R[:,:,k])/2)
    # initial V:
    V = np.random.randn(L + 1, 1) + 1j * np.random.randn(L + 1, 1)  # 把t也看做和theta一样的分布
    V = V / np.abs(V)
    V = copy.deepcopy(np.outer(V, V.conj()))
    #    for k in range(K):
    #        print(np.trace(R[:,:,k]@V))
    #        print(np.abs(c[k])**2)

    _, v = np.linalg.eigh(V)  # V=v*v^H是个自共轭矩阵，即V=V^H，返回特征值和特征向量
    #    print(v.shape)

    u = np.random.randn(L + 1, 1) + 1j * np.random.randn(L + 1, 1);
    # initial other parameters:
    # infeasible_check = False
    # initial the optimization problem:
    V_var = cp.Variable((L + 1, L + 1), hermitian=True)
    V_var.value = V
    V_partial = cp.Parameter((L + 1, L + 1), hermitian=True)
    V_partial.value = copy.deepcopy(np.outer(u, u.conj()))
    #    print(M_partial.value)
    constraints = [V_var >> 0]
    constraints += [V_var[n, n] == 1 for n in range(L)]
    constraints += [cp.real(V_var @ R[:, :, k]) + np.abs(c[k]) ** 2 >= 1 for k in range(K)]
    cost = cp.real(cp.trace((np.eye(L + 1) - V_partial) @ V_var))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    obj0 = 0
    for iter in range(maxiter):
        if verbose > 1:
            print('Solving theta, Inner iter={}'.format(iter))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open('out.log', 'w+') as f:
                sys.stdout.flush()
                stream = sys.stdout
                sys.stdout = f
                prob.solve(solver=cp.SCS, verbose=False, scale=1e-8, max_iters=50, warm_start=True)
                sys.stdout.flush()
                sys.stdout = stream
        # print(prob.value)
        if verbose:
            print('Status={}, Value={}'.format(prob.status, prob.value))
        if prob.status == 'infeasible' or prob.value == np.inf or prob.value == -np.inf:
            return None, False

        err = np.abs(prob.value - obj0)
        V = copy.deepcopy(V_var.value)
        print(iter)
        _, v = np.linalg.eigh(V)
        u = v[:, L]
        V_partial.value = copy.deepcopy(np.outer(u, u.conj()))
        obj0 = prob.value
        if err < epsilon:
            break
    u, s, _ = np.linalg.svd(V, compute_uv=True, hermitian=True)
    v_tilde = u[:, 0]
    vv = v_tilde[0:L] / v_tilde[L]
    vv = copy.deepcopy(vv / np.abs(vv))


    fp=open("sums.txt", "a+")
    print(sum(s[1:]),file=fp)
    fp.close()
    feasibility = sum(s[1:]) < 1e-6
    # if feasibility:
    #     for i in range(K):
    #         flag = np.linalg.norm(v) ** 2 / np.linalg.norm(v.conj() @ h_d[:, i]) ** 2 <= gamma
    #         if not flag:
    #             feasibility = False

    return vv, feasibility


# N天线
# K设备数，同M
# L传进去就行，里面没用
# 第一个h应该是h_d,G共同作用后的信道
def user_selection_DC(N, K, L, h, gamma, maxiter, epsilon, verbose, h_d, G):
    M_var = cp.Variable((N, N), hermitian=True
                        )
    x_var = cp.Variable(K, nonneg=True)
    x_partial = cp.Parameter(K)
    M_partial = cp.Parameter((N, N), hermitian=True
                             )

    constraints = [cp.real(cp.trace(M_var)) - 1 >= 0]
    constraints += [M_var >> 0]
    h_var = cp.Parameter((N, K), complex=True)
    h_var.value = copy.deepcopy(h)

    constraints += [cp.real(cp.trace(M_var)) - gamma * cp.real(h_var[:, k].H @ M_var @ h_var[:, k]) - x_var[k] <= 0 for
                    k in range(K)]
    cost = cp.norm(x_var, 1) - x_partial.H @ x_var + cp.real(cp.trace((np.eye(N) - M_partial.H) @ M_var))

    prob = cp.Problem(cp.Minimize(cost), constraints)
    #
    for c in range(K + 1):
        x = np.random.randn(K, )
        M = np.random.randn(N, N) + 1j * np.random.randn(N, N);
        M = M @ M.conj().T
        x_abs = np.abs(x)
        x_p = np.zeros([K, ])
        ind = np.argsort(-x_abs)
        #        x_partial[ind[c+1:end]] = 0;
        x_p[ind[0:c]] = copy.deepcopy(np.sign(x[ind[0:c]]));

        #        print(x_p)
        x_partial.value = copy.deepcopy(x_p)
        _, V = np.linalg.eigh(M)
        u = V[:, N - 1]

        M_partial.value = copy.deepcopy(np.outer(u, u.conj()))

        obj0 = 0
        for iter in range(maxiter):
            if verbose:
                print('c={} iter={}'.format(c, iter))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                #                sys.stdout=r_obj
                with open('out.log', 'w+') as f:
                    stream = sys.stdout
                    sys.stdout.flush()
                    sys.stdout = f
                    prob.solve(solver=cp.SCS, verbose=False)
                    sys.stdout.flush()
                    sys.stdout = stream
            # print(prob.status)
            if prob.status == 'infeasible' or prob.value == np.inf or prob.value == -np.inf:
                # print("prob.status=",prob.status)
                break
            x = copy.deepcopy(x_var.value)
            M = copy.deepcopy(M_var.value)
            err = abs(prob.value - obj0)
            x_p = np.zeros([K, ])
            ind = np.argsort(-x_abs)
            x_p[ind[0:c]] = copy.deepcopy(np.sign(x[ind[0:c]]));
            x_partial.value = copy.deepcopy(x_p)
            _, V = np.linalg.eigh(M)
            u = V[:, N - 1]

            M_partial.value = copy.deepcopy(np.outer(u, u.conj()))
            obj0 = prob.value
            print("err:{},value:{}".format(err,prob.value))
            if err < 1e-9 or prob.value < 1e-7:
                break
        s = np.linalg.svd(M, compute_uv=False, hermitian=True)
        print("hereherehere")
        feasibility = sum(s[1:]) < 1e-6
        if feasibility:
            break

    ind = np.argsort(x)

    #    print(x)
    #    print(ind[0:1])
    for i in np.arange(K):
        active_user_num = K - i
        active_user = np.asarray(ind[0:active_user_num])
        theta, feasibility = DC_theta(N, K, L, h_d, G, maxiter, np.ones([N, ], dtype='complex'), verbose, epsilon,
                                      gamma)

        if verbose:
            print('try user num: {}, feasible:{}'.format(active_user_num, feasibility))
        if feasibility:
            #            num_of_users=active_user_num
            break

    if not feasibility:
        theta = None
        active_user = []



    return theta, active_user


def DC_DS_RIS_2(libopt, h_d, G, x0, gamma_set, verbose):
    N = libopt.N
    M = libopt.M
    L = libopt.L
    K = libopt.K / np.mean(libopt.K)
    K2 = K ** 2
    Ksum2 = sum(K) ** 2

    # maxiter = 5
    maxiter = 100
    #    maxiter=1
    epsilon = 1e-5
    obj_DC = np.zeros([len(gamma_set), ])
    X_DC = np.zeros([M, len(gamma_set)])
    theta_DC = np.zeros([L, len(gamma_set)], dtype='complex')

    theta_DC_RIS = np.ones([L], dtype=complex)
    h = np.zeros([N, M], dtype=complex)
    for i in range(M):
        h[:, i] = h_d[:, i] + G[:, :, i] @ theta_DC_RIS

    for i in range(len(gamma_set)):
        gamma = 10 ** (gamma_set[i] / 10)
        if verbose:
            print('gamma:{:.6f},\n'.format(gamma))

        theta, active_user = user_selection_DC(N, M, L, h, gamma, maxiter, epsilon, verbose, h_d, G)

        x = np.zeros([M, ])
        if verbose:
            print(x)
        if theta is not None:
            theta_DC[:, i] = copy.deepcopy(theta)
            x[active_user] = 1

        else:
            theta = np.ones([N, ], dtype='complex')
            theta_DC[:, i] = copy.deepcopy(theta)
            x[0] = 1

        X_DC[:, i] = copy.deepcopy(x)
        if not x.any():
            print('Selection is failed! no device selected, Check it.')
            obj = np.inf
        else:
            index = (x == 1)
            gain = K2 / (np.abs(np.conjugate(theta) @ h_d) ** 2) * libopt.sigma
            obj = np.max(gain[index]) / (sum(K[index])) ** 2 + 4 / Ksum2 * (sum(K[~index])) ** 2

        obj_DC[i] = copy.deepcopy(obj)
        if verbose:
            print('obj={:.6f}\n'.format(obj))

    return X_DC, obj_DC, np.ones([N, ], dtype='complex'),theta_DC


# def DC_main(N, K, L, h_d, G, maxiter, iter_num, epsilon, verbose, epsilon2, x, libopt, M, K2):
#     F_DC_RIS = np.ones([N, ], dtype='complex')
#     #    theta_DC_RIS=np.zeros([L,],dtype='complex')
#     theta_DC_RIS = np.ones([L], dtype=complex)
#
#     h = np.zeros([N, K], dtype=complex)
#     for i in range(K):
#         h[:, i] = h_d[:, i] + G[:, :, i] @ theta_DC_RIS
#     obj_pre = min(np.abs(np.conjugate(F_DC_RIS) @ h) ** 2)
#     infeasible = False
#     stop = False
#     for iter in range(maxiter):
#         if verbose:
#             print('iter={}'.format(iter))
#         # Given theta, update F
#         F_DC_RIS = F_DC_RIS
#         #        print(F_DC_RIS.shape)
#         # Given F, update theta
#         theta_DC_RIS, infeasible = DC_theta(N, K, L, h_d, G, iter_num, F_DC_RIS, verbose, epsilon2)
#         h = np.zeros([N, K], dtype=complex)
#         for i in range(K):
#             h[:, i] = h_d[:, i] + G[:, :, i] @ theta_DC_RIS
#         obj = min(np.abs(np.conjugate(F_DC_RIS) @ h) ** 2)
#         if verbose:
#             print('Gain value={}'.format(obj))
#         if abs(obj - obj_pre) < epsilon or infeasible == True:
#             stop = True
#         obj_pre = obj
#         if stop:
#             break
#
#     for i in range(M):
#         h[:, i] = h_d[:, i] + G[:, :, i] @ theta_DC_RIS
#     gain = K2 / (np.abs(np.conjugate(F_DC_RIS) @ h) ** 2) * libopt.sigma
#     KK = libopt.K / np.mean(libopt.K)
#     obj = np.max(gain) / (sum(KK)) ** 2  # 问题在于，这里的K应该是libopt。K，而此时此刻这个变量K代表的是设备数
#     obj_DC_RIS = copy.deepcopy(obj)
#     return obj_DC_RIS, x, F_DC_RIS, theta_DC_RIS


# def DC_DS_RIS(libopt, h_d, G, x0, RISON, verbose):
#     N = libopt.N  # 服务器天线
#     L = libopt.L  # 反射元件
#     M = libopt.M  # 数量
#     Jmax = libopt.Jmax  # Gibbs loop
#     K = libopt.K / np.mean(libopt.K)  # normalize K to speed up floating computation
#     K2 = K ** 2
#     Ksum2 = sum(K) ** 2  # 这里总会是M的平方//a b c d的分别处以他们四个平均数，再求和，总会是数据个数
#     x = x0
#
#     maxiter = 1  # 轮换优化次数（由于这里不是轮换优化所以只需maxiter=1）
#     iter_num = 50  # 子问题迭代次数
#     epsilon = 1e-3
#     epsilon2 = 1e-8
#
#     # inital the return values
#
#     obj_new = np.zeros(Jmax + 1)
#     f_store = np.zeros([N, Jmax + 1], dtype=complex)
#     theta_store = np.zeros([L, Jmax + 1], dtype=complex)
#     x_store = np.zeros([Jmax + 1, M], dtype=int)
#
#     obj_DC_RIS, x, F_DC_RIS, theta_DC_RIS = DC_main(N, M, L, h_d, G, maxiter, iter_num, epsilon, verbose, epsilon2, x,
#                                                     libopt, M,
#                                                     K2)
#     # F_DC_RIS始终是1
#
#     # Gibbs搜索前
#     ind = 0
#     [obj_new[ind], x_store[ind, :], f, theta] = [copy.deepcopy(obj_DC_RIS), copy.deepcopy(x),
#                                                  copy.deepcopy(F_DC_RIS), copy.deepcopy(theta_DC_RIS)]
#     theta_store[:, ind] = copy.deepcopy(theta)
#     f_store[:, ind] = copy.deepcopy(f)
#
#     beta = min(1, obj_new[ind])
#
#     alpha = 0.9
#
#     f_loop = np.tile(f, (M + 1, 1))  # f_loop的每一行是第j次试验中和给定的x在第m个位置不同的策略的SCA搜索结果，这个数组用于在算法2的每次迭代中传递上次试验的结果
#     # 就如arxiv第20页的描述，这样做可以减少搜索时间
#
#     theta_loop = np.tile(theta, (M + 1, 1))
#
#     for j in range(Jmax):
#         if libopt.verbose > 1:
#             print('This is the {}-th Gibbs sampling iteration, beta= {:.6f}'.format(j + 1, beta));
#
#         # store the possible transition solution and their objectives
#         X_sample = np.zeros([M + 1, M], dtype=int)
#         Temp = np.zeros(M + 1)  # 这里的Temp实际上是论文里的J(x_j)
#         # the first transition => no change
#         X_sample[0, :] = copy.deepcopy(x)
#         Temp[0] = copy.deepcopy(obj_new[ind])  ############
#         f_loop[0] = copy.deepcopy(f)
#         theta_loop[0] = copy.deepcopy(theta)
#         # 2--M+1-th trnasition, change only 1 position
#         for m in range(M):  # 这个循环里面对每一个m位置改变后的选择策略进行SCA求解，然后将得到的式23的值以及对应的f,theta存入Temp,f_loop,theta_loop中
#             if libopt.verbose > 1:
#                 print('the {}-th:'.format(m + 1))
#             # filp the m-th position
#             x_sam = copy.deepcopy(x)
#             x_sam[m] = copy.deepcopy((x_sam[m] + 1) % 2)
#             X_sample[m + 1, :] = copy.deepcopy(x_sam);  # 将第m个位置不同于x的选择策略存进X_sample中
#             # print("j:",j,"m:",m)
#             Temp[m + 1], _, f_loop[m + 1], theta_loop[m + 1] = DC_main(N, M, L, h_d, G, maxiter, iter_num, epsilon,
#                                                                        verbose,
#                                                                        epsilon2, x_sam, libopt, M, K2)
#             # 在给定f_loop[m+1]和theta_loop[m+1]的条件下进行搜索，减少计算时间
#             # obj, x, f, theta
#             if libopt.verbose > 1:
#                 print('          sol:{} with obj={:.6f}'.format(x_sam, Temp[m + 1]))
#         temp2 = Temp;
#
#         Lambda = np.exp(-1 * temp2 / beta);
#         Lambda = Lambda / sum(Lambda);
#         while np.isnan(Lambda).any():
#             if libopt.verbose > 1:
#                 print('There is NaN, increase beta')
#             beta = beta / alpha;
#             Lambda = np.exp(-1. * temp2 / beta);
#             Lambda = Lambda / sum(Lambda);
#
#         if libopt.verbose > 1:
#             print('The obj distribution: {}'.format(temp2))
#             print('The Lambda distribution: {}'.format(Lambda))
#         kk_prime = np.random.choice(M + 1, p=Lambda)
#         x = copy.deepcopy(X_sample[kk_prime, :])
#         f = copy.deepcopy(f_loop[kk_prime])
#         theta = copy.deepcopy(theta_loop[kk_prime])
#         ind = ind + 1
#         obj_new[ind] = copy.deepcopy(Temp[kk_prime])
#         x_store[ind, :] = copy.deepcopy(x)
#         theta_store[:, ind] = copy.deepcopy(theta)
#         f_store[:, ind] = copy.deepcopy(f)
#
#         if libopt.verbose > 1:
#             print('Choose the solution {}, with objective {:.6f}'.format(x, obj_new[ind]))
#
#         if libopt.verbose:
#             print('This is the {}-th Gibbs sampling iteration, beta= {:.6f},obj={:.6f}'.format(j + 1, beta,
#                                                                                                obj_new[ind]));
#         beta = max(alpha * beta, 1e-4);
#
#     return x_store, obj_new, f_store, theta_store
