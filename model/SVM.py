import numpy as np
import os
import pickle
import sys


class SVM:
    @staticmethod
    def delta(i,j):
        if i == j: return 1
        else: return 0

    def __init__(self, xp =[], xn =[], epsilon=0, max_updates=0):
        self.xp = np.array(xp)
        self.xn = np.array(xn)
        self.epsilon = epsilon
        self.max_updates = max_updates
        self.alpha_p = np.array([0] * len(xp), dtype='int64')
        self.alpha_n = np.array([0] * len(xn), dtype='int64')
        self.kernel_degree = 4
        self.kernel = self.polynomial_kernel

    def train(self):
        # Initialization Step
        updates = 0
        self.sk_initialize()
        i_t, t_positive, m_t = self.sk_optimize()
        while not self.stop_condition(m_t)[0] and updates < self.max_updates:
            print "Current Iteration : {}. Difference : {}".format(updates, self.stop_condition(m_t)[1])
            self.sk_update(i_t, t_positive)
            i_t, t_positive, m_t = self.sk_optimize()
            updates += 1

        if self.stop_condition(m_t)[0]:
            print "Model converged after {} updates.\nModel Summay:\nFinal distance between hyperplanes : {}\nFinal Difference: {}".format(updates, m_t, self.stop_condition(m_t)[1])
        else:
            print "Model failed to converge after {} updates.\nModel Summay:\nFinal distance between hyperplanes : {}\nFinal Difference: {}".format(self.max_updates, m_t, self.stop_condition(m_t)[1])

        # Eliminating values corresponding to alpha's with value 0
        self.xp = self.xp[self.alpha_p != 0]
        self.xn = self.xn[self.alpha_n != 0]
        self.alpha_p = self.alpha_p[self.alpha_p != 0]
        self.alpha_n = self.alpha_n[self.alpha_n != 0]

    def test(self, test_x):
        score = 0
        test_x = np.array(test_x)

        # Positive alpha
        score += np.sum(self.kernel(self.xp, test_x) * self.alpha_p)

        # Negative alpha
        score += np.sum(self.kernel(self.xn, test_x) * self.alpha_n * -1)

        score += (self.B - self.A)/2.0

        if score>0 :
            return 1
        else:
            return 0

    def sk_initialize(self):
        i_1, j_1 = 0, 0  # can be randomized
        self.alpha_p[i_1] = 1
        self.alpha_n[j_1] = 1
        self.A = self.kernel(self.xp[i_1], self.xp[i_1])
        self.B = self.kernel(self.xn[j_1], self.xn[j_1])
        self.C = self.kernel(self.xp[i_1], self.xn[j_1])
        self.D_P = self.kernel(self.xp, self.xp[i_1])
        self.D_N = self.kernel(self.xn, self.xp[i_1])
        self.E_P = self.kernel(self.xp, self.xn[j_1])
        self.E_N = self.kernel(self.xn, self.xn[j_1])

    def sk_optimize(self):
        i_t, t_positive, m_t = 0, False, sys.maxsize
        for p in xrange(len(self.xp)):
            m_i = self.m_positive(p)
            if m_i < m_t:
                m_t = m_i
                i_t = p
                t_positive = True

        for n in xrange(len(self.xn)):
            m_i = self.m_negative(n)
            if m_i < m_t:
                m_t = m_i
                i_t = n
                t_positive = False

        return i_t, t_positive, m_t

    def sk_update(self, i_t, t_positive):
        if t_positive:
            xt_xt = self.kernel(self.xp[i_t], self.xp[i_t])
            # sample with minimum m is positive
            # calculate q
            q_numerator = self.A - self.D_P[i_t] + self.E_P[i_t] - self.C
            q_denominator = self.A + xt_xt - 2*(self.D_P[i_t] - self.E_P[i_t])
            q = min(1, q_numerator / (1.0 * q_denominator))

            # Udpating alpha values
            self.alpha_p = (1 - q) * self.alpha_p
            self.alpha_p[i_t] += q

            # Update A and C
            self.A = self.A * (1 - q) ** 2 + 2 * (1 - q) * q * self.D_P[i_t] + (q ** 2) * xt_xt
            self.C = (1 - q) * self.C + q * self.E_P[i_t]

            # Updating D
            self.D_N = (1 - q) * self.D_N + q * self.kernel(self.xn, self.xp[i_t])
            self.D_P = (1 - q) * self.D_P + q * self.kernel(self.xp, self.xp[i_t])


            """
            # Update alpha and D for positive samples of X
            for i in xrange(len(self.xp)):
                self.alpha_p[i] = ((1-q) * self.alpha_p[i]) + (q*SVM.delta(i, i_t))
                self.D_P[i] = (1-q)* self.D_P[i] + q*xt_xt
         
            # Update D for negative samples of X
            for i in xrange(len(self.xn)):
                
            """

        else:
            xt_xt = self.kernel(self.xn[i_t], self.xn[i_t])
            # sample with minimum m is negative
            # calculate q
            q_numerator = self.B - self.E_N[i_t] + self.D_N[i_t] - self.C
            q_denominator = self.B + xt_xt - 2 * (self.E_N[i_t] - self.D_N[i_t])
            q = min(1, q_numerator / (1.0 * q_denominator))

            # Update alpha for negative samples of X
            self.alpha_n = (1 - q) * self.alpha_n
            self.alpha_n[i_t] += q

            # Update B and C
            self.B = self.B * (1 - q) ** 2 + 2 * (1 - q) * q * self.E_N[i_t] + (q ** 2) * xt_xt
            self.C = (1 - q) * self.C + q * self.D_N[i_t]

            # Update E for all X
            self.E_N = (1 - q) * self.E_N + q * self.kernel(self.xn, self.xn[i_t])
            self.E_P = (1 - q) * self.E_P + q * self.kernel(self.xp, self.xn[i_t])



            """
            for i in xrange(len(self.xn)):
                self.alpha_n[i] = ((1 - q) * self.alpha_n[i]) + (q * SVM.delta(i, i_t))
                self.E_N[i] = (1 - q) * self.E_N[i] + q * xt_xt
           
            for i in xrange(len(self.xp)):
                self.E_P[i] = (1 - q) * self.E_P[i] + q * xt_xt
            """

    def stop_condition(self, m_t):
        hp_distance = np.sqrt(self.A + self.B - 2*self.C)
        if hp_distance - m_t < self.epsilon:
            return True, hp_distance-m_t
        else:
            return False, hp_distance-m_t

    def m_positive(self, p):
        numerator = self.D_P[p] - self.E_P[p] + self.B - self.C
        denominator = np.sqrt(self.A + self.B -2*self.C)
        return numerator/(denominator*1.0)

    def m_negative(self, n):
        numerator = self.E_N[n] - self.D_N[n] + self.A - self.C
        denominator = np.sqrt(self.A + self.B - 2 * self.C)
        return numerator / (denominator * 1.0)

    def polynomial_kernel(self, x, y):
        return (np.dot(x, y) + 1) ** self.kernel_degree

    def save_model(self, model_file_name):
        os.chdir(".")
        model = {
            "xp": self.xp,
            "xn": self.xn,
            "A": self.A,
            "B": self.B,
            "alpha_p": self.alpha_p,
            "alpha_n": self.alpha_n
        }

        try:
            with open(model_file_name, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            print "ERROR: Exception while trying to save model. " + str(e)

    def load_model(self, model_file_name):
        os.chdir(".")
        try:
            with open(model_file_name, "rb") as f:
                model = pickle.load(f)
                self.alpha_n = model["alpha_n"]
                self.alpha_p = model["alpha_p"]
                self.xp = model["xp"]
                self.xn = model["xn"]
                self.A = model["A"]
                self.B = model["B"]
        except Exception as e:
            print "ERROR: Exception while loading saved model. " +str(e)

