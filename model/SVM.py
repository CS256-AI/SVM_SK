import numpy as np
import sys


class SVM:
    @staticmethod
    def delta(i,j):
        if i == j: return 1
        else: return 0

    def __init__(self, xp, xn, epsilon, max_updates):
        self.xp = np.array(xp)
        self.xn = np.array(xn)
        self.epsilon = epsilon
        self.max_updates = max_updates
        self.alpha_p = [0] * len(xp)
        self.alpha_n = [0] * len(xn)
        self.kernel_degree = 4
        self.kernel = self.polynomial_kernel

    def print_state(self):
        print "XP : "
        print self.xp
        print "XN : "
        print self.xn
        print "Alpha P : "
        print self.alpha_p
        print "Alpha N : "
        print self.alpha_n
        print "A: "
        print self.A
        print "B: "
        print self.B
        print "C: "
        print self.C
        print "D_P: "
        print self.D_P
        print "D_N: "
        print self.D_N
        print "E_P: "
        print self.E_P
        print "E_N: "
        print self.E_N

    def train(self):
        # Initialization Step
        updates = 0
        self.sk_initialize()
        self.print_state()
        i_t, t_positive, m_t = self.sk_optimize()
        print m_t, i_t, t_positive
        while not self.stop_condition(m_t) and updates < self.max_updates:
            self.sk_update(i_t, t_positive)
            i_t, t_positive, m_t = self.sk_optimize()
            updates += 1

    def sk_initialize(self):
        i_x, i_y = 0, 0  # can be randomized
        self.alpha_p[i_x] = 1
        self.alpha_n[i_y] = 1
        self.A = self.kernel(self.xp[i_x], self.xp[i_x])
        self.B = self.kernel(self.xn[i_y], self.xn[i_y])
        self.C = self.kernel(self.xp[i_x], self.xn[i_y])
        self.D_P = self.kernel(self.xp[i_x], self.xp)
        self.D_N = self.kernel(self.xp[i_x], self.xn)
        self.E_P = self.kernel(self.xn[i_y], self.xp)
        self.E_N = self.kernel(self.xn[i_y], self.xn)

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
        xt_xt = self.kernel(self.xp[i_t], self.xp[i_t])
        if t_positive:
            # sample with minimum m is positive
            # calculate q
            q_numerator = self.A - self.D_P[i_t] + self.E_P[i_t] - self.C
            q_denominator = self.A + xt_xt - 2*(self.D_P[i_t] - self.E_P[i_t])
            q = min(1, q_numerator / (1.0 * q_denominator))

            self.alpha_p = (1-q) * self.alpha_p
            self.alpha_p[i_t] += q
            self.D_N = (1 - q) * self.D_N + q * xt_xt
            self.D_P = (1 - q) * self.D_P + q * xt_xt

            # Update A and C
            self.A = self.A * (1 - q) ** 2 + 2 * (1 - q) * self.D_P[i_t] + (q ** 2) * xt_xt
            self.C = (1 - q) * self.C + q * self.E_P[i_t]

            """
            # Update alpha and D for positive samples of X
            for i in xrange(len(self.xp)):
                self.alpha_p[i] = ((1-q) * self.alpha_p[i]) + (q*SVM.delta(i, i_t))
                self.D_P[i] = (1-q)* self.D_P[i] + q*xt_xt
         
            # Update D for negative samples of X
            for i in xrange(len(self.xn)):
                
            """

        else:
            # sample with minimum m is negative
            # calculate q
            q_numerator = self.B - self.E_N[i_t] + self.D_N[i_t] - self.C
            q_denominator = self.B + xt_xt - 2 * (self.E_N[i_t] - self.D_N[i_t])
            q = min(1, q_numerator / (1.0 * q_denominator))

            # Update alpha for negative samples of X
            self.alpha_n = (1 - q) * self.alpha_n
            self.alpha_n[i_t] += q

            # Update E for all X
            self.E_N = (1 - q) * self.E_N + q * xt_xt
            self.E_P = (1-q) * self.E_P + q * xt_xt

            # Update B and C
            self.B = self.B * (1 - q) ** 2 + 2 * (1 - q) * self.E_N[i_t] + (q ** 2) * xt_xt
            self.C = (1 - q) * self.C + q * self.D_N[i_t]

            """
            for i in xrange(len(self.xn)):
                self.alpha_n[i] = ((1 - q) * self.alpha_n[i]) + (q * SVM.delta(i, i_t))
                self.E_N[i] = (1 - q) * self.E_N[i] + q * xt_xt
           
            for i in xrange(len(self.xp)):
                self.E_P[i] = (1 - q) * self.E_P[i] + q * xt_xt
            """

    def stop_condition(self, m_t):
        if ((self.A - self.B - 2*self.C)**(1/2.0)) - m_t < self.epsilon:
            return True
        else:
            return False

    def m_positive(self, p):
        numerator = self.D_P[p] - self.E_P[p] + self.B - self.C
        denominator = (self.A + self.B -2*self.C) ** (1/2.0)
        return numerator/(denominator*1.0)

    def m_negative(self, n):
        numerator = self.E_N[n] - self.D_N[n] + self.A - self.C
        denominator = (self.A + self.B - 2 * self.C) ** (1 / 2.0)
        return numerator / (denominator * 1.0)

    def polynomial_kernel(self, x, y):
        return (np.dot(np.transpose(x), y) + 1) ** self.kernel_degree