# -*- coding: utf-8 -*-
import numpy as np
import random

class HMM():
    def __init__(self):
        pass
    
    def _forward(self, seq):
        m = len(seq)
        n = len(self.A)
    
        dp = np.zeros((m, n))

        for t in range(m):
            for i in range(n):
                if t == 0:
                    dp[t][i] = self.Pi[i] * self.B[i][seq[0]]
                    
                else:
                    for j in range(n):
                        dp[t][i] += dp[t-1][j] * self.A[j][i]
                    dp[t][i] *= self.B[i][seq[t]]
        
        return (sum(dp[-1]), dp)
    
    def _backward(self, seq):
        m = len(seq)
        n = len(self.A)
        
        dp = np.zeros((m, n))
        
        for t in range(m-1, -1, -1):
            for i in range(n):
                if t == m - 1:
                    dp[t][i] = 1
                
                else:
                    for j in range(n):
                        dp[t][i] += self.A[i][j] * self.B[j][seq[t+1]] * dp[t+1][j]
        prob = 0
        for i in range(n):
            prob += self.Pi[i] * self.B[i][seq[0]] * dp[0][i]
        return (prob, dp)
    
    def _init_hmm_matrix(self, num_states, num_observations):
        random.seed(0)
        
        self.A = np.zeros((num_states, num_states))
        self.B = np.zeros((num_states, num_observations))
        self.Pi = np.array([1.0 / num_states] * num_states)
        
        for i in range(num_states):
            numbers = [random.randint(0, 100) for _ in range(num_states)]
            sums = sum(numbers)
            for j in range(num_states):
                self.A[i][j] = numbers[j] / sums
                
        for i in range(num_states):
            numbers = [random.randint(0, 100) for _ in range(num_observations)]
            sums = sum(numbers)
            for j in range(num_observations):
                self.B[i][j] = numbers[j] / sums
    
    def _cal_gamma(self, t, i, alpha, beta):
        numerator = alpha[t][i] * beta[t][i]
        denominator = 0

        for j in range(len(alpha[0])):
            denominator += alpha[t][j] * beta[t][j]

        return numerator/denominator
    
    def _cal_xi(self, t, i, j, alpha, beta, seq):
        numerator = alpha[t][i] * self.A[i][j] * self.B[j][seq[t+1]] * beta[t+1][j]
        denominator = 0

        for i in range(len(alpha[0])):
            for j in range(len(alpha[0])):
                denominator += alpha[t][i] * self.A[i][j] * self.B[j][seq[t+1]] * beta[t+1][j]

        return numerator/denominator
    
    def _approximate(self, seq):
        hidden_states = []
        
        _, alpha = self._forward(seq)
        _, beta = self._backward(seq)
        
        for t in range(len(seq)):
            max_gamma = 0
            max_i = 0
            for i in range(len(self.A)):
                gamma = self._cal_gamma(t, i, alpha, beta)
                if gamma > max_gamma:
                    max_gamma = gamma
                    max_i = i
            hidden_states.append(max_i)
        return hidden_states
        
    def _viterbi(self, seq):
        m = len(seq)
        n = len(self.A)
        
        dp = np.zeros((m, n))
        track = np.zeros((m, n), dtype=np.int32)
        
        hidden_states = []
        
        for t in range(m):
            for i in range(n):
                if t == 0:
                    dp[t][i] = self.Pi[i] * self.B[i][seq[0]]
                else:
                    for j in range(n):
                        tmp = dp[t-1][j] * self.A[j][i] * self.B[i][seq[t]]
                        if tmp > dp[t][i]:
                            dp[t][i] = tmp
                            track[t][i] = j
        
        last_state = np.argmax(dp[-1, :])
        hidden_states.append(last_state)

        for t in range(m - 1, 0, -1):
            last_state = track[t][last_state]
            hidden_states.append(last_state)
        
        return hidden_states[::-1]
        
    def cal_prob(self, seq, A, B, Pi, method="forward"):
        self.A = A
        self.B = B
        self.Pi = Pi
        
        if method == "forward":
            return self._forward(seq)[0]
        
        elif method == "backward":
            return self._backward(seq)[0]
        
        else:
            raise NameError("You need pass a string parameter: forward or backward:)")
        
    def fit(self, seq, num_states, num_observations, num_iterations=1000, seed=None):
        if seed:
            random.seed(seed)
            
        T = len(seq)
        
        self._init_hmm_matrix(num_states, num_observations)

        for i in range(num_iterations):
            tmp_A = np.zeros((num_states,num_states))
            tmp_B = np.zeros((num_states, num_observations))
            tmp_Pi = np.zeros((num_states,))
            
            _, alpha = self._forward(seq)
            _, beta = self._backward(seq)

            for i in range(num_states):
                for j in range(num_states):
                    numerator=0.0
                    denominator=0.0
                    for t in range(T-1):
                        numerator += self._cal_xi(t, i, j, alpha, beta, seq)
                        denominator += self._cal_gamma(t, i, alpha, beta)
                    tmp_A[i][j] = numerator/denominator
            
            for i in range(num_states):
                for j in range(num_observations):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(T):
                        if j == seq[t]:
                            numerator += self._cal_gamma(t, i, alpha, beta)
                        denominator += self._cal_gamma(t, i, alpha, beta)
                    tmp_B[i][j] = numerator / denominator
        
            for i in range(num_states):
                tmp_Pi[i] = self._cal_gamma(0,i, alpha, beta)
                
            self.A = tmp_A
            self.B = tmp_B
            self.Pi = tmp_Pi
            
        return (self.A, self.B, self.Pi)
    
    def predict(self, seq, A, B, Pi, method="viterbi"):
        self.A = A
        self.B = B
        self.Pi = Pi
        
        if method == "approximate":
            return self._approximate(seq)
        
        elif method == "viterbi":
            return self._viterbi(seq)
        
        else:
            raise NameError("You need pass a string parameter: approximate or viterbi:)")
            
        
if __name__ == "__main__":        
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
    Pi = np.array([0.2, 0.4, 0.4])
    
    hmm = HMM()
    
    seq = [0, 1, 0]
    print(hmm.cal_prob(seq, A, B, Pi, "backward"))
    print(hmm.predict(seq, A, B, Pi, method="123"))