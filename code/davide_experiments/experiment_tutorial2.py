from brian2 import *
import matplotlib.pyplot as plt

start_scope()

taupre = taupost = 20*ms
wmax = 0.01
Apre = 0.01
Apost = -Apre*taupre/taupost*1.05

G = NeuronGroup(2, 'v:1', threshold='t>(1+i)*10*ms', refractory=100*ms)

S = Synapses(G, G,
             '''
             w : 1
             dapre/dt = -apre/taupre : 1 (clock-driven)
             dapost/dt = -apost/taupost : 1 (clock-driven)
             ''',
             on_pre='''
             v_post += w
             apre += Apre
             w = clip(w+apost, 0, wmax)
             ''',
             on_post='''
             apost += Apost
             w = clip(w+apre, 0, wmax)
             ''', method='linear')
S.connect(i=0, j=1)
M = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

run(30*ms)

plt.figure(figsize=(4, 8))
plt.subplot(211)
plt.plot(M.t/ms, M.apre[0], label='apre')
plt.plot(M.t/ms, M.apost[0], label='apost')
plt.legend()
plt.subplot(212)
plt.plot(M.t/ms, M.w[0], label='w')
plt.legend(loc='best')
plt.xlabel('Time (ms)');

plt.show()
