from brian2 import *
import matplotlib.pyplot as plt


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')



start_scope()


#this control the refractory behaviour. At reset, v = -1, and it goes up with a as
#da/dt = -a/taus

taus = 2.5 * ms

eqs = '''
dv/dt = (I-v)/tau + a/taus : 1
da/dt = -a/taus : 1
I : 1
tau : second
'''

eqs_reset = '''v = -1
                a = 1'''

G = NeuronGroup(2, eqs, threshold='v>1', reset=eqs_reset, refractory=1*ms, method='exact')
G.I = [2, 0]
G.tau = [10, 100]*ms
G.v   = [0, 0]
G.a   = [1, 1]


# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, on_pre='v_post += 0.2')
S.connect(i=0, j=1)

visualise_connectivity(S)

M = StateMonitor(G, ['v', 'a'], record=True)

run(100*ms)


plt.figure()
plt.plot(M.t/ms, M.v[0], label='Neuron 0')
plt.plot(M.t/ms, M.v[1], label='Neuron 1')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend();


plt.figure()
plt.plot(M.t/ms, M.a[0], label='A 0')
plt.plot(M.t/ms, M.a[1], label='A 1')


plt.show();