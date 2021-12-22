from brian2 import *
import matplotlib.pyplot as plt

start_scope()


#tau controls the dynamic of the membrane v(t)
#the smaller, the faster the membrane to evolve
#so it goes faster up (if it has an input current I) or down


#a(t) do not know yet what it meas

#taupre and taupost controls the dynamics of the traces
#this means that the bigger they are, the longer the traces
#apost and apre, thus the longer it is the window of learning,
#i.e, far away in time spikes are involved in the STDP rule

taus = 2.5 * ms

eqs = '''
dv/dt = (I-v)/tau + a/taus : 1
da/dt = -a/taus : 1
I : 1
tau : second
'''

eqs_reset = '''v = -1
                a = 1'''


G = NeuronGroup(3, eqs, threshold='v>1', reset=eqs_reset, refractory=1*ms, method='exact')
G.I = [2, 2, 0]
G.tau = [10, 15, 100]*ms
G.v   = [0, -1, 0]
G.a   = [0, 0, 0]
taupre = taupost = 1*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G,
                    '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.05
                    w = w+apost
                    ''',
                    on_post='''
                    apost += -0.05
                    w = w+apre
                    ''',
                    method='linear')

S.connect(i=[0,1], j=2)
S.w = '0.08'
S.delay = 1*ms


Mg = StateMonitor(G, ['v', 'a'], record=True)
Ms = StateMonitor(S, ['w', 'apre', 'apost'], record=True)

run(100*ms)

plt.figure()
plt.subplot(411)
plt.plot(Mg.t/ms, Mg.v[0], label='Neuron 0')
plt.plot(Mg.t/ms, Mg.v[1], label='Neuron 1')
plt.plot(Mg.t/ms, Mg.v[2], label='Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend();

plt.subplot(412)
plt.plot(Ms.t/ms, Ms.w[0], label='W 0')
plt.plot(Ms.t/ms, Ms.w[1], label='W 1')
plt.legend();


plt.subplot(413)
plt.plot(Ms.t/ms, Ms.apre[0], label='apre 0')
plt.plot(Ms.t/ms, Ms.apost[0], label='apost 0')
plt.legend();

plt.subplot(414)
plt.plot(Ms.t/ms, Ms.apre[1], label='apre 1')
plt.plot(Ms.t/ms, Ms.apost[1], label='apost 1')
plt.legend();


plt.show();