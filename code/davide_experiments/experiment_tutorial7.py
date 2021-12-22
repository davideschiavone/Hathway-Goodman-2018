from brian2 import *
import matplotlib.pyplot as plt
import numpy as np


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

np.random.seed(2021)
input_samples = 100*np.random.random_sample((100, 2))
input_samples = (input_samples.astype(int))/10

#make the first 2 samples very close in time

input_samples[0,:] = [9,9.5]
input_samples[1,:] = [0,0.5]


class0 = input_samples[:,0] > input_samples[:,1]
class1 = input_samples[:,0] < input_samples[:,1]

plt.figure()
plt.plot(input_samples[class0,0], input_samples[class0,1], 'bo')

plt.plot(input_samples[class1,0], input_samples[class1,1], 'r*')


indices = []
times   = []
period  = 0

for val in input_samples:
    print(val)
    indices.append(0)
    times.append(val[0]+period)
    indices.append(1)
    times.append(val[1]+period)
    period = period + 30

print(size(times))


indices = numpy.array(indices)
times = numpy.array(times)

max_time = max(times)
print("Max time " + str(max_time))

N0    = SpikeGeneratorGroup(2, indices, times*ms)
N0mon = SpikeMonitor(N0)

taupre = 5 * ms
taupost = 5 * ms


eqs = '''
dv/dt = -v/tau + a/taus: 1
da/dt = -a/taus : 1
I : 1
tau : second
taus : second
'''
eqs_reset = '''
                v = -1
                a = +1
            '''

Threshold = 0.8

N1      = NeuronGroup(2, eqs, threshold='v>Threshold', reset=eqs_reset, refractory=5*ms, method='exact')
N1mon   = SpikeMonitor(N1)
N1state = StateMonitor(N1, ['v'], record=True)

N1.tau = [10, 10]*ms
N1.taus = [3, 3]*ms
N1.v   = [0, 0]
N1.a   = [0, 0]
wmin   = -1
wmax   = 1

S = Synapses(N0, N1,
                    '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.031
                    w =  clip(w+apost, wmin, wmax)
                    ''',
                    on_post='''
                    apost += -0.033
                    w = clip(w+apre, wmin, wmax)
                    ''',
                    method='linear')

S.connect(i=[0,1,0,1], j=[0,0,1,1])
S.w[0, 0] = 1
S.w[1, 0] = -1
S.w[0, 1] = 0.85
S.w[1, 1] = -0.85

S.delay = 1*ms
Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
visualise_connectivity(S)


#run((max_time*1.1)*ms)

run(400*ms)

#each time step is by default 100us
#so 130ms are 1300 steps

plot_start_ms = 0
plot_stop_ms  = plot_start_ms + 200

plot_start_index = plot_start_ms*10
plot_stop_index  = plot_stop_ms*10

N0mon_times_n0_plot   = N0mon.spike_trains()[0][N0mon.spike_trains()[0]/ms < plot_stop_ms]
N0mon_times_n1_plot   = N0mon.spike_trains()[1][N0mon.spike_trains()[1]/ms < plot_stop_ms]
N0mon_nspikes_n0_plot = np.ones(size(N0mon_times_n0_plot))*0
N0mon_nspikes_n1_plot = np.ones(size(N0mon_times_n1_plot))*1

N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*2

plt.figure()

ax1= plt.subplot(411)
plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.r')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.subplot(412, sharex = ax1)
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(413, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='N1 W 0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='N1 W 1')
plt.legend();


plt.subplot(414, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[0][plot_start_index:plot_stop_index], label='N1 W 0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[1][plot_start_index:plot_stop_index], label='N1 W 1')
plt.legend();

stop()

indices = np.append(indices, N1mon_nspikes_n0_plot)
times   = np.append(times, N1mon_times_n0_plot/ms)

N0    = SpikeGeneratorGroup(3, indices, times*ms)
N0mon = SpikeMonitor(N0)

N1      = NeuronGroup(1, eqs, threshold='v>Threshold', reset=eqs_reset, refractory=5*ms, method='exact')
N1mon   = SpikeMonitor(N1)
N1state = StateMonitor(N1, ['v'], record=True)
N1.tau = [10]*ms
N1.taus = [3]*ms
N1.v   = [0]
N1.a   = [0]

S = Synapses(N0, N1,
                    '''
                    w : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.031
                    w =  clip(w+apost, wmin, wmax)
                    ''',
                    on_post='''
                    apost += -0.033
                    w = clip(w+apre, wmin, wmax)
                    ''',
                    method='linear')

S.connect(i=[0,1,2], j=[0,0,0])
S.w[0, 0] = 0.5
S.w[1, 0] = 0.5
S.w[2, 0] = -1
S.delay[0, 0] = 5*ms
S.delay[1, 0] = 5*ms
S.delay[2, 0] = 0*ms
Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
visualise_connectivity(S)

run(400*ms)

plot_start_ms = 0
plot_stop_ms  = plot_start_ms + 200

plot_start_index = plot_start_ms*10
plot_stop_index  = plot_stop_ms*10

N0mon_times_n0_plot   = N0mon.spike_trains()[0][N0mon.spike_trains()[0]/ms < plot_stop_ms]
N0mon_times_n1_plot   = N0mon.spike_trains()[1][N0mon.spike_trains()[1]/ms < plot_stop_ms]
N0mon_times_n2_plot   = N0mon.spike_trains()[2][N0mon.spike_trains()[2]/ms < plot_stop_ms]
N0mon_nspikes_n0_plot = np.ones(size(N0mon_times_n0_plot))*0
N0mon_nspikes_n1_plot = np.ones(size(N0mon_times_n1_plot))*1
N0mon_nspikes_n2_plot = np.ones(size(N0mon_times_n2_plot))*2

N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*3

plt.figure()

ax1= plt.subplot(411)
plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
plt.plot(N0mon_times_n2_plot/ms, N0mon_nspikes_n2_plot, '.r')
plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.b')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.subplot(412, sharex = ax1)
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(413, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='N1 W 0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='N1 W 1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[2][plot_start_index:plot_stop_index], label='N1 W 2')
plt.legend();


plt.subplot(414, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[0][plot_start_index:plot_stop_index], label='N1 W 0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[1][plot_start_index:plot_stop_index], label='N1 W 1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[2][plot_start_index:plot_stop_index], label='N1 W 2')
plt.legend();


plt.show();
