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
input_samples = 100*np.random.random_sample((500, 4))
input_samples = (input_samples.astype(int))/10

indices   = []
times     = []
period    = 0
p_del     = 0.4
for val in input_samples:
    print(val)

    if(np.random.uniform(0, 1) > p_del):
        indices.append(0)
        times.append(val[0]+period)
    if(np.random.uniform(0, 1) > p_del):
        indices.append(1)
        times.append(val[1]+period)
    if(np.random.uniform(0, 1) > p_del):
        indices.append(2)
        times.append(val[2]+period)
    if(np.random.uniform(0, 1) > p_del):
        indices.append(3)
        times.append(val[3]+period)
    period = period + 30

print(size(times))


indices = numpy.array(indices)
times = numpy.array(times)

max_time = max(times)
print("Max time " + str(max_time))

N0    = SpikeGeneratorGroup(4, indices, times*ms)
N0mon = SpikeMonitor(N0)

taupre = 2 * ms
taupost = 5 * ms


eqs = '''
dv/dt = -v/tau + a/taus: 1
da/dt = -a/taus : 1
ddynThreshold/dt = (0.8-dynThreshold)/taut : 1
tau : second
taus : second
taut : second
'''
eqs_reset = '''
                v = -1
                a = +1
                dynThreshold = dynThreshold+0.0
            '''

Threshold = 0.8

N1      = NeuronGroup(2, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=10*ms, method='exact')
N1mon   = SpikeMonitor(N1)
N1state = StateMonitor(N1, ['v', 'dynThreshold'], record=True)

N1.tau  = [10, 10]*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                 #you can also increase the spacex0x1 and keep tau to 10ms for example

N1.taus = [30, 30]*ms
N1.taut = [50, 50]*ms
N1.v    = [0]
N1.a    = [0]
N1.dynThreshold = [Threshold]

S = Synapses(N0, N1,
                    '''
                    w : 1
                    wmin : 1
                    wmax : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.1
                    w =  clip(w+apost, wmin, wmax)
                    ''',
                    on_post='''
                    apost += -0.25
                    w = clip(w+apre, wmin, wmax)
                    ''',
                    method='linear')

S.connect(i=[0,1,2,3,0,1,2,3], j=[0,0,0,0,1,1,1,1])
S.w[0, 0] = 1/5+1/8
S.w[1, 0] = 1/5+1/8
S.w[2, 0] = 1/5+1/8
S.w[3, 0] = 1/5+1/8
S.w[0, 1] = 1/5+1/8
S.w[1, 1] = 1/5+1/8
S.w[2, 1] = 1/5+1/8
S.w[3, 1] = 1/5+1/8
S.wmax = 1
S.wmin = 0
S.delay[:, 0] = 1*ms
S.delay[:, 1] = 2*ms

Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
visualise_connectivity(S)

S2 = Synapses(N1, N1,
                    '''
                    w : 1
                    ''',
                    on_pre='''
                    v_post += w
                    ''',
                    method='linear')

S2.connect(i=[0], j=[1])
S2.w[0, 1] = -1
S2.delay[0, 1] = 0*ms
S2state  = StateMonitor(S2, ['w'], record=True)
visualise_connectivity(S2)

N2      = NeuronGroup(1, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=10*ms, method='exact')
N2mon   = SpikeMonitor(N2)
N2state = StateMonitor(N2, ['v', 'dynThreshold'], record=True)

N2.tau  = [10]*ms #fast such that cumulative output membrana forgets quickly, otherwise all the neurons get premiated
                 #you can also increase the spacex0x1 and keep tau to 10ms for example

N2.taus = [30]*ms
N2.taut = [50]*ms
N2.v    = [0]
N2.a    = [0]
N2.dynThreshold = [Threshold]

S3 = Synapses(N1, N2,
                    '''
                    w : 1
                    wmin : 1
                    wmax : 1
                    dapre/dt = -apre/taupre : 1 (clock-driven)
                    dapost/dt = -apost/taupost : 1 (clock-driven)
                    ''',
                    on_pre='''
                    v_post += w
                    apre += 0.1
                    w =  clip(w+apost, wmin, wmax)
                    ''',
                    on_post='''
                    apost += -0.25
                    w = clip(w+apre, wmin, wmax)
                    ''',
                    method='linear')

S3.connect(i=[0,1], j=[0,0])
S3.w[0, 0] = 0.7
S3.w[1, 0] = 0.7

S3.wmax = 1
S3.wmin = 0
S3.delay = 1*ms

S3state  = StateMonitor(S3, ['w', 'apre', 'apost'], record=True)
visualise_connectivity(S3)

#run((max_time*1.1)*ms)

run(15000*ms)

#each time step is by default 100us
#so 130ms are 1300 steps

plot_start_ms = 0
plot_stop_ms  = plot_start_ms + 15000

plot_start_index = plot_start_ms*10
plot_stop_index  = plot_stop_ms*10

N0mon_times_n0_plot   = N0mon.spike_trains()[0][N0mon.spike_trains()[0]/ms < plot_stop_ms]
N0mon_times_n1_plot   = N0mon.spike_trains()[1][N0mon.spike_trains()[1]/ms < plot_stop_ms]
N0mon_times_n2_plot   = N0mon.spike_trains()[2][N0mon.spike_trains()[2]/ms < plot_stop_ms]
N0mon_times_n3_plot   = N0mon.spike_trains()[3][N0mon.spike_trains()[3]/ms < plot_stop_ms]

N0mon_nspikes_n0_plot = np.ones(size(N0mon_times_n0_plot))*0
N0mon_nspikes_n1_plot = np.ones(size(N0mon_times_n1_plot))*1
N0mon_nspikes_n2_plot = np.ones(size(N0mon_times_n2_plot))*2
N0mon_nspikes_n3_plot = np.ones(size(N0mon_times_n3_plot))*3

N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*4
N1mon_times_n1_plot   = N1mon.spike_trains()[1][N1mon.spike_trains()[1]/ms < plot_stop_ms]
N1mon_nspikes_n1_plot = np.ones(size(N1mon_times_n1_plot))*5

N2mon_times_n0_plot   = N2mon.spike_trains()[0][N2mon.spike_trains()[0]/ms < plot_stop_ms]
N2mon_nspikes_n0_plot = np.ones(size(N2mon_times_n0_plot))*6

plt.figure()

ax1= plt.subplot(511)
plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
plt.plot(N0mon_times_n2_plot/ms, N0mon_nspikes_n2_plot, '.k')
plt.plot(N0mon_times_n3_plot/ms, N0mon_nspikes_n3_plot, '.k')
plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.r')
plt.plot(N1mon_times_n1_plot/ms, N1mon_nspikes_n1_plot, '.r')
plt.plot(N2mon_times_n0_plot/ms, N2mon_nspikes_n0_plot, '.b')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index');

plt.subplot(512, sharex = ax1)
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.dynThreshold[0][plot_start_index:plot_stop_index], label='Threshold')
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[1][plot_start_index:plot_stop_index], label='N1,1')
plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.dynThreshold[1][plot_start_index:plot_stop_index], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(513, sharex = ax1)
plt.plot(N2state.t[plot_start_index:plot_stop_index]/ms, N2state.v[0][plot_start_index:plot_stop_index], label='N2,0')
plt.plot(N2state.t[plot_start_index:plot_stop_index]/ms, N2state.dynThreshold[0][plot_start_index:plot_stop_index], label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('v')

plt.subplot(514, sharex = ax1)
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='0-0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='1-0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[2][plot_start_index:plot_stop_index], label='2-0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[3][plot_start_index:plot_stop_index], label='3-0')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[4][plot_start_index:plot_stop_index], label='0-1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[5][plot_start_index:plot_stop_index], label='1-1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[6][plot_start_index:plot_stop_index], label='2-1')
plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[7][plot_start_index:plot_stop_index], label='3-1')
plt.legend();

plt.subplot(515, sharex = ax1)
plt.plot(S3state.t[plot_start_index:plot_stop_index]/ms, S3state.w[0][plot_start_index:plot_stop_index], label='0-0')
plt.plot(S3state.t[plot_start_index:plot_stop_index]/ms, S3state.w[1][plot_start_index:plot_stop_index], label='1-0')
plt.legend();

stop()

#indices = np.append(indices, N1mon_nspikes_n0_plot)
#times   = np.append(times, N1mon_times_n0_plot/ms)
#
#N0    = SpikeGeneratorGroup(3, indices, times*ms)
#N0mon = SpikeMonitor(N0)
#
#N1      = NeuronGroup(1, eqs, threshold='v>dynThreshold', reset=eqs_reset, refractory=5*ms, method='exact')
#N1mon   = SpikeMonitor(N1)
#N1state = StateMonitor(N1, ['v'], record=True)
#N1.tau  = [2]*ms
#N1.taus = [3]*ms
#N1.v    = [0]
#N1.a    = [0]
#N1.dynThreshold = [Threshold]
#
#S = Synapses(N0, N1,
#                    '''
#                    w : 1
#                    wmin : 1
#                    wmax : 1
#                    dapre/dt = -apre/taupre : 1 (clock-driven)
#                    dapost/dt = -apost/taupost : 1 (clock-driven)
#                    ''',
#                    on_pre='''
#                    v_post += w
#                    apre += 0.1
#                    w =  clip(w+apost, wmin, wmax)
#                    ''',
#                    on_post='''
#                    apost += -0.25
#                    w = clip(w+apre, wmin, wmax)
#                    ''',
#                    method='linear')
#
#S.connect(i=[0,1,2], j=[0,0,0])
#S.w[0, 0] = 0.8
#S.w[1, 0] = 0.8
#S.w[2, 0] = -0.6
#S.delay[0, 0] = 5*ms
#S.delay[1, 0] = 5*ms
#S.delay[2, 0] = 0*ms
##the output of the N2 (which is the one that learned x1>x2) should be faster to inhibit the membrane
#S.wmax[:, 0] = 1
#S.wmin[:, 0] = -1
#
#
#
#Sstate  = StateMonitor(S, ['w', 'apre', 'apost'], record=True)
#visualise_connectivity(S)
#
#run(400*ms)
#
#plot_start_ms = 0
#plot_stop_ms  = plot_start_ms + 200
#
#plot_start_index = plot_start_ms*10
#plot_stop_index  = plot_stop_ms*10
#
#N0mon_times_n0_plot   = N0mon.spike_trains()[0][N0mon.spike_trains()[0]/ms < plot_stop_ms]
#N0mon_times_n1_plot   = N0mon.spike_trains()[1][N0mon.spike_trains()[1]/ms < plot_stop_ms]
#N0mon_times_n2_plot   = N0mon.spike_trains()[2][N0mon.spike_trains()[2]/ms < plot_stop_ms]
#N0mon_nspikes_n0_plot = np.ones(size(N0mon_times_n0_plot))*0
#N0mon_nspikes_n1_plot = np.ones(size(N0mon_times_n1_plot))*1
#N0mon_nspikes_n2_plot = np.ones(size(N0mon_times_n2_plot))*2
#
#N1mon_times_n0_plot   = N1mon.spike_trains()[0][N1mon.spike_trains()[0]/ms < plot_stop_ms]
#N1mon_nspikes_n0_plot = np.ones(size(N1mon_times_n0_plot))*3
#
#plt.figure()
#
#ax1= plt.subplot(411)
#plt.plot(N0mon_times_n0_plot/ms, N0mon_nspikes_n0_plot, '.k')
#plt.plot(N0mon_times_n1_plot/ms, N0mon_nspikes_n1_plot, '.k')
#plt.plot(N0mon_times_n2_plot/ms, N0mon_nspikes_n2_plot, '.r')
#plt.plot(N1mon_times_n0_plot/ms, N1mon_nspikes_n0_plot, '.b')
#plt.xlabel('Time (ms)')
#plt.ylabel('Neuron index');
#
#plt.subplot(412, sharex = ax1)
#plt.plot(N1state.t[plot_start_index:plot_stop_index]/ms, N1state.v[0][plot_start_index:plot_stop_index], label='N1,0')
#plt.xlabel('Time (ms)')
#plt.ylabel('v')
#
#plt.subplot(413, sharex = ax1)
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[0][plot_start_index:plot_stop_index], label='N1 W 0')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[1][plot_start_index:plot_stop_index], label='N1 W 1')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.w[2][plot_start_index:plot_stop_index], label='N1 W 2')
#plt.legend();
#
#
#plt.subplot(414, sharex = ax1)
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[0][plot_start_index:plot_stop_index], label='N1 W 0')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[1][plot_start_index:plot_stop_index], label='N1 W 1')
#plt.plot(Sstate.t[plot_start_index:plot_stop_index]/ms, Sstate.apre[2][plot_start_index:plot_stop_index], label='N1 W 2')
#plt.legend();


plt.show();
