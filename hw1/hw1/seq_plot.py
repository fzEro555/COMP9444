"""
   seq_plot.py
"""

import argparse
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from seq_models import SRN_model, LSTM_model
from reber import lang_reber
from anbn import lang_anbn
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA
parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, default='reber', help='reber, anbn or anbncn')
parser.add_argument('--embed', type=bool, default=False, help='embedded or not (reber)')
parser.add_argument('--length', type=int, default=0, help='min (reber) or max (anbn)')
# network options
parser.add_argument('--model', type=str, default='srn', help='srn or lstm')
parser.add_argument('--hid', type=int, default=0, help='number of hidden units')
# visualization options
parser.add_argument('--out_path', type=str, default='net', help='outputs path')
parser.add_argument('--epoch', type=int, default=100, help='epoch to load from')
parser.add_argument('--num_plot', type=int, default=10, help='number of plots')
args = parser.parse_args()

if args.lang == 'reber':
    num_class = 7
    hid_default = 2
    lang = lang_reber(args.embed,args.length)
    if args.embed:
        max_state = 18
    else:
        max_state =  6
elif args.lang == 'anbn':
    num_class = 2
    hid_default = 2
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)
    max_state = args.length
elif args.lang == 'anbncn':
    num_class = 3
    hid_default = 3
    if args.length == 0:
        args.length = 8
    lang = lang_anbn(num_class,args.length)
    max_state = args.length

if args.hid == 0:
    args.hid = hid_default
    
if args.model == 'srn':
    net = SRN_model(num_class,args.hid,num_class)
elif args.model == 'lstm':
    net = LSTM_model(num_class,args.hid,num_class)

path = args.out_path+'/'
net.load_state_dict(torch.load(path+'%s_%s%d_%d.pth'
                    %(args.lang,args.model,args.hid,args.epoch)))

np.set_printoptions(suppress=True,precision=2)

for weight in net.parameters():
    print(weight.data.numpy())

#if args.hid == 2:
#    plt.plot(net.H0.data[0],net.H0.data[1],'bx') 
#elif args.hid == 3:    
#    fig = plt.figure()
#    ax = Axes3D(fig)
#    ax.plot(net.H0.data[0],net.H0.data[1],net.H0.data[2],'bx') 
    
with torch.no_grad():
    net.eval()
    meet2 = meet10 = False
    Ret = []
    State = []
    Xfinal = []
    if args.hid > 3:
        fig = plt.figure()
        ax = Axes3D(fig)
                
    for epoch in range(args.num_plot):

        input, seq, target, state = lang.get_sequence()
        label = seq[1:]

        net.init_hidden()
        hidden_seq, output, context_seq = net(input)

        hidden = hidden_seq.squeeze()
        context = context_seq.squeeze()
        print(context)
        ret = lang.print_outputs(epoch, seq, state, hidden, target, output)
        sys.stdout.flush()
        if epoch == args.num_plot - 1 and args.hid <= 3:
            #print(len(hidden))
            if args.hid == 2 and args.model != 'lstm':
                for i in range(0, len(hidden) + 1):
                    plt.clf()
                    plt.plot(net.H0.data[0],net.H0.data[1],'bx')
                    plt.scatter(hidden[:i,0],hidden[:i,1], c=state[1:i+1],
                                cmap='jet', vmin=0, vmax=max_state)
                    for j in range(0, i):
                        plt.annotate(ret[j], (hidden[j][0], hidden[j][1]), textcoords="offset points", xytext=(0,10), ha='center')
                    plt.savefig(args.lang + "_" + str(i))
                plt.clf()
                plt.plot(net.H0.data[0],net.H0.data[1],'bx')
                plt.scatter(hidden[:,0],hidden[:,1], c=state[1:], cmap='jet', vmin=0, vmax=max_state)
                plt.show()
            elif args.hid == 3:
                for i in range(0, len(hidden) + 1):
                    #ax.clear()
                    fig = plt.figure()
                    ax = Axes3D(fig)
                    ax.plot(net.H0.data[0],net.H0.data[1],net.H0.data[2],'bx')                            
                    ax.scatter(hidden[:i,0],hidden[:i,1],hidden[:i,2], c=state[1:i+1], cmap='jet', vmin=0, vmax=max_state)
                    for j in range(0, i):
                        if j != i - 1:
                            ax.text(hidden[j][0],hidden[j][1],hidden[j][2],  ret[j], size=10, zorder=1,  color='k')
                        else:
                            label = "c"
                            if ret[j] == "A":
                                label = "a"
                            if ret[j] == "B":
                                label = "b"
                            ax.text(hidden[j][0],hidden[j][1],hidden[j][2], label, size=10, zorder=1,  color='k')
                    plt.savefig("figure3d_" + str(i))
                    plt.close(fig)
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.plot(net.H0.data[0],net.H0.data[1],net.H0.data[2],'bx')                                
                ax.scatter(hidden[:,0],hidden[:,1],hidden[:,2], c=state[1:], cmap='jet', vmin=0, vmax=max_state)  
                print(state[1:]) 
                plt.show()
                plt.close(fig)
            else:
                plt.clf()
                plt.scatter(hidden[:,0],hidden[:,1], c=state[1:], cmap='jet', vmin=0, vmax=max_state)
                plt.show()
        elif args.hid > 3:
                X = context.numpy()
                #X_embedded = TSNE(n_components=3).fit_transform(X)
                #print(X_embedded)
                if state[2] == 2 and meet2 == False:
                    for v in X:
                        Xfinal.append(v)
                    for v in state[1:]:
                        State.append(v)
                    for v in ret:
                        Ret.append(v)
                    meet2 = True
                    #print(len(Ret), len(State), len(Xfinal))
                if state[2] == 10 and meet10 == False:
                    for v in X:
                        Xfinal.append(v)
                    for v in state[1:]:
                        State.append(v)
                    for v in ret:
                        Ret.append(v)
                    #print(len(Ret), len(State), len(Xfinal))
                    meet10 = True
    if args.hid > 3:
        pca = PCA(n_components=3)
        #tsne = TSNE(n_components=3)
        X_embedded = pca.fit_transform(Xfinal)
        sp = 0
        for i in range(1, len(X_embedded)):
            if State[i] == 1:
                sp = i
        ax.scatter(X_embedded[:,0],X_embedded[:,1], X_embedded[:,2], c=State, cmap='jet', vmin=0, vmax=max_state)
        ax.plot3D(X_embedded[:sp,0],X_embedded[:sp,1], X_embedded[:sp,2], c='green')
        ax.plot3D(X_embedded[sp:,0],X_embedded[sp:,1], X_embedded[sp:,2], c='red')   
        for j in range(0, len(X_embedded)):
            ax.text(X_embedded[j][0], X_embedded[j][1], X_embedded[j][2], str(Ret[j]), size=10, zorder=1,  color='k')
        print(pca.explained_variance_ratio_)
        print(Ret)
        print(State)
        plt.show()
        plt.close(fig)
                