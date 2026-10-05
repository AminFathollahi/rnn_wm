"""Render the four explanatory figures used by PROJECT_IMPLEMENTATION_STATUS.md.

Run with: MPLCONFIGDIR=/tmp/mpl python docs/tutorial_assets/make_tutorial_figures.py
The figures are conceptual maps of the implemented pipeline; they contain no data.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent
BG, INK, BLUE, TEAL, GOLD, CORAL, PALE = '#fbfcfe', '#172033', '#3d6fb6', '#3a9d91', '#d99a2b', '#cf6260', '#eef3f8'

def box(ax, x, y, w, h, text, color=BLUE, fs=9, **kw):
    p = FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025', ec=color, fc='white', lw=1.4, **kw)
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,wrap=True)
    return p
def arrow(ax, a, b, color=INK, label=None, off=(0,0)):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.25,color=color,connectionstyle='arc3,rad=0.0'))
    if label: ax.text((a[0]+b[0])/2+off[0],(a[1]+b[1])/2+off[1],label,fontsize=7.5,color=INK,ha='center',va='center')
def setup(ax, title, sub):
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off'); ax.set_facecolor(BG)
    ax.text(.02,.96,title,fontsize=15,weight='bold',color=INK,va='top'); ax.text(.02,.90,sub,fontsize=8.5,color='#526176',va='top')
def finish(fig, name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor=BG); fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight',facecolor=BG); plt.close(fig)

def paths():
    fig,ax=plt.subplots(figsize=(11,5.7)); setup(ax,'Two trial paths: learn first, replay later','The human recording is a comparison target. It is never an input to the training loss.')
    ax.text(.06,.80,'TRAINING',weight='bold',fontsize=10,color=BLUE); ax.text(.06,.38,'FROZEN REPLAY',weight='bold',fontsize=10,color=TEAL)
    box(ax,.06,.57,.17,.14,'Natural-image bank\n+generated Sternberg trial',BLUE); box(ax,.31,.57,.15,.14,'Image + cue\nsequence',BLUE); box(ax,.54,.57,.15,.14,'RNN forward\npass',BLUE); box(ax,.77,.57,.16,.14,'Actions + value\nlabels / rewards',GOLD)
    arrow(ax,(.23,.64),(.31,.64)); arrow(ax,(.46,.64),(.54,.64)); arrow(ax,(.69,.64),(.77,.64)); arrow(ax,(.85,.57),(.62,.46),GOLD,'learning signal',(-.02,.02)); ax.text(.61,.41,'update model parameters',fontsize=8,color=GOLD,weight='bold')
    box(ax,.06,.15,.18,.14,'Human recorded trial\nimage identities + timing',TEAL); box(ax,.31,.15,.15,.14,'Same model input\nsequence',TEAL); box(ax,.54,.15,.15,.14,'Frozen model\nforward pass',TEAL); box(ax,.77,.15,.16,.14,'Saved model\nstates',TEAL)
    arrow(ax,(.24,.22),(.31,.22)); arrow(ax,(.46,.22),(.54,.22)); arrow(ax,(.69,.22),(.77,.22)); ax.text(.53,.34,'no parameter updates',fontsize=8,color=TEAL,weight='bold')
    box(ax,.77,.36,.16,.10,'Recorded spikes\n→ firing rates',CORAL,8); arrow(ax,(.85,.36),(.85,.29),CORAL); ax.text(.70,.35,'compare matched\nrepresentations',fontsize=8,ha='right',va='center',color=INK,weight='bold')
    finish(fig,'01_training_and_replay_paths')

def architecture():
    fig,ax=plt.subplots(figsize=(12,5.9)); setup(ax,'From image to recurrent state','Counts describe state dimensions. A current-configuration audit gives 73,728 flat and 74,052 hierarchical structural synapses.')
    box(ax,.03,.49,.11,.16,'Image\n224 × 224',BLUE); box(ax,.18,.49,.13,.16,'Frozen\nResNet-18',BLUE); box(ax,.35,.49,.12,.16,'Visual feature\n512',BLUE); box(ax,.51,.70,.12,.10,'Task cue\n10',GOLD); box(ax,.51,.49,.12,.16,'concat\n522',GOLD); box(ax,.67,.49,.12,.16,'linear 64\nLayerNorm',BLUE); box(ax,.83,.49,.12,.16,'linear 64\n+ noise ε',BLUE)
    for a,b in [((.14,.57),(.18,.57)),((.31,.57),(.35,.57)),((.47,.57),(.51,.57)),((.63,.57),(.67,.57)),((.79,.57),(.83,.57))]: arrow(ax,a,b)
    arrow(ax,(.57,.70),(.57,.65),GOLD)
    ax.text(.89,.41,'recurrent input zₜ',fontsize=8,ha='center',color='#526176')
    box(ax,.11,.12,.23,.18,'Flat baseline\n128 recurrent units',BLUE,10); box(ax,.47,.12,.23,.18,'Sparse worker\n196 units on 14 × 14 grid',TEAL,10); box(ax,.76,.12,.16,.18,'Manager\n24 units\nupdates every 5 steps',TEAL,9)
    arrow(ax,(.89,.49),(.23,.30),BLUE); arrow(ax,(.89,.49),(.58,.30),TEAL); arrow(ax,(.89,.49),(.84,.30),TEAL)
    ax.text(.11,.05,'state count: 128',fontsize=8,color='#526176'); ax.text(.47,.05,'state count: 196 + 24',fontsize=8,color='#526176'); ax.text(.76,.05,'configuration audit: 73,728 vs 74,052 structural synapses',fontsize=8,color='#526176')
    ax.text(.03,.83,'Only image pixels enter the frozen ResNet. The cue joins after visual encoding.',fontsize=9,color=INK)
    finish(fig,'02_input_and_recurrent_architectures')

def timescales():
    fig,ax=plt.subplots(figsize=(11,5.6)); setup(ax,'Three quantities with different lifetimes','State and fast trace reset with the trial; learned parameters persist and change through optimization.')
    for y,label,col in [(.68,'Recurrent state hₜ',BLUE),(.43,'Fast-weight trace Aₜ',TEAL),(.18,'Slow weights θ',GOLD)]:
        ax.text(.05,y+0.06,label,fontsize=11,weight='bold',color=col); ax.plot([.30,.92],[y,y],color='#b8c4d2',lw=2)
    ax.plot([.31,.44,.54,.66,.78,.90],[.68,.76,.61,.74,.64,.71],color=BLUE,lw=2); ax.text(.31,.57,'updated each model step; reset at the next trial',fontsize=8.5,color=INK)
    ax.plot([.31,.45,.58,.72,.86,.90],[.43,.46,.40,.44,.42,.43],color=TEAL,lw=2); ax.text(.31,.32,'Hebbian trace evolves within a trial; reset at the next trial',fontsize=8.5,color=INK)
    ax.plot([.31,.43,.55,.67,.79,.90],[.18,.18,.19,.19,.20,.20],color=GOLD,lw=2); ax.text(.31,.07,'learned parameters persist; optimizer updates them between batches',fontsize=8.5,color=INK)
    for x,t in [(.31,'trial start'),(.60,'within trial'),(.90,'next trial')]: ax.axvline(x,color='#d9e0e8',lw=1); ax.text(x,.88,t,ha='center',fontsize=8,color='#526176')
    box(ax,.70,.78,.20,.08,'optimizer step\n(no replay learning)',GOLD,8)
    finish(fig,'03_state_weights_and_trace_timescales')

def aggregation():
    fig,ax=plt.subplots(figsize=(12,5.7)); setup(ax,'Aggregation choices define the comparison','This diagram names the array shapes and two valid routes; it intentionally shows no outcome.')
    box(ax,.04,.50,.18,.16,'Trial × time × units\nmodel states / firing rates',BLUE,10); box(ax,.29,.50,.17,.16,'Epoch mean\nper trial × units',BLUE,10); box(ax,.55,.70,.17,.16,'Within-session\ncondition means',TEAL,10); box(ax,.55,.25,.17,.16,'Pooled probe\nrows / units',GOLD,10); box(ax,.79,.70,.16,.16,'Condition RDM\nper session',TEAL,10); box(ax,.79,.25,.16,.16,'Pseudopopulation\nRDM',GOLD,10)
    arrow(ax,(.22,.58),(.29,.58),'#172033','epoch average',(0,.05)); arrow(ax,(.46,.58),(.55,.78),TEAL,'group trials',(0,.04)); arrow(ax,(.46,.58),(.55,.33),GOLD,'select / pool',(0,-.04)); arrow(ax,(.72,.78),(.79,.78),TEAL,'distances'); arrow(ax,(.72,.33),(.79,.33),GOLD,'distances')
    ax.text(.56,.49,'conditions: load, epoch, probe status',fontsize=7.5,ha='center',color='#526176'); ax.text(.87,.52,'compare matched\nrepresentational geometry',fontsize=9,ha='center',va='center',color=INK,weight='bold')
    finish(fig,'04_rdm_aggregation_routes')

if __name__ == '__main__':
    paths(); architecture(); timescales(); aggregation()
