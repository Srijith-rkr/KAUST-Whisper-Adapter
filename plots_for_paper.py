import os 
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

fig = plt.figure(figsize=(10,4))
ax = fig.add_axes([0.1,0.1,0.6,0.6])
#ax.set_xticklabels([500,1000,2000,5000,10000])


df = pd.read_csv('the.csv')

x = [500,1000,2000,5000,10000]
for i in df.columns:
    if i == 'sampes':continue
    df_ = df[i].copy()
    ys = []
    for element in df_:
        
        if isinstance(element, float): element = 50.
            
        if isinstance(element, str) :
            element = element.replace('%','')
            try : element = float(element)
            except: 
                print('skipping',element,'from',i)
                element = 50.
        ys.append(element)
        
        
    markersize_ = 6
    markeredgewidth_ = 2
    markerfacecolor_ = 'gray'
    color_ ='lightblue'
    markeredgecolor ='gray'
    (linestyle_,marker_) = ('dashed','d')
    if 'Input' in i :
        print(i)
    
    
    if 'apter' in i: 
        markersize_ = 8
        color_  = 'limegreen'
        (linestyle_,markersize_) = ('dashed',10)
        if '64' in i: (marker_,markeredgecolor) = ('1','red')
        if '128' in i: (marker_,markeredgecolor) = ('2','red')# was 4 
       # if '256' in i: (marker_,markeredgecolor) = ('2','red')
        if '256' in i: (marker_,markeredgecolor,markerfacecolor_,color_,markersize_,markeredgewidth_) = ('o','black','pink','royalblue',6,2)
        
        
        
    elif'tuning' in i :
        markersize_ = 8
        markeredgewidth_ =1
        color_ = 'black'
        (linestyle_,marker_) = ('dashed','+')
        if 'Full' in i: (marker_,markerfacecolor_,markeredgecolor) = ('^','yellow','black')
        if 'Encoder' in i: (marker_,markerfacecolor_,markeredgecolor) = ('v','yellow','black')
        if 'Decoder' in i: (marker_,markerfacecolor_,markeredgecolor) = ('>','yellow','black')

    
    
    elif 'BitFit' in i :  
        markersize_ = 8
        color_ = 'darkorange'
        (linestyle_,marker_) = ('dashed','x')
        if 'Encoder' in i: (marker_,markerfacecolor_,markeredgecolor) = ('+','royalblue','black')
        elif 'Decoder' in i: (marker_,markerfacecolor_,markeredgecolor) = (2,'royalblue','black')
        else: (marker_,markerfacecolor_,markeredgecolor) = ('x','royalblue','black')
    
    
    
    
    ax.plot(x, ys, label = i, linestyle= linestyle_, marker =marker_ ,markersize=markersize_,markerfacecolor=markerfacecolor_,markeredgecolor=markeredgecolor,markeredgewidth=markeredgewidth_, linewidth=2,color=color_)
ax.legend(loc="center left",bbox_to_anchor=(1.04, 0.5))
# plt.title('Dev set accuracy on utterances over 20s')
plt.ylabel('Accuarcy')
plt.xlabel('Number of samples')
plt.show()