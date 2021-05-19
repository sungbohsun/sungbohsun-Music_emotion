import os
import torch
import argparse
import warnings
warnings.filterwarnings("ignore")
from model import *
from model_bert import *
from dataloader import *
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import ConcatDataset,TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
    
class MIX(nn.Module):

    def __init__(self,model_cnn,model_lyrics):
        super(MIX, self).__init__()
        
        parms = {'sample_rate':44100,
         'window_size':1024,
         'hop_size':320,
         'mel_bins':64,
         'fmin':50,
         'fmax':14000,
         'classes_num':4 if args.mode == '4Q' else 2}
        
        if args.mode == '4Q':
            model_lyr = eval(model_lyrics+'()')
        else:
            model_lyr = eval(model_lyrics+'_TL()')
        
        if args.model2 == 'BERT':
            self.lyrics = model_lyr.encoder.bert
        
        if args.model2 == 'ALBERT':
            self.lyrics = model_lyr.encoder.albert
        
        self.audio  = eval(model_cnn+'(**parms)')
        self.audio.fc_audioset = layer_pass()
        self.fc1 = nn.Linear(768+512,768+512)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(768+512,4 if args.mode == '4Q' else 2)
    def forward(self,x1,x2):
        
#         for param in self.audio.parameters():
#             param.requires_grad = False
        
#         for param in self.lyrics.parameters():
#             param.requires_grad = False
        
        x2 = torch.tensor(x2, dtype=torch.long)
        out1 = self.audio(x1)['clipwise_output']
        out2 = self.lyrics(x2)['pooler_output']
        out = torch.cat((out1,out2),dim=1)
        out = self.drop(out)
        out = self.fc1(out)
        result = self.fc2(out)
        
        return result
    
def test_class(model):
    model.eval()
    all_prediction = []
    all_label = []
    for batch_idx, (audio,lyrics,target) in enumerate(test_loader):
        target = two_labal(target,args.mode)
        audio = audio.to(device)
        lyrics = lyrics.to(device)
        target = target.to(device)
        output = model(audio,lyrics)
        all_prediction.extend(output.argmax(axis=1).cpu().detach().numpy())
        all_label.extend(target.cpu().detach().numpy())
    
    return all_label,all_prediction


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bt", type=int,default=4 ,help="batch size")
    parser.add_argument("--path",help='weight path')
    args = parser.parse_args()

    #args.path = './model/Lyr/MIX_Cnn6_ALBERT_l_2_pq/best_net.pt'
    args.model1 = args.path.split('/')[-2].split('_')[2]
    args.model2 = args.path.split('/')[-2].split('_')[3]
    
    args.fold = int(args.path.split('/')[2].split('_')[-1][-1])
    args.CV = args.path.split('/')[-2].split('-')[0]
    args.mode = args.path.split('/')[-2].split('_')[1]
        
    data_size_Bi  = 133 
    data_size_Q4  = 479
    data_size_PME = 629
    
    if args.model2 == 'BERT':
        pretrain_tk = 'bert-base-uncased'
    
    elif args.model2 == 'ALBERT':
        pretrain_tk = 'albert-base-v2'
        
    else:
        print('--- use Cnn model without lyrics ---')
        pretrain_tk = 'bert-base-uncased'
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_sets  = []
    #device = 'cpu'
    if args.CV == 'PME' :
        print('CV in PMEmo')
        all_set = PMEmix_dataset(list(range(data_size_PME)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue

            print('now is in fold',fold)
            print('-------- load Lyr_PME')
            PME_test_set = PMEmix_dataset(test_index,pretrain_tk)
            test_sets.append(PME_test_set)

    elif args.CV == 'ALL' :
        print('CV in ALL')
        all_set = PMEmix_dataset(list(range(data_size_PME)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue

            print('now is in fold',fold)
            print('-------- load Lyr_PME')
            PME_test_set = PMEmix_dataset(test_index,pretrain_tk)
            test_sets.append(PME_test_set)

        all_set = Q4mix_dataset(list(range(data_size_Q4)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):
            if fold != args.fold: continue     
            print('-------- load Lyr_Q4')
            Q4_test_set = Q4mix_dataset(test_index,pretrain_tk)               
            test_sets.append(Q4_test_set)

        all_set = Bimix_dataset(list(range(data_size_Bi)),pretrain_tk)
        skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=8848)
        skf.get_n_splits(all_set)

        for fold, (train_index, test_index) in enumerate(skf.split(all_set,[y for _,_,y in all_set])):

            if fold != args.fold: continue
            print('-------- load Bi')
            Bi_test_set = Bimix_dataset(test_index,pretrain_tk)               
            test_sets.append(Bi_test_set)

    test_set = ConcatDataset(test_sets)

    kwargs = {'num_workers': 5, 'pin_memory': True} if device == 'cuda' else {} #needed for using datasets on gpu
    test_loader = DataLoader(test_set, batch_size = args.bt ,shuffle = True, **kwargs)

    parms = {'sample_rate':44100,
     'window_size':1024,
     'hop_size':320,
     'mel_bins':64,
     'fmin':50,
     'fmax':14000,
     'classes_num':4 if args.mode == '4Q' else 2}

    model = MIX(args.model1,args.model2) 
    model.to(device)

    model.load_state_dict(torch.load(args.path))

    y_true, y_pred = test_class(model)
    print('Classification Report:')
    report = classification_report(y_true, y_pred, digits=4)
    print(report)

    dic = {}
    dic['y_true'] = y_true
    dic['y_pred'] = y_pred
    torch.save(dic,'predict/{}{}.pt'.format(args.path.split('/')[-2][:-1],args.fold))

    cm = confusion_matrix(y_true, y_pred,list(range(4 if args.mode == '4Q' else 2)))
    if args.mode =='Ar':
        cm = pd.DataFrame(cm,index=['Arousal +', 'Arousal -']).rename(columns={0: "Arousal +", 1: "Arousal -"})
    if args.mode =='Va':
        cm = pd.DataFrame(cm,index=['Valence +', 'Valence -']).rename(columns={0: "Valence +", 1: "Valence -"})            
    if args.mode =='4Q':
        cm = pd.DataFrame(cm,index=['Arousal +\nValence  +',
                                    'Arousal +\nValence  -',
                                    'Arousal -\nValence  -',
                                    'Arousal -\nValence  +',]).rename(columns={0: "Arousal +\nValence  +", 
                                                                               1: "Arousal +\nValence  -",
                                                                               2: "Arousal -\nValence  -",
                                                                               3: "Arousal -\nValence  +",})                                                              
    fig = plt.figure(figsize=(10,10),dpi=120)
    ax = fig.add_subplot(2,1,1)
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d",annot_kws={'size':16})

    ax.set_title('{}-MIX_{}_{}_{}_fold{} Confusion Matrix'.format(args.CV,args.mode,args.model1,args.model2,args.fold))

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    fig.savefig('png/{}-MIX_{}_{}_{}_fold{}.png'.format(args.CV,args.mode,args.model1,args.model2,args.fold),bbox_inches='tight')

    f = open('res/{}{}.txt'.format(args.path.split('/')[-2][:-1],args.fold),'w+')
    f.write(report)
    f.close()

    dic = {}
    dic['y_true'] = y_true
    dic['y_pred'] = y_pred
    torch.save(dic,'predict/{}{}.pt'.format(args.path.split('/')[-2][:-1],args.fold))

